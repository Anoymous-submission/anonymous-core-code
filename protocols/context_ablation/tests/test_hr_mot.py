"""Regression tests for the H&R MoT rewrite.

Each test here corresponds to something the audit of the previous run found
wrong, so a future edit that reintroduces the failure fails a test instead of
silently burning 45 minutes of GPU time.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fasterwam.datasets.hr_raw_stream import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    ACTION_HORIZON,
    FRAME_STRIDE,
    HUMAN_ACTION_DIM,
    HUMAN_ACTION_SOURCE,
    NUM_RGB_FRAMES,
    RawHREpisode,
    RawHRStreamingDataset,
    StateStats,
)
from fasterwam.models.hr_mot import (
    ROLE_HUMAN_CONTEXT,
    ROLE_ROBOT_CURRENT,
    ROLE_ROBOT_FUTURE,
    HRMoTConfig,
    HRMoTFlowModel,
    build_availability_attention_mask,
    build_condition_prediction_mask,
    build_linear_attention_groups,
    count_parameters,
)
from fasterwam.models.hr_eval import (
    HR_MOT_SCHEMA_VERSION,
    canonical_sha256,
    dataset_geometry,
    different_instruction_indices,
    future_pixel_mse,
    human_context_metadata,
    validate_checkpoint_run_pair,
)
from fasterwam.models.hr_inference import build_human_mask, generate_rollout


def _fake_record(name: str, group: str, frames: int = 400) -> RawHREpisode:
    import pathlib

    return RawHREpisode(
        path=pathlib.Path(f"/{name}.hdf5"),
        annotation_path=name,
        instruction=group,
        group_key=group,
        task="t",
        needs_review=False,
        clean_behavior_eligible=True,
        trajectory_quality=None,
        human_frames=frames,
        robot_frames=frames,
    )


def tiny_config(**overrides) -> HRMoTConfig:
    base = dict(
        latent_height=4,
        latent_width=6,
        num_rgb_frames=9,
        frame_stride=2,
        action_horizon=16,
        num_layers=2,
        num_heads=2,
        attn_head_dim=128,
        video_hidden_dim=64,
        video_ffn_dim=128,
        action_hidden_dim=32,
        action_ffn_dim=64,
        gradient_checkpointing=False,
    )
    base.update(overrides)
    return HRMoTConfig(**base)


def make_batch(config: HRMoTConfig, batch: int = 2):
    shape = (config.latent_channels, config.latent_height, config.latent_width)
    return {
        "robot_current_latent": torch.randn(batch, 1, *shape),
        "human_latents": torch.randn(batch, config.num_latent_steps, *shape),
        "robot_future_latents": torch.randn(batch, config.num_horizons, *shape),
        "robot_future_action": torch.randn(batch, config.num_action_tokens, config.action_dim),
        "robot_current_state": torch.randn(batch, config.action_dim),
        "human_future_action": torch.randn(
            batch, config.num_action_tokens, config.human_action_dim
        ),
    }


# --- D. Data and target correctness -----------------------------------------


def test_segment_uses_the_vae_temporal_contract() -> None:
    """FastWAM encodes the ordered segment once so the VAE also compresses time:
    9 frames -> 3 latent steps, z0<-f0, z1<-f1..4, z2<-f5..8."""
    assert NUM_RGB_FRAMES == 9
    config = HRMoTConfig()
    assert config.num_rgb_frames == 9
    assert config.num_latent_steps == 3
    assert config.num_horizons == 2  # predicted robot latent steps


def test_video_is_downsampled_by_stride_four() -> None:
    """9 CONTIGUOUS frames span 8 control steps = 0.27 s at 29.4 fps, in which
    the gripper changes in only 6.4% of windows. FastWAM keeps every 4th of 33
    control steps, so the same 9 frames span 32 steps = 1.09 s."""
    assert FRAME_STRIDE == 4
    config = HRMoTConfig()
    assert config.frame_stride == 4
    assert config.video_span == 32
    # latent step positions are CONTROL steps, not video-frame indices
    assert config.latent_step_frames == (0, 16, 32)
    assert config.frame_rope_positions == (0, 0, 16, 32, 16, 32)


def test_action_horizon_is_thirty_two_contiguous_steps() -> None:
    assert ACTION_HORIZON == 32
    config = HRMoTConfig()
    assert config.num_action_tokens == 32
    assert config.action_horizon == config.video_span
    # A predicted state shares its RoPE position with human motion at the same step.
    positions = config.action_rope_positions
    assert positions[0] == 0  # robot anchor at t0
    assert positions[1:33] == tuple(range(1, 33))  # human-motion conditions
    assert positions[33:] == tuple(range(1, 33))  # predicted states


def test_action_horizon_must_cover_the_video_span() -> None:
    with pytest.raises(ValueError, match="action_horizon must equal"):
        HRMoTConfig(action_horizon=8).validate()


def test_seven_dimensional_robot_setpoint_cannot_be_human_context() -> None:
    with pytest.raises(ValueError, match="human motion must be 84-D"):
        HRMoTConfig(human_action_dim=7).validate()


def test_human_action_tokens_follow_their_latent_step() -> None:
    """Masking a human latent step must also mask the action tokens covering the
    same real time, or the mask leaks through the action branch."""
    config = HRMoTConfig()
    mapping = config.human_action_latent_step()
    assert len(mapping) == 32
    assert set(mapping[:16]) == {1}  # control steps 1..16 -> latent step 1
    assert set(mapping[16:]) == {2}  # control steps 17..32 -> latent step 2


def test_sequence_layout_matches_the_specified_design() -> None:
    config = HRMoTConfig()
    # robot z0 + human z0,z1,z2 + robot z1,z2 = 6 latent-step slots
    assert config.num_video_frames == 6
    assert config.num_condition_frames == 4
    assert config.frame_roles == (
        ROLE_ROBOT_CURRENT,
        ROLE_HUMAN_CONTEXT,
        ROLE_HUMAN_CONTEXT,
        ROLE_HUMAN_CONTEXT,
        ROLE_ROBOT_FUTURE,
        ROLE_ROBOT_FUTURE,
    )
    # A predicted robot latent step shares its temporal position with the human
    # step covering the same RGB frames; the role embedding separates them.
    assert config.frame_rope_positions == (0, 0, 16, 32, 16, 32)


def test_linear_attention_type_is_explicit_and_validated() -> None:
    assert HRMoTConfig().mixed_attention_type == "softmax"
    HRMoTConfig(mixed_attention_type="linear").validate()
    with pytest.raises(ValueError, match="mixed_attention_type"):
        HRMoTConfig(mixed_attention_type="unknown").validate()


def test_direct_fastwam_robot_only_sequence_contract() -> None:
    config = HRMoTConfig(
        direct_fastwam_robot_only=True,
        use_human_video_context=False,
        use_human_motion_context=False,
    )
    config.validate()
    assert config.num_video_frames == 3
    assert config.num_video_tokens == 252
    assert config.num_condition_tokens == 84
    assert config.frame_rope_positions == (0, 16, 32)
    assert config.num_action_condition_tokens == 1
    assert config.num_action_sequence == 33
    assert config.action_rope_positions == tuple(range(33))
    assert config.num_video_tokens + config.num_action_sequence == 285


def test_direct_fastwam_rejects_human_or_linear_context() -> None:
    with pytest.raises(ValueError, match="human context sources off"):
        HRMoTConfig(direct_fastwam_robot_only=True).validate()
    with pytest.raises(ValueError, match="original softmax"):
        HRMoTConfig(
            direct_fastwam_robot_only=True,
            use_human_video_context=False,
            use_human_motion_context=False,
            mixed_attention_type="linear",
        ).validate()


def test_direct_fastwam_attention_graph_is_exact() -> None:
    config = tiny_config(
        direct_fastwam_robot_only=True,
        use_human_video_context=False,
        use_human_motion_context=False,
    )
    human_mask = torch.ones(1, config.num_latent_steps, dtype=torch.bool)
    mask = build_availability_attention_mask(config, human_mask)[0, 0]
    current = config.tokens_per_frame
    video_end = config.num_video_tokens
    state = video_end
    actions = state + 1
    assert mask[:current, :current].all()
    assert mask[:current, state].all()
    assert not mask[:current, current:video_end].any()
    assert not mask[:current, actions:].any()
    assert mask[current:video_end, :video_end].all()
    assert mask[current:video_end, state].all()
    assert not mask[current:video_end, actions:].any()
    assert mask[state, :current].all() and mask[state, state]
    assert not mask[state, current:video_end].any()
    assert not mask[state, actions:].any()
    assert mask[actions:, :current].all()
    assert mask[actions:, state:].all()
    assert not mask[actions:, current:video_end].any()


def test_direct_fastwam_physically_omits_human_modules_and_inputs() -> None:
    torch.manual_seed(0)
    config = tiny_config(
        direct_fastwam_robot_only=True,
        use_human_video_context=False,
        use_human_motion_context=False,
    )
    model = HRMoTFlowModel(config).eval()
    assert not hasattr(model.video_expert, "availability_embedding")
    assert not hasattr(model.action_expert, "availability_embedding")
    assert not hasattr(model.action_expert, "human_action_embedding")
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)

    def run(human_latents, human_action):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=human_latents,
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                robot_current_state=batch["robot_current_state"],
                human_future_action=human_action,
            )

    first = run(batch["human_latents"], batch["human_future_action"])
    second = run(batch["human_latents"] + 1000, batch["human_future_action"] - 1000)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# --- C. Real DiT, not a one-token-per-frame bottleneck -----------------------


def test_spatial_tokens_survive_into_the_transformer() -> None:
    """The old model pooled each frame to ONE token. That was the core defect."""
    config = HRMoTConfig()
    assert config.tokens_per_frame == 84
    assert config.num_video_tokens == 504
    assert config.num_condition_tokens == 336
    # 1 robot anchor + 32 human-motion conditions + 32 predicted robot states
    assert config.num_action_condition_tokens == 33
    assert config.num_action_sequence == 65
    assert config.num_video_tokens + config.num_action_sequence == 569


def test_every_block_is_timestep_modulated() -> None:
    """AdaLN per block, not a single timestep added once at the input."""
    model = HRMoTFlowModel(tiny_config())
    for expert in (model.video_expert, model.action_expert):
        for block in expert.blocks:
            assert block.modulation.shape == (1, 6, expert.hidden_dim)
        assert expert.time_projection[-1].out_features == expert.hidden_dim * 6


def test_parameter_count_is_about_zero_point_six_billion() -> None:
    model = HRMoTFlowModel(HRMoTConfig())
    total = count_parameters(model)
    assert total == 592_926_279


def test_state_dict_has_no_duplicated_experts() -> None:
    """MoT owns the experts; registering them twice would double the checkpoint."""
    model = HRMoTFlowModel(tiny_config())
    keys = list(model.state_dict())
    assert all(key.startswith("mot.mixtures.") for key in keys), keys[:5]


# --- Attention visibility ----------------------------------------------------


def test_mask_isolates_condition_from_prediction() -> None:
    mask = build_condition_prediction_mask(
        num_condition_tokens=4, num_video_tokens=6, num_action_tokens=2
    )
    assert mask.shape == (8, 8)
    assert mask[:4, :4].all()
    assert not mask[:4, 4:].any()  # condition never reads prediction
    assert mask[4:, :].all()  # prediction reads everything


def test_condition_tokens_do_not_depend_on_the_noisy_future() -> None:
    """Functional proof of the mask: gradient from condition outputs to the
    noisy robot-future input must be exactly zero."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    noisy = batch["robot_future_latents"].clone().requires_grad_(True)
    noisy_action = batch["robot_future_action"].clone().requires_grad_(True)
    human = batch["human_latents"].clone().requires_grad_(True)
    timestep = torch.full((1,), 500.0)

    tokens_out, _, _ = model.forward_tokens(
        robot_current_latent=batch["robot_current_latent"],
        human_latents=human,
        noisy_robot_future_latents=noisy,
        noisy_robot_future_action=noisy_action,
        timestep_video=timestep,
        timestep_action=timestep,
        robot_current_state=batch["robot_current_state"],
        human_future_action=batch["human_future_action"],
    )
    condition_out = tokens_out["video"][:, : config.num_condition_tokens, :]
    grads = torch.autograd.grad(
        condition_out.sum(), [noisy, noisy_action], allow_unused=True, retain_graph=True
    )
    for grad in grads:
        assert grad is None or float(grad.abs().sum()) == 0.0

    # ...and the prediction side really does read the human condition.
    prediction_out = tokens_out["video"][:, config.num_condition_tokens :, :]
    human_grad = torch.autograd.grad(prediction_out.sum(), human, allow_unused=True)[0]
    assert human_grad is not None and float(human_grad.abs().sum()) > 0.0


def test_prediction_depends_on_human_future_frames() -> None:
    """The whole point of the design: change the human ground-truth future and
    the robot-future prediction must change."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)

    def run(human):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=human,
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                robot_current_state=batch["robot_current_state"],
                human_future_action=batch["human_future_action"],
            )

    video_a, action_a = run(batch["human_latents"])
    perturbed = batch["human_latents"].clone()
    perturbed[:, 1:] += 1.0  # only the human FUTURE frames
    video_b, action_b = run(perturbed)
    assert not torch.allclose(video_a, video_b)
    assert not torch.allclose(action_a, action_b)


# --- E. Video learning must be possible --------------------------------------


def test_gradient_reaches_every_parameter() -> None:
    """A zero-initialized head silently starves all upstream tensors on step 1.

    Uses a mixed availability mask so both the available and unavailable rows of
    the availability embedding are exercised.
    """
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config)
    human_mask = torch.tensor([[True, True, False], [True, False, True]])
    loss, _ = model.training_loss(**make_batch(config), human_mask=human_mask)
    loss.backward()
    dead = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not dead, dead[:8]


def test_direct_fastwam_gradient_reaches_every_retained_parameter() -> None:
    torch.manual_seed(0)
    config = tiny_config(
        direct_fastwam_robot_only=True,
        use_human_video_context=False,
        use_human_motion_context=False,
    )
    model = HRMoTFlowModel(config)
    loss, _ = model.training_loss(**make_batch(config))
    loss.backward()
    dead = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not dead, dead[:8]


def test_loss_decreases_when_overfitting_one_batch() -> None:
    """The old video branch was flat for 10,000 steps. This one must move."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config)
    batch = make_batch(config, batch=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(99)  # fixed noise and timestep -> a clean signal
        generator = torch.Generator().manual_seed(99)
        loss, parts = model.training_loss(**batch, generator=generator)
        loss.backward()
        optimizer.step()
        value = float(parts["loss_video_unweighted"])
        first = value if step == 0 else first
        last = value
    assert last < first * 0.95, (first, last)


# --- Variable human context ---


def test_linear_groups_exactly_reconstruct_the_canonical_visibility() -> None:
    config = tiny_config(mixed_attention_type="linear")
    human_mask = torch.tensor([[True, False, True], [False, False, False]])
    mask = build_availability_attention_mask(config, human_mask)
    groups = build_linear_attention_groups(config, mask)
    condition = groups["condition"]
    prediction = groups["prediction"]
    missing = groups["missing"]
    reconstructed = condition[:, :, None] & condition[:, None, :]
    reconstructed |= prediction[:, :, None] & (condition | prediction)[:, None, :]
    reconstructed |= missing[:, :, None] & torch.eye(mask.shape[-1], dtype=torch.bool)[None]
    assert torch.equal(reconstructed[:, None], mask)
    assert torch.all(
        condition.to(torch.int8) + prediction.to(torch.int8) + missing.to(torch.int8) == 1
    )


def test_linear_rope_kernel_matches_explicit_quadratic_reference() -> None:
    torch.manual_seed(7)
    config = tiny_config(mixed_attention_type="linear")
    mot = HRMoTFlowModel(config).mot
    batch = 2
    seq = config.num_video_tokens + config.num_action_sequence
    packed = mot.num_heads * mot.attn_head_dim
    human_mask = torch.tensor([[True, False, True], [False, True, False]])
    groups = build_linear_attention_groups(
        config, build_availability_attention_mask(config, human_mask)
    )
    q_num = torch.randn(batch, seq, packed)
    k_num = torch.randn(batch, seq, packed)
    # The denominator feature maps are positive and intentionally unrotated.
    q_den = torch.rand(batch, seq, packed) + 0.1
    k_den = torch.rand(batch, seq, packed) + 0.1
    value = torch.randn(batch, seq, packed)

    actual = mot._linear_mixed_attention(q_num, k_num, value, q_den, k_den, groups)
    heads, dim = mot.num_heads, mot.attn_head_dim
    qn = q_num.reshape(batch, seq, heads, dim)
    kn = k_num.reshape(batch, seq, heads, dim)
    qd = q_den.reshape(batch, seq, heads, dim)
    kd = k_den.reshape(batch, seq, heads, dim)
    vv = value.reshape(batch, seq, heads, dim)
    expected = torch.empty_like(vv)
    for b in range(batch):
        for query in range(seq):
            if groups["condition"][b, query]:
                readable = groups["condition"][b]
            elif groups["prediction"][b, query]:
                readable = groups["condition"][b] | groups["prediction"][b]
            else:
                expected[b, query] = vv[b, query]
                continue
            scores = torch.einsum("hd,khd->hk", qn[b, query], kn[b, readable])
            numerator = torch.einsum("hk,khd->hd", scores, vv[b, readable])
            denominator = torch.einsum("hd,hd->h", qd[b, query], kd[b, readable].sum(dim=0))
            expected[b, query] = numerator / denominator[:, None].clamp_min(1e-6)
    torch.testing.assert_close(actual, expected.reshape(batch, seq, packed))


def test_linear_attention_hard_mask_blocks_all_human_information() -> None:
    torch.manual_seed(0)
    config = tiny_config(mixed_attention_type="linear")
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    no_human = torch.zeros((1, config.num_latent_steps), dtype=torch.bool)

    def run(human_latents, human_motion):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=human_latents,
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                human_mask=no_human,
                robot_current_state=batch["robot_current_state"],
                human_future_action=human_motion,
            )

    original = run(batch["human_latents"], batch["human_future_action"])
    changed = run(
        batch["human_latents"] + 100.0,
        batch["human_future_action"] - 100.0,
    )
    assert torch.equal(original[0], changed[0])
    assert torch.equal(original[1], changed[1])


def test_linear_attention_with_human_information_affects_both_predictions() -> None:
    torch.manual_seed(0)
    config = tiny_config(mixed_attention_type="linear")
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    full = torch.ones((1, config.num_latent_steps), dtype=torch.bool)

    def run(human_latents, human_motion):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=human_latents,
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                human_mask=full,
                robot_current_state=batch["robot_current_state"],
                human_future_action=human_motion,
            )

    original = run(batch["human_latents"], batch["human_future_action"])
    changed = run(
        batch["human_latents"] + 1.0,
        batch["human_future_action"] - 1.0,
    )
    assert not torch.allclose(original[0], changed[0])
    assert not torch.allclose(original[1], changed[1])


def test_linear_attention_backward_is_finite_and_keeps_parameter_count() -> None:
    torch.manual_seed(0)
    softmax_config = tiny_config(mixed_attention_type="softmax")
    linear_config = tiny_config(mixed_attention_type="linear")
    softmax_model = HRMoTFlowModel(softmax_config)
    linear_model = HRMoTFlowModel(linear_config)
    assert count_parameters(softmax_model) == count_parameters(linear_model)
    mask = torch.tensor([[True, True, False], [False, True, True]])
    loss, _ = linear_model.training_loss(**make_batch(linear_config), human_mask=mask)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in linear_model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_inference_mask_can_drop_all_human_context() -> None:
    mask = build_human_mask(3, batch=2, keep=[])
    assert mask.shape == (2, 3)
    assert not bool(mask.any())


def test_inference_mask_exactly_keeps_requested_steps() -> None:
    mask = build_human_mask(3, batch=1, keep=[1])
    assert mask.tolist() == [[False, True, False]]


def test_inference_mask_pins_current_for_legacy_models() -> None:
    mask = build_human_mask(3, batch=1, keep=[], pin_current=True)
    assert mask.tolist() == [[True, False, False]]


def test_masked_human_frames_carry_no_episode_information() -> None:
    """A dropped human frame must make the prediction independent of what that
    frame actually contained."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    # keep only the human current frame; drop all human future frames
    mask = torch.tensor([[True] + [False] * (config.num_latent_steps - 1)])

    def run(human):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=human,
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                human_mask=mask,
                robot_current_state=batch["robot_current_state"],
                human_future_action=batch["human_future_action"],
            )

    a_video, a_action = run(batch["human_latents"])
    perturbed = batch["human_latents"].clone()
    perturbed[:, 1:] += 5.0  # scramble exactly the dropped frames
    b_video, b_action = run(perturbed)
    assert torch.allclose(a_video, b_video, atol=1e-5)
    assert torch.allclose(a_action, b_action, atol=1e-5)


def test_unmasked_human_frames_still_matter() -> None:
    """The mask must not be a no-op in the other direction."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    full = torch.ones((1, config.num_latent_steps), dtype=torch.bool)
    partial = torch.tensor([[True] + [False] * (config.num_latent_steps - 1)])

    def run(mask):
        with torch.no_grad():
            return model.predict_velocity(
                robot_current_latent=batch["robot_current_latent"],
                human_latents=batch["human_latents"],
                noisy_robot_future_latents=batch["robot_future_latents"],
                noisy_robot_future_action=batch["robot_future_action"],
                timestep_video=timestep,
                timestep_action=timestep,
                human_mask=mask,
                robot_current_state=batch["robot_current_state"],
                human_future_action=batch["human_future_action"],
            )[0]

    assert not torch.allclose(run(full), run(partial))


def test_human_current_step_is_droppable_by_default() -> None:
    """Pinning the human z0 step leaves the no-human path untrained (its
    availability embedding row never gets gradient) and lets the model lean on
    the same-moment correspondence instead of reading the demonstration."""
    assert HRMoTConfig().droppable_human_current is True
    model = HRMoTFlowModel(tiny_config(full_human_context_prob=0.9))
    mask = model.sample_human_mask(60000, device=torch.device("cpu"))
    full = mask.all(dim=1).float().mean().item()
    assert abs(full - 0.9) < 0.02, full
    # every pattern must be reachable, including "no human context at all",
    # otherwise that condition stays out of distribution
    patterns = {tuple(r) for r in mask.tolist()}
    assert len(patterns) == 2 ** tiny_config().num_latent_steps, sorted(patterns)
    assert tuple([False] * tiny_config().num_latent_steps) in patterns
    dropped = (~mask[:, 0]).float().mean().item()
    assert 0.03 < dropped < 0.09, dropped


def test_non_full_patterns_are_drawn_uniformly() -> None:
    """The 10% is a regularizer: it must not concentrate on a few patterns."""
    model = HRMoTFlowModel(tiny_config(full_human_context_prob=0.0))
    mask = model.sample_human_mask(70000, device=torch.device("cpu"))
    from collections import Counter

    counts = Counter(tuple(r) for r in mask.tolist())
    slots = tiny_config().num_latent_steps
    assert len(counts) == 2**slots - 1  # all-ones excluded from this branch
    share = [v / 70000 for v in counts.values()]
    expected = 1.0 / (2**slots - 1)
    assert max(abs(x - expected) for x in share) < 0.01, sorted(share)


def test_pinning_the_human_current_step_disables_that_dropout() -> None:
    model = HRMoTFlowModel(tiny_config(droppable_human_current=False))
    mask = model.sample_human_mask(8000, device=torch.device("cpu"))
    assert bool(mask[:, 0].all())
    patterns = {tuple(r) for r in mask.tolist()}
    assert len(patterns) == 2 ** (tiny_config().num_latent_steps - 1)


def test_pinned_mode_covers_every_future_pattern() -> None:
    config = tiny_config(full_human_context_prob=0.5, droppable_human_current=False)
    model = HRMoTFlowModel(config)
    mask = model.sample_human_mask(8192, device=torch.device("cpu"))
    assert bool(mask[:, 0].all()), "human z0 must always be supplied when pinned"
    patterns = {tuple(row) for row in mask[:, 1:].tolist()}
    assert len(patterns) == 2 ** (config.num_latent_steps - 1), patterns


def test_full_context_share_matches_the_configured_probability() -> None:
    """Context dropout is a regularizer: the default must keep full context on
    the large majority of samples, or it weakens the pathway under study."""
    assert HRMoTConfig().full_human_context_prob == 0.9
    for probability in (0.5, 0.9):
        model = HRMoTFlowModel(
            tiny_config(full_human_context_prob=probability, droppable_human_current=False)
        )
        mask = model.sample_human_mask(20000, device=torch.device("cpu"))
        full_share = mask.all(dim=1).float().mean().item()
        # The non-full branch excludes the all-ones pattern, so the full share is
        # the configured probability itself, up to sampling noise.
        assert abs(full_share - probability) < 0.03, (probability, full_share)


def test_masking_is_whole_frame_never_scattered_patches() -> None:
    config = tiny_config()
    # human z0, z1 supplied; z2 missing. Human step 2 is video slot 3.
    human_mask = torch.tensor([[True, True, False]])
    mask = build_availability_attention_mask(config, human_mask)
    tpf = config.tokens_per_frame
    missing = slice(3 * tpf, 4 * tpf)
    keys_into_missing = mask[0, 0, :, missing]
    # nothing may read a missing frame except those tokens themselves (diagonal)
    off_diagonal = keys_into_missing.clone()
    off_diagonal[missing] &= ~torch.eye(tpf, dtype=torch.bool)
    assert not bool(off_diagonal.any())
    # and the whole frame is dropped together, not part of it
    assert int(keys_into_missing.sum()) == tpf


def test_rope_positions_are_not_renumbered_when_frames_are_missing() -> None:
    """The step covering control steps 17-32 keeps position 32 even if the step
    covering 1-16 is absent."""
    config = HRMoTConfig()
    assert config.frame_rope_positions == (0, 0, 16, 32, 16, 32)
    model = HRMoTFlowModel(tiny_config())
    freqs_all = model.video_expert.rope_freqs(torch.device("cpu"))
    # masking changes token values, never the frequency table
    assert freqs_all.shape[0] == tiny_config().num_video_tokens


def test_prediction_never_reads_a_missing_human_frame() -> None:
    config = tiny_config()
    human_mask = torch.tensor([[True, False, True]])  # human z1 missing
    mask = build_availability_attention_mask(config, human_mask)[0, 0]
    tpf = config.tokens_per_frame
    # Prediction rows are the robot-future video steps and the predicted states.
    # The action-side CONDITION block sits between them and is not a prediction:
    # its masked human-action tokens are diagonal-only rows.
    video_pred = torch.arange(config.num_condition_tokens, config.num_video_tokens)
    action_pred = torch.arange(
        config.num_video_tokens + config.num_action_condition_tokens, mask.shape[0]
    )
    rows = torch.cat((video_pred, action_pred))
    assert not bool(mask[rows][:, 2 * tpf : 3 * tpf].any())  # human z1 unreadable
    for slot in (0, 1, 3):  # robot z0, human z0, human z2 stay readable
        assert bool(mask[rows][:, slot * tpf : (slot + 1) * tpf].all())


def test_action_condition_tokens_are_conditions_not_predictions() -> None:
    """The robot anchor and human motion are clean conditions: they must not
    be able to read the noisy predicted states."""
    config = tiny_config()
    human_mask = torch.ones((1, config.num_latent_steps), dtype=torch.bool)
    mask = build_availability_attention_mask(config, human_mask)[0, 0]
    cond = torch.arange(
        config.num_video_tokens,
        config.num_video_tokens + config.num_action_condition_tokens,
    )
    pred = torch.arange(config.num_video_tokens + config.num_action_condition_tokens, mask.shape[0])
    assert not bool(mask[cond][:, pred].any())  # condition cannot read prediction
    assert bool(mask[pred][:, cond].all())  # prediction can read condition
    assert bool(mask[cond][:, : config.num_condition_tokens].all())  # and the video conditions


def test_masking_a_human_step_also_masks_its_action_tokens() -> None:
    """A human latent step and action tokens covering the same control steps
    must be dropped together, or the mask leaks through the action branch."""
    config = tiny_config()
    human_mask = torch.tensor([[True, False, True]])  # step 1 missing
    mask = build_availability_attention_mask(config, human_mask)[0, 0]
    base = config.num_video_tokens + 1  # skip the always-present robot anchor
    steps = config.human_action_latent_step()
    pred = torch.arange(config.num_video_tokens + config.num_action_condition_tokens, mask.shape[0])
    for index, step in enumerate(steps):
        column = base + index
        readable = bool(mask[pred, column].all())
        assert readable == (step != 1), (index, step, readable)


def test_video_can_be_masked_while_human_motion_remains_visible() -> None:
    """Carrier ablation must hard-mask video keys without also hiding motion."""
    config = tiny_config()
    video_mask = torch.zeros((1, config.num_latent_steps), dtype=torch.bool)
    motion_mask = torch.ones_like(video_mask)
    mask = build_availability_attention_mask(config, video_mask, human_motion_mask=motion_mask)[
        0, 0
    ]
    tpf = config.tokens_per_frame
    prediction = torch.arange(
        config.num_video_tokens + config.num_action_condition_tokens,
        mask.shape[0],
    )
    # Every human-video frame is absent as a key.
    for slot in range(1, 1 + config.num_latent_steps):
        assert not bool(mask[prediction][:, slot * tpf : (slot + 1) * tpf].any())
    # Every human-motion token remains readable; skip the robot t0 anchor.
    motion = slice(
        config.num_video_tokens + 1,
        config.num_video_tokens + config.num_action_condition_tokens,
    )
    assert bool(mask[prediction][:, motion].all())


def test_split_masks_default_to_the_original_joint_mask() -> None:
    """The eval-only API must not change training/checkpoint behavior."""
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    mask = torch.tensor([[True, False, True]])
    kwargs = dict(
        robot_current_latent=batch["robot_current_latent"],
        human_latents=batch["human_latents"],
        noisy_robot_future_latents=batch["robot_future_latents"],
        noisy_robot_future_action=batch["robot_future_action"],
        timestep_video=timestep,
        timestep_action=timestep,
        human_mask=mask,
        robot_current_state=batch["robot_current_state"],
        human_future_action=batch["human_future_action"],
    )
    with torch.no_grad():
        implicit = model.predict_velocity(**kwargs)
        explicit = model.predict_velocity(**kwargs, human_motion_mask=mask)
    assert torch.equal(implicit[0], explicit[0])
    assert torch.equal(implicit[1], explicit[1])


def test_video_only_config_hard_masks_human_motion_but_keeps_robot_anchor() -> None:
    """The same-size control reads human video, never the 84-D motion tokens."""
    torch.manual_seed(0)
    config = tiny_config(use_human_motion_context=False)
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)
    video_mask = torch.ones((1, config.num_latent_steps), dtype=torch.bool)
    kwargs = dict(
        robot_current_latent=batch["robot_current_latent"],
        human_latents=batch["human_latents"],
        noisy_robot_future_latents=batch["robot_future_latents"],
        noisy_robot_future_action=batch["robot_future_action"],
        timestep_video=timestep,
        timestep_action=timestep,
        human_mask=video_mask,
        robot_current_state=batch["robot_current_state"],
        human_future_action=batch["human_future_action"],
    )
    with torch.no_grad():
        implicit = model.predict_velocity(**kwargs)
        explicit = model.predict_velocity(**kwargs, human_motion_mask=torch.zeros_like(video_mask))
    assert torch.equal(implicit[0], explicit[0])
    assert torch.equal(implicit[1], explicit[1])

    attention = model.attention_mask(
        video_mask,
        human_motion_mask=torch.zeros_like(video_mask),
        device=torch.device("cpu"),
    )[0, 0]
    prediction = torch.arange(
        config.num_video_tokens + config.num_action_condition_tokens,
        attention.shape[0],
    )
    robot_anchor = config.num_video_tokens
    human_motion = slice(robot_anchor + 1, robot_anchor + 1 + config.num_action_tokens)
    assert bool(attention[prediction, robot_anchor].all())
    assert not bool(attention[prediction][:, human_motion].any())


def test_robot_video_only_config_exposes_only_robot_z0_to_predictions() -> None:
    """Strict same-size control has no human or robot-state condition side path."""
    config = tiny_config(
        use_human_video_context=False,
        use_human_motion_context=False,
        use_robot_state_context=False,
    )
    model = HRMoTFlowModel(config).eval()
    supplied = torch.ones((1, config.num_latent_steps), dtype=torch.bool)
    attention = model.attention_mask(
        supplied,
        human_motion_mask=supplied,
        device=torch.device("cpu"),
    )[0, 0]
    prediction = torch.cat(
        (
            torch.arange(config.num_condition_tokens, config.num_video_tokens),
            torch.arange(
                config.num_video_tokens + config.num_action_condition_tokens,
                attention.shape[0],
            ),
        )
    )
    tpf = config.tokens_per_frame
    assert bool(attention[prediction, :tpf].all())
    assert not bool(attention[prediction, tpf : config.num_condition_tokens].any())
    action_conditions = slice(
        config.num_video_tokens,
        config.num_video_tokens + config.num_action_condition_tokens,
    )
    assert not bool(attention[prediction, action_conditions].any())


def test_robot_video_only_outputs_ignore_every_masked_input() -> None:
    """Changing human video, 84-D motion or robot state cannot change outputs."""
    torch.manual_seed(0)
    config = tiny_config(
        use_human_video_context=False,
        use_human_motion_context=False,
        use_robot_state_context=False,
    )
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    timestep = torch.full((1,), 500.0)

    def run(human_latents, human_motion, robot_state):
        return model.predict_velocity(
            robot_current_latent=batch["robot_current_latent"],
            human_latents=human_latents,
            noisy_robot_future_latents=batch["robot_future_latents"],
            noisy_robot_future_action=batch["robot_future_action"],
            timestep_video=timestep,
            timestep_action=timestep,
            human_mask=torch.ones((1, config.num_latent_steps), dtype=torch.bool),
            human_motion_mask=torch.ones((1, config.num_latent_steps), dtype=torch.bool),
            robot_current_state=robot_state,
            human_future_action=human_motion,
        )

    with torch.no_grad():
        baseline = run(
            batch["human_latents"],
            batch["human_future_action"],
            batch["robot_current_state"],
        )
        changed = run(
            torch.randn_like(batch["human_latents"]) * 100,
            torch.randn_like(batch["human_future_action"]) * 100,
            torch.randn_like(batch["robot_current_state"]) * 100,
        )
    assert torch.equal(baseline[0], changed[0])
    assert torch.equal(baseline[1], changed[1])


def test_robot_video_only_keeps_every_parameter_in_the_ddp_graph() -> None:
    """Same-size disabled parameters need zero gradients, never missing gradients."""
    torch.manual_seed(0)
    config = tiny_config(
        use_human_video_context=False,
        use_human_motion_context=False,
        use_robot_state_context=False,
    )
    model = HRMoTFlowModel(config)
    loss, _ = model.training_loss(**make_batch(config, batch=2))
    loss.backward()
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not unused, unused


def test_no_query_row_is_entirely_masked() -> None:
    """A fully-False row makes softmax produce NaN."""
    config = tiny_config()
    slots = config.num_latent_steps
    # every one of the 2^Tz patterns, including "no human context at all"
    for code in range(2**slots):
        human_mask = torch.tensor([[bool((code >> i) & 1) for i in range(slots)]], dtype=torch.bool)
        mask = build_availability_attention_mask(config, human_mask)[0, 0]
        assert bool(mask.any(dim=1).all()), human_mask


# --- Flow-matching objective -------------------------------------------------


def test_flow_interpolation_and_target() -> None:
    config = tiny_config()
    model = HRMoTFlowModel(config)
    scheduler = model.train_video_scheduler
    sample = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    timestep = torch.tensor([0.0, float(scheduler.num_train_timesteps)])
    noisy = scheduler.add_noise(sample, noise, timestep)
    assert torch.allclose(noisy[0], sample[0], atol=1e-6)  # sigma=0 -> clean
    assert torch.allclose(noisy[1], noise[1], atol=1e-6)  # sigma=1 -> pure noise
    assert torch.allclose(scheduler.training_target(sample, noise, timestep), noise - sample)


def test_generate_returns_the_requested_shapes() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = HRMoTFlowModel(config).eval()
    batch = make_batch(config, batch=1)
    latents, actions = model.generate(
        robot_current_latent=batch["robot_current_latent"],
        human_latents=batch["human_latents"],
        robot_current_state=batch["robot_current_state"],
        human_future_action=batch["human_future_action"],
        num_inference_steps=3,
    )
    assert latents.shape == (
        1,
        config.num_horizons,
        config.latent_channels,
        config.latent_height,
        config.latent_width,
    )
    assert actions.shape == (1, config.num_action_tokens, config.action_dim)
    assert torch.isfinite(latents).all() and torch.isfinite(actions).all()


# --- Config guards -----------------------------------------------------------


def test_rope_head_dim_must_split_evenly() -> None:
    with pytest.raises(ValueError, match="3D RoPE"):
        tiny_config(attn_head_dim=64).validate()


def test_image_dims_must_be_divisible_by_thirty_two() -> None:
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    record = RawHREpisode(
        path=__import__("pathlib").Path("/nonexistent.hdf5"),
        annotation_path="a",
        instruction="pick",
        group_key="pick",
        task="t",
        needs_review=False,
        clean_behavior_eligible=True,
        trajectory_quality=None,
        human_frames=100,
        robot_frames=100,
    )
    with pytest.raises(ValueError, match="divisible by 32"):
        RawHRStreamingDataset([record, record], stats=stats, image_height=100)


def test_segment_length_must_match_the_vae_contract() -> None:
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    records = [_fake_record("a", "pick"), _fake_record("b", "pick")]
    with pytest.raises(ValueError, match=r"\(T-1\) % 4"):
        RawHRStreamingDataset(records, stats=stats, num_rgb_frames=8)
    ds = RawHRStreamingDataset(records, stats=stats, num_rgb_frames=9)
    assert ds.num_latent_steps == 3
    # strided video window vs contiguous action window
    assert ds.video_window(100) == (100, 104, 108, 112, 116, 120, 124, 128, 132)
    assert ds.action_window(100) == tuple(range(101, 133))


def test_dataset_human_condition_is_raw_84d_hand_motion(tmp_path) -> None:
    """Human context must read the hand fields and never the robot `action`."""
    import h5py

    path = tmp_path / "episode.hdf5"
    length = 40
    robot_action_sentinel = np.full((length, 7), 99999, dtype=np.float32)
    hand_frames = np.arange(length * 12, dtype=np.float32).reshape(length, 4, 3)
    hand_coords = (1000 + np.arange(length * 72, dtype=np.float32)).reshape(length, 24, 3)
    with h5py.File(path, "w") as handle:
        camera = handle.create_group("cam_data")
        camera.create_dataset("human_camera", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        camera.create_dataset("robot_camera", data=np.zeros((length, 8, 8, 3), dtype=np.uint8))
        handle.create_dataset("action", data=robot_action_sentinel)
        handle.create_dataset("end_position", data=np.zeros((length, 6), dtype=np.float32))
        handle.create_dataset("gripper_state", data=np.zeros(length, dtype=np.float32))
        handle.create_dataset("transformed_hand_frames", data=hand_frames)
        handle.create_dataset("transformed_hand_coords", data=hand_coords)
    record = RawHREpisode(
        path=path,
        annotation_path="episode.hdf5",
        instruction="pick",
        group_key="pick",
        task="t",
        needs_review=False,
        clean_behavior_eligible=True,
        trajectory_quality=None,
        human_frames=length,
        robot_frames=length,
    )
    human_mean = (0.0,) * HUMAN_ACTION_DIM
    human_std = (1.0,) * HUMAN_ACTION_DIM
    stats = StateStats(
        (0.0,) * 7,
        (1.0,) * 7,
        human_mean,
        human_std,
    )
    dataset = RawHRStreamingDataset(
        [record],
        stats=stats,
        image_height=32,
        image_width=32,
        deterministic=True,
        require_cross_episode_context=False,
    )
    sample = dataset[0]
    indices = dataset.action_window(int(sample["human_start_frame"]))
    expected = np.concatenate(
        (
            hand_frames[list(indices)].reshape(len(indices), -1),
            hand_coords[list(indices)].reshape(len(indices), -1),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(sample["human_future_action"].numpy(), expected)
    assert sample["human_future_action"].shape == (32, HUMAN_ACTION_DIM)
    assert HUMAN_ACTION_SOURCE.startswith("transformed_hand_frames")


def test_windows_per_episode_scales_the_epoch() -> None:
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    records = [_fake_record("a", "pick"), _fake_record("b", "pick")]
    assert len(RawHRStreamingDataset(records, stats=stats)) == 2
    assert len(RawHRStreamingDataset(records, stats=stats, windows_per_episode=64)) == 128


# --- Data leakage ------------------------------------------------------------


def test_window_start_is_sampled_across_the_trajectory() -> None:
    """Reading only frame 0 left just one sample per episode (1,248 total,
    2.3% of each trajectory), which a 593M model memorizes outright."""
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    records = [_fake_record("a", "pick", 400), _fake_record("b", "pick", 400)]
    dataset = RawHRStreamingDataset(records, stats=stats, seed=0)
    rng = dataset._rng(0)
    starts = {dataset._start_frame(rng, 400) for _ in range(200)}
    assert len(starts) > 50, starts
    # the window now reaches t0 + 32, so t0 must stop 32 short of the end
    assert dataset.max_offset == 32
    assert max(starts) <= 400 - dataset.max_offset
    assert min(starts) >= 0


def test_holdout_sampling_is_deterministic() -> None:
    """The holdout curve is only comparable across steps if the batch is fixed."""
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    records = [_fake_record(n, "pick", 400) for n in "abc"]
    a = RawHRStreamingDataset(records, stats=stats, seed=3, deterministic=True)
    b = RawHRStreamingDataset(records, stats=stats, seed=3, deterministic=True)
    for index in range(len(a)):
        ra, rb = a._rng(index), b._rng(index)
        assert a._context_index(index, ra) == b._context_index(index, rb)
        assert a._start_frame(ra, 400) == b._start_frame(rb, 400)


def test_aligned_context_reads_the_same_episode_at_the_same_index() -> None:
    """human_camera[t] and robot_camera[t] are the same phase of the same task in
    the same episode. Aligned mode must preserve that, or the human future frames
    carry no information about the robot future."""
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    records = [_fake_record("a", "pick", 400), _fake_record("b", "pick", 400)]
    # deterministic=True so _rng returns a freshly seeded generator per index;
    # in the training path it is one shared advancing stream on purpose.
    dataset = RawHRStreamingDataset(records, stats=stats, context="aligned", deterministic=True)
    for index in range(len(dataset)):
        first = dataset._start_frame(dataset._rng(index), 400)
        assert first == dataset._start_frame(dataset._rng(index), 400)
    assert dataset.context == "aligned"
    # aligned mode needs no cross-episode partner, so nothing is dropped
    lonely = [*records, _fake_record("lonely", "unique", 400)]
    assert len(RawHRStreamingDataset(lonely, stats=stats, context="aligned")) == 3


def test_singleton_groups_are_dropped_so_context_is_cross_episode() -> None:
    """The old pipeline fell back to the target episode itself for singleton
    groups, making the human 'context' the very episode being predicted."""
    import pathlib

    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)

    def record(name: str, group: str) -> RawHREpisode:
        return RawHREpisode(
            path=pathlib.Path(f"/{name}.hdf5"),
            annotation_path=name,
            instruction=group,
            group_key=group,
            task="t",
            needs_review=False,
            clean_behavior_eligible=True,
            trajectory_quality=None,
            human_frames=100,
            robot_frames=100,
        )

    records = [record("a", "pick"), record("b", "pick"), record("lonely", "unique")]
    dataset = RawHRStreamingDataset(
        records,
        stats=stats,
        image_height=DEFAULT_IMAGE_HEIGHT,
        image_width=DEFAULT_IMAGE_WIDTH,
        context="cross_episode",
    )
    assert len(dataset) == 2
    assert dataset.singleton_group_episodes == 1
    assert all(r.group_key == "pick" for r in dataset.records)
    for index in range(len(dataset)):
        assert dataset._context_index(index, dataset._rng(index)) != index


# --- Standalone inference/evaluation contracts ------------------------------


def test_other_task_mapping_always_changes_instruction() -> None:
    groups = ["pick", "pick", "pour", "pick", "pour", "open"]
    mapping = different_instruction_indices(groups)
    assert len(mapping) == len(groups)
    assert all(groups[source] != groups[target] for source, target in enumerate(mapping))
    usage = np.bincount(mapping, minlength=len(groups))
    assert int(usage.max() - usage.min()) <= 1
    with pytest.raises(ValueError, match="two different instruction"):
        different_instruction_indices(["pick", "pick"])


def test_pixel_metric_excludes_known_robot_current_frame() -> None:
    truth = torch.zeros(2, 3, 9, 4, 4)
    predicted = truth.clone()
    predicted[:, :, 0] = 1000.0
    assert torch.equal(future_pixel_mse(predicted, truth), torch.zeros(2))
    predicted[:, :, 1:] = 2.0
    assert torch.equal(future_pixel_mse(predicted, truth), torch.full((2,), 4.0))


def test_eval_geometry_comes_from_checkpoint_config() -> None:
    config = tiny_config(num_rgb_frames=13, frame_stride=3, action_horizon=36)
    assert dataset_geometry(config) == {
        "num_rgb_frames": 13,
        "frame_stride": 3,
        "action_horizon": 36,
    }


def test_checkpoint_run_pair_rejects_mixed_metadata() -> None:
    config = tiny_config().to_dict()
    stats = {"robot_mean": [0.0] * 7, "robot_std": [1.0] * 7}
    contract = {"schema_version": HR_MOT_SCHEMA_VERSION, "sentinel": "exact-run"}
    contract_hash = canonical_sha256(contract)
    checkpoint = {
        "schema_version": HR_MOT_SCHEMA_VERSION,
        "run_id": "run-a",
        "evaluation_contract_sha256": contract_hash,
        "config": config,
        "state_stats": stats,
    }
    run_config = {
        "schema_version": HR_MOT_SCHEMA_VERSION,
        "run_id": "run-a",
        "evaluation_contract": contract,
        "evaluation_contract_sha256": contract_hash,
        "model": config,
        "state_stats": stats,
    }
    validate_checkpoint_run_pair(checkpoint, run_config)
    with pytest.raises(ValueError, match="run_id differs"):
        validate_checkpoint_run_pair(checkpoint, {**run_config, "run_id": "run-b"})
    with pytest.raises(ValueError, match="legacy or unsupported"):
        validate_checkpoint_run_pair(
            {"config": config, "state_stats": stats},
            {"model": config, "state_stats": stats},
        )
    with pytest.raises(ValueError, match="legacy or unsupported"):
        validate_checkpoint_run_pair(
            {**checkpoint, "schema_version": 2},
            {**run_config, "schema_version": 2},
        )
    with pytest.raises(ValueError, match="legacy or unsupported"):
        validate_checkpoint_run_pair(
            {**checkpoint, "schema_version": 3},
            {**run_config, "schema_version": 3},
        )
    with pytest.raises(ValueError, match="contract hash is invalid"):
        validate_checkpoint_run_pair(
            checkpoint, {**run_config, "evaluation_contract": {"sentinel": "changed"}}
        )


def test_config_metadata_matches_droppable_context_and_action_count() -> None:
    config = HRMoTConfig()
    metadata = human_context_metadata(config)
    assert config.num_action_tokens == 32
    assert metadata["always_supplied"] == ["robot_z0"]
    assert metadata["maskable"] == ["human_z0", "human_z1", "human_z2"]
    assert metadata["availability_patterns"] == 8
    assert metadata["human_action_condition"] == {
        "enabled": True,
        "source": ("HDF5 transformed_hand_frames(4x3) + " "transformed_hand_coords(24x3)"),
        "dimension": 84,
        "control_steps": [1, 32],
        "tokens": 32,
        "ground_truth": True,
    }


def test_video_only_metadata_marks_human_motion_disabled() -> None:
    metadata = human_context_metadata(HRMoTConfig(use_human_motion_context=False))
    assert metadata["use_state_context"] is True
    assert metadata["human_action_condition"]["enabled"] is False
    assert metadata["human_action_condition"]["ground_truth"] is False


def test_robot_video_only_metadata_marks_all_auxiliary_context_disabled() -> None:
    metadata = human_context_metadata(
        HRMoTConfig(
            use_human_video_context=False,
            use_human_motion_context=False,
            use_robot_state_context=False,
        )
    )
    assert metadata["use_human_video_context"] is False
    assert metadata["use_robot_state_context"] is False
    assert metadata["human_action_condition"]["enabled"] is False


def test_online_rollout_encodes_only_robot_current_frame() -> None:
    config = tiny_config()

    class FakeModel:
        def __init__(self):
            self.config = config
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def generate(self, **kwargs):
            current = kwargs["robot_current_latent"]
            assert kwargs["robot_current_state"].dtype == torch.float32
            assert kwargs["human_future_action"].dtype == torch.float32
            assert kwargs["robot_current_state"].device == current.device
            assert kwargs["human_future_action"].device == current.device
            batch, _, channels, height, width = current.shape
            return (
                torch.zeros(batch, config.num_horizons, channels, height, width),
                torch.zeros(batch, config.num_action_tokens, config.action_dim),
            )

    class FakeEncoder:
        def __init__(self):
            self.encoded_lengths = []

        def encode_video(self, video):
            self.encoded_lengths.append(video.shape[2])
            latent_steps = 1 if video.shape[2] == 1 else config.num_latent_steps
            return torch.zeros(video.shape[0], config.latent_channels, latent_steps, 2, 2)

        def decode_video(self, latent):
            return torch.zeros(latent.shape[0], 3, config.num_rgb_frames, 4, 4)

    model = FakeModel()
    encoder = FakeEncoder()
    stats = StateStats((0.0,) * 7, (1.0,) * 7, (0.0,) * HUMAN_ACTION_DIM, (1.0,) * HUMAN_ACTION_DIM)
    rollout = generate_rollout(
        model,
        encoder,
        stats,
        robot_current_frame=torch.zeros(2, 3, 4, 4),
        human_video=torch.zeros(2, 3, config.num_rgb_frames, 4, 4),
        robot_current_state=torch.zeros(2, 7, dtype=torch.float64),
        human_future_action=torch.zeros(
            2, config.num_action_tokens, config.human_action_dim, dtype=torch.float64
        ),
    )
    assert encoder.encoded_lengths == [1, config.num_rgb_frames]
    assert rollout.robot_future_state.shape == (2, config.num_action_tokens, 7)
    assert rollout.robot_future_frames.shape == (2, 3, config.num_rgb_frames - 1, 4, 4)
    assert model.training  # helper restores the caller's mode
