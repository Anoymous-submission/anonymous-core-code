"""Graph isolation and gradient-path tests, not result-dependent checks."""

from dataclasses import replace
import json
import torch
from fasterwam.models.hr_mot import (
    HRMoTConfig,
    HRMoTFlowModel,
    build_availability_attention_mask as mask,
)

torch.set_num_threads(2)
torch.manual_seed(17)
c = HRMoTConfig(
    num_layers=2,
    num_heads=2,
    attn_head_dim=96,
    video_hidden_dim=64,
    video_ffn_dim=128,
    action_hidden_dim=32,
    action_ffn_dim=64,
    latent_height=4,
    latent_width=4,
    gradient_checkpointing=False,
    full_human_context_prob=0.5,
    none_human_context_prob=0.5,
)
c.validate()
patterns = ((torch.arange(8)[:, None] >> torch.arange(3)[None, :]) & 1).bool()
a = mask(c, patterns)[:, 0]
b = mask(replace(c, future_video_action_coupling=False), patterns)[:, 0]
v0 = c.num_condition_frames * c.tokens_per_frame
v1 = c.num_video_tokens
a0 = v1 + c.num_action_condition_tokens
expected = a.clone()
expected[:, v0:v1, a0:] = False
expected[:, a0:, v0:v1] = False
assert torch.equal(expected, b)
assert (a[:, v0:v1, a0:]).all() and not b[:, v0:v1, a0:].any()
# Removing absent carriers recovers the direct FastWAM graph exactly.
direct = replace(
    c, direct_fastwam_robot_only=True, use_human_video_context=False, use_human_motion_context=False
)
keep = list(range(c.tokens_per_frame)) + list(range(v0, v1)) + [v1] + list(range(a0, b.shape[-1]))
assert torch.equal(b[0][keep][:, keep], mask(direct, patterns[:1])[0, 0])
m = HRMoTFlowModel(replace(c, future_video_action_coupling=False))
kw = dict(
    robot_current_latent=torch.randn(2, 1, 48, 4, 4),
    human_latents=torch.randn(2, 3, 48, 4, 4),
    noisy_robot_future_latents=torch.randn(2, 2, 48, 4, 4),
    noisy_robot_future_action=torch.randn(2, 32, 7),
    timestep_video=torch.ones(2) * 0.4,
    timestep_action=torch.ones(2) * 0.6,
    human_mask=torch.ones(2, 3, dtype=torch.bool),
    robot_current_state=torch.randn(2, 7),
    human_future_action=torch.randn(2, 32, 84),
)
# Randomize the zero-init output projections so invariance is non-vacuous.
with torch.no_grad():
    for name, p in m.named_parameters():
        if torch.count_nonzero(p) == 0:
            p.normal_(0, 0.02)
x = m.predict_velocity(**kw)
y = m.predict_velocity(**dict(kw, noisy_robot_future_latents=kw["noisy_robot_future_latents"] + 3))
z = m.predict_velocity(**dict(kw, noisy_robot_future_action=kw["noisy_robot_future_action"] + 3))
assert torch.equal(x[1], y[1]) and torch.equal(x[0], z[0])
assert not torch.equal(x[0], y[0]) and not torch.equal(x[1], z[1])
loss = sum(t.square().mean() for t in x)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
for coupling in (False, True):
    for demo in (False, True):
        torch.manual_seed(0)
        cfg = replace(
            c,
            future_video_action_coupling=coupling,
            none_human_context_prob=0.5 if demo else 1.0,
            full_human_context_prob=0.5 if demo else 0.0,
        )
        model = HRMoTFlowModel(cfg)
        loss, metrics = model.training_loss(
            robot_current_latent=kw["robot_current_latent"],
            human_latents=kw["human_latents"],
            robot_future_latents=kw["noisy_robot_future_latents"],
            robot_future_action=kw["noisy_robot_future_action"],
            robot_current_state=kw["robot_current_state"],
            human_future_action=kw["human_future_action"],
        )
        loss.backward()
        assert torch.isfinite(loss) and all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        )
print(
    json.dumps(
        dict(
            state="passed",
            checks=[
                "only_two_future_cross_edges_changed",
                "direct_graph_exact_after_carrier_removal",
                "nonvacuous_prediction_independence_both_directions",
                "all_four_training_loss_and_gradients_finite",
            ],
        )
    )
)
