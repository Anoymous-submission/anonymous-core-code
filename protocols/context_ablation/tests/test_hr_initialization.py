"""Behavioral checks for preserving a robot base when opening human inputs."""

import json

import pytest
import torch

from fasterwam.models.hr_eval import HR_MOT_SCHEMA_VERSION, canonical_sha256, file_sha256
from fasterwam.models.hr_initialization import load_initial_model, validate_initialization_data
from fasterwam.models.hr_mot import HRMoTFlowModel
from test_hr_mot import tiny_config, make_batch


def saved_base(tmp_path, mutate=None):
    torch.manual_seed(41)
    model = HRMoTFlowModel(
        tiny_config(use_human_video_context=False, use_human_motion_context=False)
    ).eval()
    # Nontrivial predictions, retaining the randomly initialized availability
    # embeddings bypassed by the real nohuman training run.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "availability_embedding" not in name:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    stats = {"robot_mean": [0.0], "robot_std": [1.0]}
    contract = {"state_stats": stats}
    ckpt = dict(
        schema_version=HR_MOT_SCHEMA_VERSION,
        step=10000,
        run_id="base-test",
        config=model.config.to_dict(),
        state_stats=stats,
        model=model.state_dict(),
        evaluation_contract_sha256=canonical_sha256(contract),
        optimizer={"state": "must not restore"},
        scheduler={"last_epoch": 10000},
    )
    if mutate:
        mutate(ckpt)
    path = tmp_path / "base.pt"
    torch.save(ckpt, path)
    (tmp_path / "config.json").write_text(
        json.dumps(dict(model=model.config.to_dict(), evaluation_contract=contract))
    )
    return model, path, stats


def test_strict_initialization_preserves_no_demo_output(tmp_path):
    source, path, stats = saved_base(tmp_path)
    target = HRMoTFlowModel(
        tiny_config(none_human_context_prob=0.3, full_human_context_prob=1 - 0.3 - 6 / 70)
    ).eval()
    info, _ = load_initial_model(target, path, file_sha256(path), stats)
    assert info["source_cumulative_steps"] == 10000
    assert not info["optimizer_restored"] and not info["scheduler_restored"]
    assert len(info["neutralized_unused_absent_codes"]) == 2
    for name, tensor in source.state_dict().items():
        loaded = target.state_dict()[name]
        if "availability_embedding" in name:
            assert torch.count_nonzero(loaded[:, 0]) == 0
            assert torch.equal(tensor[:, 1], loaded[:, 1])
        else:
            assert torch.equal(tensor, loaded)
    batch = make_batch(source.config, batch=1)
    kwargs = dict(
        robot_current_latent=batch["robot_current_latent"],
        human_latents=batch["human_latents"],
        noisy_robot_future_latents=batch["robot_future_latents"],
        noisy_robot_future_action=batch["robot_future_action"],
        timestep_video=torch.tensor([500.0]),
        timestep_action=torch.tensor([500.0]),
        human_mask=torch.zeros(1, 3, dtype=torch.bool),
        robot_current_state=batch["robot_current_state"],
        human_future_action=batch["human_future_action"],
    )
    with torch.no_grad():
        expected, actual = source.predict_velocity(**kwargs), target.predict_velocity(**kwargs)
    assert actual[0].abs().sum() > 0 and actual[1].abs().sum() > 0
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    kwargs["human_mask"] = torch.ones(1, 3, dtype=torch.bool)
    video, action = target.predict_velocity(**kwargs)
    (video.square().mean() + action.square().mean()).backward()
    gradient = target.mot.mixtures["action"].human_action_embedding.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_unchanged_context_keeps_all_weights(tmp_path):
    source, path, stats = saved_base(tmp_path)
    target = HRMoTFlowModel(source.config).eval()
    info, _ = load_initial_model(target, path, file_sha256(path), stats)
    assert info["neutralized_unused_absent_codes"] == []
    assert all(
        torch.equal(value, target.state_dict()[name]) for name, value in source.state_dict().items()
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c.update(schema_version=3), "schema"),
        (lambda c: c["config"].update(mixed_attention_type="linear_ropefp32"), "architecture"),
        (lambda c: c["state_stats"].update(robot_std=[2.0]), "normalization"),
        (lambda c: c.update(evaluation_contract_sha256="0" * 64), "configuration"),
    ],
)
def test_incompatible_initialization_fails(tmp_path, mutation, match):
    _, path, _ = saved_base(tmp_path, mutation)
    model = HRMoTFlowModel(tiny_config())
    with pytest.raises(ValueError, match=match):
        load_initial_model(
            model, path, file_sha256(path), {"robot_mean": [0.0], "robot_std": [1.0]}
        )


def test_missing_tensor_is_not_silently_initialized(tmp_path):
    _, path, stats = saved_base(tmp_path, lambda c: c["model"].pop(next(iter(c["model"]))))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_initial_model(HRMoTFlowModel(tiny_config()), path, file_sha256(path), stats)


def test_changed_holdout_is_rejected():
    from fasterwam.models.hr_initialization import DATA_CONTRACT_FIELDS

    source = {key: None for key in DATA_CONTRACT_FIELDS}
    source["holdout_record_ids"] = ["old"]
    with pytest.raises(ValueError, match="holdout_record_ids"):
        validate_initialization_data(source, dict(source, holdout_record_ids=["new"]))


def test_wrong_checkpoint_hash_is_rejected(tmp_path):
    _, path, stats = saved_base(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_initial_model(HRMoTFlowModel(tiny_config()), path, "0" * 64, stats)
