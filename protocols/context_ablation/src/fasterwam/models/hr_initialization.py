"""Strict model-only initialization for a new, separately logged training stage."""

import json
from pathlib import Path

import torch

from fasterwam.models.hr_eval import HR_MOT_SCHEMA_VERSION, canonical_sha256, file_sha256
from fasterwam.models.hr_mot import HRMoTConfig

CONTEXT_TRAINING_FIELDS = {
    "use_human_motion_context",
    "use_human_video_context",
    "full_human_context_prob",
    "none_human_context_prob",
    "gradient_checkpointing",
    "future_video_action_coupling",
}
DATA_CONTRACT_FIELDS = (
    "schema_version",
    "state_stats",
    "data_root",
    "annotations",
    "vae",
    "image_height",
    "image_width",
    "seed",
    "context",
    "eval_windows_per_episode",
    "inventory_record_ids",
    "inventory_files",
    "inventory_fingerprint",
    "train_record_ids",
    "holdout_record_ids",
)


def validate_initialization_config(source, target):
    before = HRMoTConfig(**source).to_dict()
    after = target.to_dict()
    changes = {key: [before[key], after[key]] for key in before if before[key] != after[key]}
    allowed = set(CONTEXT_TRAINING_FIELDS)
    if before["direct_fastwam_robot_only"] and not after["direct_fastwam_robot_only"]:
        allowed.add("direct_fastwam_robot_only")
    incompatible = set(changes) - allowed
    if incompatible:
        raise ValueError(f"Initialization changes architecture: {sorted(incompatible)}")
    return changes


def validate_initialization_data(source, target):
    for key in DATA_CONTRACT_FIELDS:
        if source[key] != target[key]:
            raise ValueError(f"Initialization data contract differs: {key}")


def load_initial_model(model, path, expected_sha256, state_stats, *, verify_hash=True):
    path = Path(path).resolve()
    if len(expected_sha256) != 64:
        raise ValueError("Initialization requires a full checkpoint SHA-256")
    if verify_hash and file_sha256(path) != expected_sha256:
        raise ValueError("Initialization checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint["schema_version"] != HR_MOT_SCHEMA_VERSION:
        raise ValueError("Initialization checkpoint schema mismatch")
    if checkpoint["state_stats"] != state_stats:
        raise ValueError("Initialization state normalization mismatch")
    changes = validate_initialization_config(checkpoint["config"], model.config)
    configuration = json.loads((path.parent / "config.json").read_text())
    contract = configuration["evaluation_contract"]
    if canonical_sha256(contract) != checkpoint["evaluation_contract_sha256"]:
        raise ValueError("Initialization source configuration is not bound to checkpoint")
    if (
        HRMoTConfig(**configuration["model"]).to_dict()
        != HRMoTConfig(**checkpoint["config"]).to_dict()
    ):
        raise ValueError("Initialization source model configuration mismatch")
    if contract["state_stats"] != checkpoint["state_stats"]:
        raise ValueError("Initialization source contract normalization mismatch")
    # This experiment explicitly expands one shared direct FastWAM checkpoint.
    # Account for every tensor; no permissive load or silent missing keys.
    added = []
    expanded = (
        checkpoint["config"]["direct_fastwam_robot_only"]
        and not model.config.direct_fastwam_robot_only
    )
    if expanded:
        target = model.state_dict()
        expected = {
            "mot.mixtures.video.availability_embedding",
            "mot.mixtures.action.availability_embedding",
            "mot.mixtures.action.human_action_embedding.weight",
            "mot.mixtures.action.human_action_embedding.bias",
        }
        added = sorted(set(target) - set(checkpoint["model"]))
        if set(added) != expected or set(checkpoint["model"]) - set(target):
            raise ValueError(f"Unexpected checkpoint expansion keys: {added}")
        for name in added:
            if name.endswith("availability_embedding"):
                target[name].zero_()
        target.update(checkpoint["model"])
        model.load_state_dict(target, strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    # Disabled human branches bypass these randomly initialized codes. Opening
    # a branch would otherwise add an untrained constant even with no demo.
    # Neutralize only its previously unused absent rows; preserve every trained
    # tensor and all present rows, and record the explicit transformation.
    neutralized = []
    with torch.no_grad():
        for branch, flag in (
            ("video", "use_human_video_context"),
            ("action", "use_human_motion_context"),
        ):
            if not checkpoint["config"][flag] and getattr(model.config, flag):
                name = f"mot.mixtures.{branch}.availability_embedding"
                embedding = model.get_parameter(name)
                before_max = float(embedding[:, 0].abs().max())
                embedding[:, 0].zero_()
                neutralized.append(
                    dict(
                        parameter=name,
                        rows="[:, 0, :]",
                        previous_max_abs=before_max,
                        replacement=0.0,
                    )
                )
    source_steps = int(checkpoint.get("cumulative_training_steps", checkpoint["step"]))
    return (
        dict(
            path=str(path),
            sha256=expected_sha256,
            source_step=checkpoint["step"],
            source_cumulative_steps=source_steps,
            source_run_id=checkpoint["run_id"],
            source_contract_sha256=checkpoint["evaluation_contract_sha256"],
            allowed_config_changes=changes,
            model_load="explicit_fastwam_expansion_strict" if expanded else "strict_all_tensors",
            added_parameters=added,
            neutralized_unused_absent_codes=neutralized,
            optimizer_restored=False,
            scheduler_restored=False,
            sampler_restored=False,
        ),
        contract,
    )
