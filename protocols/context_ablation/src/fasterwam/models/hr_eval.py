"""Shared, testable contracts for standalone H&R evaluation."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

HR_MOT_SCHEMA_VERSION = 4


def dataset_geometry(config: Any) -> dict[str, int]:
    """Dataset time geometry must come from the checkpoint, never defaults."""

    return {
        "num_rgb_frames": int(config.num_rgb_frames),
        "frame_stride": int(config.frame_stride),
        "action_horizon": int(config.action_horizon),
    }


def human_context_metadata(config: Any) -> dict[str, Any]:
    """Truthful config.json description of the trained availability contract."""

    first_maskable = 0 if config.droppable_human_current else 1
    return {
        "human_latent_steps": config.num_latent_steps,
        "always_supplied": ["robot_z0"] + ([] if config.droppable_human_current else ["human_z0"]),
        "maskable": [f"human_z{i}" for i in range(first_maskable, config.num_latent_steps)],
        "availability_patterns": 2 ** (config.num_latent_steps - first_maskable),
        "full_context_prob": config.full_human_context_prob,
        "explicit_none_context_prob": config.none_human_context_prob,
        "droppable_human_current": config.droppable_human_current,
        "use_state_context": config.use_state_context,
        "use_robot_state_context": config.use_robot_state_context,
        "use_human_video_context": config.use_human_video_context,
        "human_action_condition": {
            "enabled": config.use_human_motion_context,
            "source": ("HDF5 transformed_hand_frames(4x3) + " "transformed_hand_coords(24x3)"),
            "dimension": config.human_action_dim,
            "control_steps": [1, config.num_action_tokens],
            "tokens": config.num_action_tokens,
            "ground_truth": bool(config.use_human_motion_context),
        },
        "masking": "whole-frame hard attention mask",
        "note": (
            "robot z0 is always supplied; configured-off context sources are "
            "zeroed and hard-masked without changing their slots. Every maskable human latent step, "
            "including z0 when configured droppable, is supplied or not. A "
            "missing step is excluded as key/value, its query row is diagonal-only "
            "and discarded, and its RoPE position is not renumbered. Prediction "
            "tokens carry a learned availability code."
        ),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_run_pair(
    checkpoint: Mapping[str, Any], run_config: Mapping[str, Any]
) -> None:
    """Refuse to combine weights with metadata from a different training run."""

    checkpoint_schema = checkpoint.get("schema_version")
    config_schema = run_config.get("schema_version")
    if checkpoint_schema != HR_MOT_SCHEMA_VERSION or config_schema != HR_MOT_SCHEMA_VERSION:
        raise ValueError(
            "legacy or unsupported H&R MoT artifact: expected schema_version "
            f"{HR_MOT_SCHEMA_VERSION}, got checkpoint={checkpoint_schema!r}, "
            f"config={config_schema!r}; do not fill missing semantic fields from defaults"
        )
    checkpoint_run_id = checkpoint.get("run_id")
    config_run_id = run_config.get("run_id")
    if not checkpoint_run_id or not config_run_id:
        raise ValueError("schema-v4 checkpoint and config must both contain run_id")
    if checkpoint_run_id != config_run_id:
        raise ValueError("checkpoint and --run-dir do not match: run_id differs")

    contract = run_config.get("evaluation_contract")
    config_contract_hash = run_config.get("evaluation_contract_sha256")
    checkpoint_contract_hash = checkpoint.get("evaluation_contract_sha256")
    if not isinstance(contract, Mapping) or not config_contract_hash:
        raise ValueError("run config lacks the schema-v4 evaluation contract")
    actual_contract_hash = canonical_sha256(contract)
    if actual_contract_hash != config_contract_hash:
        raise ValueError("config.json evaluation contract hash is invalid")
    if checkpoint_contract_hash != config_contract_hash:
        raise ValueError("checkpoint and --run-dir do not match: evaluation contract differs")

    # The content checks also protect legacy checkpoints that predate run_id,
    # and catch corruption even when a directory was copied as a unit.
    required = (("config", "model"), ("state_stats", "state_stats"))
    for checkpoint_key, run_key in required:
        if checkpoint_key not in checkpoint:
            raise ValueError(f"checkpoint lacks required provenance field {checkpoint_key!r}")
        if run_key not in run_config:
            raise ValueError(f"run config lacks required provenance field {run_key!r}")
        if _canonical(checkpoint[checkpoint_key]) != _canonical(run_config[run_key]):
            raise ValueError(
                "checkpoint and --run-dir do not match: "
                f"{checkpoint_key!r} differs from config.json {run_key!r}"
            )


def validate_runtime_assets(run_config: Mapping[str, Any]) -> None:
    """Detect replaced annotations or VAE weights before scoring a checkpoint."""

    contract = run_config["evaluation_contract"]
    vae = contract["vae"]
    if file_sha256(vae["path"]) != vae["sha256"]:
        raise ValueError("WanVAE file content differs from the training contract")
    for item in contract["annotations"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(
                f"annotation content differs from the training contract: {item['path']}"
            )


def select_record_ids(records: Sequence[Any], expected_ids: Sequence[str]) -> list[Any]:
    """Restore an exact persisted split and fail on missing or duplicate IDs."""

    by_id: dict[str, Any] = {}
    for record in records:
        key = str(record.annotation_path)
        if key in by_id:
            raise ValueError(f"duplicate discovered annotation_path: {key}")
        by_id[key] = record
    missing = [key for key in expected_ids if key not in by_id]
    if missing:
        raise ValueError(f"training-time split records are missing: {missing[:5]}")
    return [by_id[key] for key in expected_ids]


def validate_record_files(records: Sequence[Any], contract: Mapping[str, Any]) -> None:
    """Fast raw-data fingerprint check without building or reading a cache."""

    expected = contract.get("inventory_files")
    if not isinstance(expected, list) or len(expected) != len(records):
        raise ValueError("evaluation contract lacks the complete raw-data manifest")
    for record, item in zip(records, expected):
        stat = record.path.stat()
        actual = (
            str(record.annotation_path),
            str(record.path),
            stat.st_size,
            stat.st_mtime_ns,
        )
        recorded = (
            item.get("id"),
            item.get("path"),
            item.get("size_bytes"),
            item.get("mtime_ns"),
        )
        if actual != recorded:
            raise ValueError(
                f"raw HDF5 differs from the training contract: {record.annotation_path}"
            )


def different_instruction_indices(group_keys: Sequence[str]) -> tuple[int, ...]:
    """Balanced deterministic map to provably different-instruction records."""

    size = len(group_keys)
    if size < 2 or len(set(group_keys)) < 2:
        raise ValueError("other_task evaluation requires at least two different instruction groups")
    usage = [0] * size
    result: list[int] = []
    for index, group in enumerate(group_keys):
        eligible = [
            candidate
            for candidate in range(size)
            if candidate != index and group_keys[candidate] != group
        ]
        replacement = min(
            eligible,
            key=lambda candidate: (usage[candidate], (candidate - index) % size),
        )
        result.append(replacement)
        usage[replacement] += 1
    return tuple(result)


def future_pixel_mse(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-sample RGB MSE over generated future frames only (exclude known R0)."""

    if predicted.shape != truth.shape or predicted.ndim != 5:
        raise ValueError(
            "predicted and truth must have identical [B, 3, T, H, W] shapes, got "
            f"{tuple(predicted.shape)} and {tuple(truth.shape)}"
        )
    if predicted.shape[2] < 2:
        raise ValueError("pixel evaluation needs R0 plus at least one future frame")
    return (predicted[:, :, 1:] - truth[:, :, 1:]).pow(2).flatten(1).mean(1)
