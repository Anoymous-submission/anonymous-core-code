"""Bind tasks, splits, interventions and raw file digests before reading model inputs."""

import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(directory, files):
    directory = Path(directory).resolve()
    for name, digest in files.items():
        path = (directory / name).resolve()
        assert path.parent == directory, ("Unsafe manifest file", name)
        assert sha(path) == digest, ("Raw input digest mismatch", str(path))


def dataset_contract(data, task, *, formal, plan_path, condition=None, split=None):
    data = Path(data)
    plan_path = Path(plan_path)
    path = data / "GENERATION.json"
    meta = json.loads(path.read_text())
    assert meta["task"] == task, "Checkpoint/dataset task mismatch"
    assert meta["records"] == 8 * meta["families"] and meta["records"] > 0
    assert all(
        k in meta["files"]
        for k in ("records.npz", "source_rgb.npy", "source_motion.npy", "source_audit.npz")
    )
    verify_files(data, meta["files"])
    if formal:
        plan = json.loads(plan_path.read_text())
        assert split in (
            "validation",
            "test",
        ), "Formal inference requires an explicit held-out split"
        assert (
            meta["stage"] == "frozen-data-generation" and meta["split"] == split
        ), "Training/development data cannot be formal evaluation"
        conditions = {x["name"]: x for x in plan["conditions"]}
        assert condition in conditions, "Missing or unknown prespecified condition"
        target = conditions[condition]["target"]
        assert meta["geometry_ood"] == (target == "geometry"), "Condition/geometry mismatch"
        assert (
            meta["seed"] == plan["data_seeds"][split + "_" + target][task]
        ), "Unexpected held-out data seed"
        assert (
            meta["families"] == plan[split + "_families_per_task"]
        ), "Unexpected held-out population"
        assert meta["evaluation_plan_sha"] == sha(
            plan_path
        ), "Dataset bound to another evaluation plan"
        assert (
            json.loads((data / "EVALUATION_COMPLETE.json").read_text()) == meta
        ), "Dataset incomplete"
    return dict(
        task=task,
        split=meta.get("split", "training"),
        seed=meta["seed"],
        stage=meta["stage"],
        geometry_ood=meta["geometry_ood"],
        records=meta["records"],
        families=meta["families"],
        formal=formal,
        condition=condition,
        manifest_name=path.name,
        manifest_sha=sha(path),
        evaluation_plan_sha=sha(plan_path),
        files=meta["files"],
    )


def source_contract(inputs, parent, task, *, condition, context, plan_path, formal):
    inputs = Path(inputs)
    path = inputs / "VARIANT.json"
    meta = json.loads(path.read_text())
    assert meta["task"] == task and meta["parent_records_sha"] == parent["files"]["records.npz"]
    assert meta["variant"] in ("changed_command", "heldout_style")
    expected = {
        "id_changed_source_command": "changed_command",
        "source_appearance": "heldout_style",
    }
    if formal:
        assert (
            condition in expected and meta["variant"] == expected[condition]
        ), "Wrong source intervention"
    assert context == "matching", "Source interventions require matching context"
    assert meta["plan_sha"] == sha(
        plan_path
    ), "Source intervention bound to another evaluation plan"
    assert (
        meta["target_records_modified"] is False and meta["source_geometry_and_physics_unchanged"]
    )
    verify_files(inputs, meta["files"])
    return dict(
        task=task,
        variant=meta["variant"],
        manifest_sha=sha(path),
        parent_records_sha=meta["parent_records_sha"],
        evaluation_plan_sha=meta["plan_sha"],
        files=meta["files"],
    )


def verify_recorded_contract(data, contract):
    data = Path(data)
    assert (
        sha(data / contract["manifest_name"]) == contract["manifest_sha"]
    ), "Dataset manifest changed after prediction"
    verify_files(data, contract["files"])
