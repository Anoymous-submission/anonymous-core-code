"""Execute saved policy commands and preserve native/finer trajectories and constraints."""

import argparse
import json
from pathlib import Path
import numpy as np
from predict_policy import sha
from action_limits import decode
from task_adapter import execute, QUERY_DIMS
from data_contract import verify_recorded_contract

SCALAR_FIELDS = {
    "gate": (
        "success",
        "endpoint_error",
        "gate_error",
        "gate_contact_steps",
        "crossed_gate_plane",
        "crossed_gate_aperture",
        "gate_crossing_time",
        "gate_crossing_count",
    ),
    "bank": (
        "success",
        "endpoint_error",
        "hit_panel",
        "panel_contact_steps",
        "other_contact_steps",
    ),
    "ramp": (
        "success",
        "endpoint_error",
        "ramp_contact_steps",
        "other_contact_steps",
        "departed_downhill_edge",
        "crossing_time",
    ),
}


def scalar_row(task, result, record):
    """Missing crossings are failed observations, never missing records or columns."""
    row = {}
    for key in SCALAR_FIELDS[task]:
        value = result[key]
        if value is None:
            assert task == "ramp" and key == "crossing_time", (task, key)
            value = float("nan")
        if isinstance(value, np.generic):
            value = value.item()
        assert isinstance(value, (bool, int, float)), (task, key, type(value))
        row[key] = value
    row["record"] = record
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    pred = Path(args.prediction)
    data = Path(args.data)
    out = Path(args.out)
    proof = json.loads((pred / "PREDICTION.json").read_text())
    task = proof["task"]
    if "dataset_contract" in proof:
        assert proof["dataset_contract"]["task"] == task
        verify_recorded_contract(data, proof["dataset_contract"])
    else:
        assert proof["development"], "Formal prediction lacks validated dataset provenance"
    assert proof["record_file_sha"] == sha(data / "records.npz")
    assert proof["action_file_sha"] == sha(pred / "ACTIONS.npz")
    with np.load(pred / "ACTIONS.npz", allow_pickle=False) as f:
        actions = {k: f[k].copy() for k in f.files}
    assert np.array_equal(actions["executed"], decode(actions["raw"], task))
    with np.load(data / "records.npz", allow_pickle=False) as f:
        records = {k: f[k].copy() for k in f.files}
    for k in ("family", "sibling", "query_index"):
        assert np.array_equal(actions[k], records[k])
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for condition, dt in [
        ("native", dict(gate=0.004, bank=0.001, ramp=0.001)[task]),
        ("finer", dict(gate=0.002, bank=0.0005, ramp=0.00025)[task]),
    ]:
        scalar = []
        paths = []
        offsets = [0]
        with (out / (condition + ".jsonl")).open("w") as audit:
            for i, action in enumerate(actions["executed"]):
                result = execute(
                    task,
                    records["query"][i, : QUERY_DIMS[task]],
                    action,
                    records["physics"][i],
                    dt=dt,
                )
                row = scalar_row(task, result, i)
                scalar.append(row)
                serial = {
                    k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in row.items()
                }
                audit.write(json.dumps(serial, allow_nan=False) + "\n")
                paths.append(result["positions"])
                offsets.append(offsets[-1] + len(paths[-1]))
                if (i + 1) % 32 == 0:
                    audit.flush()
                    print(json.dumps(dict(condition=condition, records_complete=i + 1)), flush=True)
        metrics = {k: np.array([row[k] for row in scalar]) for k in scalar[0]}
        np.savez_compressed(
            out / (condition + ".npz"),
            **metrics,
            positions=np.concatenate(paths),
            path_offsets=np.asarray(offsets),
            executed=actions["executed"],
            family=records["family"],
            sibling=records["sibling"],
            query_index=records["query_index"],
        )
        errors = metrics["endpoint_error"]
        finite = np.isfinite(errors)
        summary[condition] = dict(
            timestep=dt,
            records=len(scalar),
            successes=int(metrics["success"].sum()),
            success_rate=float(metrics["success"].mean()),
            nonfinite_endpoint_errors=int((~finite).sum()),
            finite_endpoint_error_mean=float(errors[finite].mean()) if finite.any() else None,
            all_record_mean_error=float(errors.mean()) if finite.all() else None,
            scalar_fields=list(metrics),
            execution_sha=sha(out / (condition + ".npz")),
        )
    summary.update(
        task=task,
        mode=proof["mode"],
        context=proof["context"],
        development=proof["development"],
        formal_updates=0,
        checkpoint_sha=proof["checkpoint_sha"],
        scalar_schema_version=3,
        executor_sha=sha(Path(__file__)),
        dataset_contract=proof.get("dataset_contract"),
        source_contract=proof.get("source_contract"),
        condition=proof.get("condition"),
        split=proof.get("split"),
        prediction_sha=sha(pred / "PREDICTION.json"),
        clipped_records=proof["clipped_records"],
        note="Same native simulator at two timesteps; independent replay verification is separate.",
    )
    (out / "EXECUTION.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
