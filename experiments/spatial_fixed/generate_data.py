"""Native simulation data. RGB is saved; learned encoding stays online.

Every sampled family is retained, including solver failures. A failed expert
prevents the completion marker; no outcome-based query/physics resampling.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from task_adapter import (
    target,
    parameters,
    source,
    source_video,
    expert,
    execute,
    ACTION_DIMS,
    QUERY_DIMS,
)
from action_limits import BOUNDS

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["gate", "bank", "ramp"])
    parser.add_argument("--families", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--geometry-ood", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--split", choices=["training", "validation", "test"], default="training")
    args = parser.parse_args()
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    n = args.families * 8
    videos = np.lib.format.open_memmap(
        out / "source_rgb.npy", mode="w+", dtype=np.uint8, shape=(args.families * 2, 16, 96, 96, 3)
    )
    rows = []
    sources = []
    start = time.time()
    failures = []
    sampling_rejections = 0
    audit = (out / "generation.jsonl").open("w")
    for family in range(args.families):
        # Source commands and all target queries are sampled before physics.
        sq, sa, sm = source(args.task, rng)
        queries = [target(args.task, rng, ood=args.geometry_ood) for _ in range(4)]
        physics, strata = parameters(args.task, rng)
        for sibling in range(2):
            sid = family * 2 + sibling
            p = physics[sibling]
            frames, response = source_video(args.task, sq, sa, p, style=family % 3)
            videos[sid] = frames
            sources.append(dict(query=sq, action=sa, motion=sm, response=response, physics=p))
            for qi, (q, nominal, rejections) in enumerate(queries):
                if args.split == "training":
                    a, result, solver = expert(args.task, q, p, nominal)
                else:
                    from finite_reference import solve

                    a, result, solver = solve(args.task, q, p, nominal)
                prior = execute(args.task, q, nominal, p)
                validation_dt = dict(gate=0.002, bank=0.0005, ramp=0.00025)[args.task]
                finer = execute(args.task, q, a, p, dt=validation_dt)
                good = bool(result["success"])
                fine_good = bool(finer["success"])
                lo, hi = BOUNDS[args.task]
                legal = bool(
                    np.isfinite(a).all()
                    and np.isfinite(nominal).all()
                    and np.all(a >= lo)
                    and np.all(a <= hi)
                    and np.all(nominal >= lo)
                    and np.all(nominal <= hi)
                )
                if not good or not fine_good or not legal:
                    failures.append([family, sibling, qi, good, fine_good, legal])
                row = dict(
                    family=family,
                    sibling=sibling,
                    query_index=qi,
                    source_id=sid,
                    query=np.pad(q, (0, 16 - len(q))),
                    action=np.pad(a, (0, 6 - len(a))),
                    nominal_action=np.pad(nominal, (0, 6 - len(nominal))),
                    physics=np.pad(p, (0, 4 - len(p))),
                    stratum=int(strata[sibling]),
                    oracle_success=good,
                    fine_oracle_success=fine_good,
                    nominal_success=bool(prior["success"]),
                    action_limits_valid=legal,
                    endpoint_error=result["endpoint_error"],
                    fine_error=finer["endpoint_error"],
                    nominal_error=prior["endpoint_error"],
                    **solver,
                )
                rows.append(row)
                audit.write(
                    json.dumps(
                        {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in row.items()
                        }
                    )
                    + "\n"
                )
            if sibling == 0:
                sampling_rejections += sum(qr[2] for qr in queries)
        assert np.array_equal(
            videos[family * 2, 0], videos[family * 2 + 1, 0]
        ), "Initial-frame physics leakage"
        assert np.array_equal(sources[-1]["motion"], sources[-2]["motion"])
        audit.flush()
        videos.flush()
        progress = dict(
            pid=os.getpid(),
            families_complete=family + 1,
            families_total=args.families,
            records_complete=len(rows),
            expert_failures=len(failures),
            elapsed_seconds=time.time() - start,
        )
        (out / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
    audit.close()
    keys = [
        "family",
        "sibling",
        "query_index",
        "source_id",
        "query",
        "action",
        "nominal_action",
        "physics",
        "stratum",
        "oracle_success",
        "fine_oracle_success",
        "nominal_success",
        "action_limits_valid",
        "endpoint_error",
        "fine_error",
        "nominal_error",
    ]
    arrays = {key: np.array([r[key] for r in rows]) for key in keys}
    np.savez_compressed(out / "records.npz", **arrays)
    np.savez_compressed(
        out / "source_audit.npz", **{key: np.array([s[key] for s in sources]) for key in sources[0]}
    )
    # Deliberately separate allowed model inputs from source response/physics audit.
    np.save(out / "source_motion.npy", np.array([s["motion"] for s in sources], dtype=np.float32))
    code = {p.name: sha(p) for p in HERE.glob("*.py")}
    files = {
        name: sha(out / name)
        for name in [
            "source_rgb.npy",
            "source_motion.npy",
            "records.npz",
            "source_audit.npz",
            "generation.jsonl",
        ]
    }
    proof = dict(
        task=args.task,
        seed=args.seed,
        families=args.families,
        records=n,
        stage="development" if args.development else "frozen-data-generation",
        formal_updates=0,
        split=args.split,
        geometry_ood=args.geometry_ood,
        action_dim=ACTION_DIMS[args.task],
        query_dim=QUERY_DIMS[args.task],
        evaluation_plan_sha=(
            sha(HERE / "EVALUATION_PLAN.json") if args.split != "training" else None
        ),
        action_field_role=(
            "successful_training_label"
            if args.split == "training"
            else "finite_reference_output_not_ground_truth"
        ),
        reference_version=1 if args.split != "training" else None,
        bank_label_version=2 if args.task == "bank" and args.split == "training" else None,
        half_step_used_in_planning=args.task == "bank" and args.split == "training",
        legacy_oracle_fields_are_reference_outcomes=args.split != "training",
        source_motion_contains_response=False,
        source_response_audit_not_model_input=True,
        additional_nominal_action_labels=True,
        initial_frame_and_imposed_motion_siblings_equal=True,
        validation_timestep=dict(gate=0.002, bank=0.0005, ramp=0.00025)[args.task],
        nominal_query_sampling_rejections=sampling_rejections,
        actual_physics_resampling=0,
        nominal_query_sampling_policy=(
            "fixed_geometry_command_search"
            if args.task == "bank" and args.geometry_ood
            else "original_nominal_sampler"
        ),
        oracle_successes=int(arrays["oracle_success"].sum()),
        fine_successes=int(arrays["fine_oracle_success"].sum()),
        action_limits_valid=bool(arrays["action_limits_valid"].all()),
        nominal_successes=int(arrays["nominal_success"].sum()),
        failures=failures,
        elapsed_seconds=time.time() - start,
        files=files,
        code=code,
    )
    (out / "GENERATION.json").write_text(json.dumps(proof, indent=2))
    if failures and args.split == "training":
        raise RuntimeError(f"{len(failures)} expert/fine executions failed; all records retained")
    marker = "COMPLETE.json" if args.split == "training" else "EVALUATION_COMPLETE.json"
    (out / marker).write_text(json.dumps(proof, indent=2))
    print("SPATIAL_NATIVE_DATA_OK", flush=True)


if __name__ == "__main__":
    main()
