"""Paired frozen-policy geometry OOD; no training or source-data writes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

TASKS = {
    "throw": ("throwing", "fixed_test.npz", 2026092501),
    "rebound": ("adaptive_rebound", "test.npz", 2026092502),
    "pushing": ("pushing", "test.npz", 2026092503),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def paired_query(task, z):
    q = z["query"].copy()
    rng = np.random.default_rng(TASKS[task][2])
    changes = {}
    for i, (family, qi) in enumerate(zip(z["family"], z["query_index"])):
        key = (int(family), int(qi))
        if key not in changes:
            changes[key] = rng.uniform(0.05, 0.10, 2)
        u, v = changes[key]
        if task == "throw":
            q[i, 3] = 1.5 + 0.8 * u
        elif task == "rebound":
            q[i, 6] = 1.2 + 0.4 * u
            angle = np.copysign(0.5 + v, np.arctan2(q[i, 8], q[i, 7]))
            q[i, 7:9] = np.cos(angle), np.sin(angle)
        else:
            q[i, 2] = q[i, 0] + 0.34 + 0.16 * u
    changed = {"throw": [3], "rebound": [6, 7, 8], "pushing": [2]}[task]
    if task == "throw":
        assert np.all((q[:, 3] >= 1.54) & (q[:, 3] <= 1.58))
    elif task == "rebound":
        angle = np.abs(np.arctan2(q[:, 8], q[:, 7]))
        assert np.all((q[:, 6] >= 1.22) & (q[:, 6] <= 1.24))
        assert np.all((angle >= 0.55) & (angle <= 0.60))
    else:
        distance = q[:, 2] - q[:, 0]
        assert np.all((distance >= 0.348 - 1e-12) & (distance <= 0.356 + 1e-12))
    kept = [j for j in range(q.shape[1]) if j not in changed]
    assert np.array_equal(q[:, kept], z["query"][:, kept])
    seen = {}
    for i, (family, qi) in enumerate(zip(z["family"], z["query_index"])):
        key = (int(family), int(qi))
        if key in seen:
            assert np.array_equal(q[i], q[seen[key]])
        seen[key] = i
    return q


def init_physics(source):
    sys.path.insert(0, str(Path(source)))


def execute(job):
    task, i, q, params, action = job
    physics = importlib.import_module("physics")
    oracle = action is None
    if task == "throw":
        wind, beta = params
        if oracle:
            action = physics.oracle(q[:3], q[3:5], q[5], wind, beta)
        r = physics.flight(q[:3], [*action, 4.5], q[5], wind, beta, seconds=1.1)
        final = np.asarray(r["landing"])
        _, independent = physics.fast_trajectory(q[:3], [*action, 4.5], q[5], wind, beta, steps=275)
        assert np.linalg.norm(final - independent) < 1e-6
        error = float(np.linalg.norm(final - q[3:5]))
        hit, speed = True, 0.0
        success = error <= 0.2
        if oracle:
            assert error < 1e-6
    elif task == "rebound":
        damping, friction = params
        angle = float(np.arctan2(q[8], q[7]))
        if oracle:
            action, _ = physics.oracle(q[3:5], q[5], q[6], angle, damping, friction)
            assert action is not None
        r = physics.simulate(action, q[5], q[6], angle, damping, friction, goal=q[3:5])
        final = np.asarray(r["final"])
        error = float(np.linalg.norm(final - q[3:5]))
        hit, speed = bool(r["hit"]), 0.0
        success = hit and error <= 0.1
        if oracle:
            assert hit and error < 0.015, (i, error, "oracle failure")
    else:
        mu = params[0]
        if oracle:
            scalar, _ = physics.expert(q[2], mu, q[4], q[5], q[:2])
            action = [scalar]
        r = physics.simulate([float(action[0]), 0], mu, q[4], q[5], q[:2], goal=q[2:4])
        final = np.asarray(r["final"])
        error = float(np.linalg.norm(final[:2] - q[2:4]))
        hit, speed = bool(r["contact"]), float(np.linalg.norm(r["velocity"]))
        success = hit and error <= 0.025 and speed <= 0.03
        if oracle:
            assert hit and error < 0.01 and speed < 0.03, (i, error, speed, "oracle failure")
    assert np.isfinite(error) and np.isfinite(speed) and np.isfinite(action).all()
    return dict(
        index=i,
        action=np.asarray(action).tolist(),
        final=final.tolist(),
        error_m=error,
        contact=hit,
        speed_mps=speed,
        success=bool(success),
    )


def jobs(task, z, queries, actions=None):
    for i, q in enumerate(queries):
        params = (
            [z["wind"][i], z["beta"][i]]
            if task == "throw"
            else [z["damping"][i], z["friction"][i]] if task == "rebound" else [z["mu"][i]]
        )
        yield task, i, q, params, None if actions is None else actions[i]


def model_run(task, mode, seed):
    stored = (
        {"motion": "m", "video": "v", "full": "vm"}.get(mode, mode) if task == "pushing" else mode
    )
    return ("fixed_" if task == "throw" else "") + f"{stored}_seed{seed}", stored


def predict(model, task, z, query, mode, intervention):
    import torch

    actions = []
    with torch.inference_mode():
        for off in range(0, len(query), 64):
            ids = np.arange(off, min(off + 64, len(query)))
            vi = z["video_index"][ids].copy()
            donor_ids = ids.copy()
            if intervention == "wrong":
                vi ^= 1
                donor_ids ^= 4
                assert np.array_equal(z["family"][ids], z["family"][donor_ids])
                assert np.array_equal(z["query_index"][ids], z["query_index"][donor_ids])
            effective = "none" if intervention == "absent" else mode

            def tensor(a):
                return torch.as_tensor(a, device="cuda", dtype=torch.float32)

            video = tensor(z["video"][vi]).permute(0, 1, 4, 2, 3) / 255
            motion = tensor(z["motion"][vi if task == "pushing" else donor_ids])
            q, probe = tensor(query[ids]), tensor(z["probe"][ids])
            if task == "pushing":
                tokens = torch.as_tensor(z["tokens"][vi], device="cuda", dtype=torch.long)
                a, _ = model(video, motion, tokens, q, probe, effective)
            else:
                a, _ = model(video, motion, q, probe, effective)
            actions.append(a.cpu().numpy())
    return np.concatenate(actions)


def summarize(out, z, metrics):
    labels = ("none", "motion", "video", "full", "full_absent", "full_wrong")
    family = z["family"]
    families = np.unique(family)
    rng = np.random.default_rng(2026092599)
    boot = rng.integers(0, len(families), (2000, len(families)))
    result = {}
    family_rates = {}
    queries = np.load(out / "paired_queries.npz")
    for label in labels:
        by_condition = {}
        for condition in ("id", "ood"):
            rates = []
            all_scores = []
            for seed in range(1):
                rows = json.loads((out / f"{label}_seed{seed}_{condition}.json").read_text())
                scores = np.array([r["success"] for r in rows], dtype=float)
                # Reconstruct scoring from stored outcomes, not stored success flags.
                task = metrics["task"]
                q = queries[condition + "_query"]
                goal = q[:, 2:4] if task == "pushing" else q[:, 3:5]
                calculated_errors = np.linalg.norm(
                    np.array([r["final"][:2] for r in rows]) - goal, axis=1
                )
                assert np.allclose(
                    calculated_errors, [r["error_m"] for r in rows], atol=1e-12, rtol=0
                )
                thresholds = {"throw": 0.2, "rebound": 0.1, "pushing": 0.025}
                recomputed = np.array(
                    [
                        r["error_m"] <= thresholds[task]
                        and r["contact"]
                        and (task != "pushing" or r["speed_mps"] <= 0.03)
                        for r in rows
                    ]
                )
                assert np.array_equal(recomputed, scores)
                rates.append(float(scores.mean() * 100))
                all_scores.append(scores)
            mean = np.mean(all_scores, axis=0)
            fam = np.array([mean[family == f].mean() for f in families])
            family_rates[label, condition] = fam
            ci = np.percentile(fam[boot].mean(axis=1) * 100, [2.5, 97.5])
            by_condition[condition] = dict(
                seed_success_pct=rates,
                mean_success_pct=float(mean.mean() * 100),
                family_bootstrap_95ci=ci.tolist(),
            )
        result[label] = by_condition
    contrasts = {}
    pairs = [(f"{label}: OOD-ID", (label, "ood"), (label, "id")) for label in labels]
    pairs += [
        (f"OOD: {label}-none", (label, "ood"), ("none", "ood"))
        for label in ("motion", "video", "full")
    ]
    for title, a, b in pairs:
        delta = family_rates[a] - family_rates[b]
        contrasts[title] = dict(
            mean_pp=float(delta.mean() * 100),
            paired_family_bootstrap_95ci=np.percentile(
                delta[boot].mean(axis=1) * 100, [2.5, 97.5]
            ).tolist(),
        )
    write_json(
        out / "SUMMARY.json",
        dict(
            **metrics,
            results=result,
            contrasts=contrasts,
            interval_scope="Family bootstrap after averaging model seeds; does not capture training-seed uncertainty.",
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = args.base / TASKS[args.task][0]
    tracked = [source / "data" / TASKS[args.task][1]]
    for name in ("model.py", "physics.py"):
        tracked.append(source / name)
    if args.task == "pushing":
        tracked.append(source / "language.py")
    for mode in ("none", "motion", "video", "full"):
        for seed in range(1):
            run, _ = model_run(args.task, mode, seed)
            tracked.append(source / "runs" / run / "checkpoint.pt")
    hashes = {str(f): sha(f) for f in tracked}
    write_json(args.out / "source_hashes.json", hashes)
    init_physics(source)
    import torch
    from model import Model

    torch.set_num_threads(1)
    torch.manual_seed(20260925)
    z = dict(np.load(tracked[0]))
    assert len(z["query"]) == 2048 and len(np.unique(z["family"])) == 256
    if args.smoke:
        ids = np.flatnonzero(np.isin(z["family"], np.unique(z["family"])[:2]))
        assert np.array_equal(ids, np.arange(16))
        n = len(z["query"])
        z = {k: (v[ids] if len(v) == n else v) for k, v in z.items()}
    query = paired_query(args.task, z)
    np.savez_compressed(
        args.out / "paired_queries.npz",
        id_query=z["query"],
        ood_query=query,
        family=z["family"],
        sibling=z["sibling"],
        query_index=z["query_index"],
    )
    configs = [(m, "matched") for m in ("none", "motion", "video", "full")]
    configs += [("full", "absent"), ("full", "wrong")]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_physics,
        initargs=(str(source),),
    ) as pool:
        oracle = list(pool.map(execute, jobs(args.task, z, query), chunksize=16))
        write_json(args.out / "oracle.json", oracle)
        write_json(args.out / "progress.json", dict(stage="oracle_pass", records=len(oracle)))
        for mode, intervention in configs:
            for seed in range(1):
                run, stored_mode = model_run(args.task, mode, seed)
                ck = torch.load(
                    source / "runs" / run / "checkpoint.pt", map_location="cpu", weights_only=False
                )
                assert ck["step"] == 4096
                model = Model().cuda()
                model.load_state_dict(ck["model"])
                model.eval()
                del ck
                for condition, q in [("id", z["query"]), ("ood", query)]:
                    label = mode + ("" if intervention == "matched" else "_" + intervention)
                    stem = f"{label}_seed{seed}_{condition}"
                    actions = predict(model, args.task, z, q, stored_mode, intervention)
                    if condition == "id" and intervention == "matched":
                        reference = np.load(source / "runs" / run / "test/predictions.npz")[
                            "action"
                        ][: len(q)]
                        assert np.allclose(actions, reference, atol=1e-5, rtol=1e-5), (
                            stem,
                            float(np.max(np.abs(actions - reference))),
                        )
                    rows = list(pool.map(execute, jobs(args.task, z, q, actions), chunksize=16))
                    for r in rows:
                        i = r["index"]
                        r.update(
                            family=int(z["family"][i]),
                            sibling=int(z["sibling"][i]),
                            query_index=int(z["query_index"][i]),
                        )
                    write_json(args.out / (stem + ".json"), rows)
                    progress = dict(
                        stage="evaluation",
                        last=stem,
                        records=len(rows),
                        success_pct=float(np.mean([r["success"] for r in rows]) * 100),
                        elapsed_s=time.time() - started,
                    )
                    write_json(args.out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                del model
    assert hashes == {str(f): sha(f) for f in tracked}
    metrics = dict(
        task=args.task,
        records=len(query),
        families=len(np.unique(z["family"])),
        training_seeds=1,
        smoke=args.smoke,
        zero_updates=True,
        checkpoint_step=4096,
        original_files_unchanged=True,
        elapsed_s=time.time() - started,
    )
    summarize(args.out, z, metrics)
    write_json(args.out / "COMPLETE.json", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            if out.exists():
                write_json(
                    out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc())
                )
        raise
