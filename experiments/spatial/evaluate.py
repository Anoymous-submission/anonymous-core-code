import os, sys, json, hashlib, time, multiprocessing as mp, fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "spatial_fixed"
sys.path.insert(0, str(ROOT))
from task_adapter import execute, QUERY_DIMS
from action_limits import decode

METRIC_COLUMNS = [
    "success",
    "endpoint_error",
    "hit_panel",
    "departed_downhill_edge",
    "other_or_gate_contact_steps",
    "gate_error",
    "crossed_gate_aperture",
    "crossed_gate_plane",
    "required_contact_steps",
    "crossing_x",
    "crossing_y",
    "crossing_z",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    p = Path(p)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(j, indent=2))
    tmp.replace(p)


def work(job):
    task, folder, name, start, end = job
    folder = Path(folder)
    r = np.load(folder / "records.npz")
    a = np.load(folder / "predictions" / f"{name}.npz")["executed"]
    metrics = []
    positions = []
    for i in range(start, end):
        values = []
        traces = []
        for dt in [
            dict(gate=0.004, bank=0.001, ramp=0.001)[task],
            dict(gate=0.002, bank=0.0005, ramp=0.00025)[task],
        ]:
            z = execute(task, r["query"][i, : QUERY_DIMS[task]], a[i], r["physics"][i], dt=dt)
            crossing = z.get("crossing")
            crossing = np.full(3, np.nan) if crossing is None else crossing
            values.append(
                [
                    float(z["success"]),
                    z["endpoint_error"],
                    float(z.get("hit_panel", True)),
                    float(z.get("departed_downhill_edge", True)),
                    float(z.get("other_contact_steps", z.get("gate_contact_steps", 0))),
                    float(z.get("gate_error", np.nan)),
                    float(z.get("crossed_gate_aperture", False)),
                    float(z.get("crossed_gate_plane", False)),
                    float(z.get("panel_contact_steps", z.get("ramp_contact_steps", 0))),
                    *crossing,
                ]
            )
            traces.append(z["positions"][np.linspace(0, len(z["positions"]) - 1, 17, dtype=int)])
        metrics.append(values)
        positions.append(traces)
    path = folder / "execution" / f"{name}_{start:05d}_{end:05d}.npz"
    np.savez_compressed(
        path,
        record=np.arange(start, end),
        metrics=np.asarray(metrics),
        positions=np.asarray(positions),
        executed=a[start:end],
    )
    return str(path)


def main(task):
    import torch
    from policy_moments import Policy

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = {}
    cell_counts = {}
    evaluator_sha = sha(__file__)
    write(
        ROOT / task / "EXECUTION_SCHEMA.json",
        dict(
            metrics=METRIC_COLUMNS,
            timesteps=dict(gate=[0.004, 0.002], bank=[0.001, 0.0005], ramp=[0.001, 0.00025])[task],
            positions="17 evenly spaced samples of each native/finer execution; critical crossing/contact metrics additionally retained",
        ),
    )
    for policy in ["fixed", "adaptive"]:
        folder = ROOT / task / policy / "evaluation"
        r = dict(np.load(folder / "records.npz"))
        video = np.load(folder / "source_rgb.npy", mmap_mode="r")
        motion = np.load(folder / "source_motion.npy")
        context = np.load(folder / "source_context.npy")
        (folder / "predictions").mkdir()
        (folder / "execution").mkdir()
        names = []
        for mode in (
            ["none", "motion", "video", "full"]
            if policy == "fixed"
            else ["motion", "video", "full"]
        ):
            ck = ROOT / task / "runs" / f"{policy}_{mode}_seed0/final.pt"
            saved = torch.load(ck, map_location="cpu", weights_only=False)
            assert saved["step"] == 4096
            model = Policy().cuda().eval()
            model.load_state_dict(saved["model"])
            for ctx in (["match", "absent", "wrong"] if mode == "full" else ["match"]):
                pred = []
                with torch.inference_mode():
                    for start in range(0, len(r["query"]), 64):
                        sl = slice(start, start + 64)
                        ids = r["source_id"][sl].copy()
                        ids = ids ^ 1 if ctx == "wrong" else ids
                        source_geometry = context[ids]
                        assert np.array_equal(source_geometry, context[r["source_id"][sl]])
                        q = torch.as_tensor(
                            np.concatenate([r["query"][sl], source_geometry], axis=1),
                            device="cuda",
                            dtype=torch.float32,
                        )
                        n = len(q)
                        v = (
                            torch.as_tensor(np.array(video[ids]), device="cuda")
                            .permute(0, 1, 4, 2, 3)
                            .float()
                            / 255
                            if mode in ["video", "full"]
                            else q.new_zeros((n, 16, 3, 96, 96))
                        )
                        m = (
                            torch.as_tensor(motion[ids] / 8, device="cuda", dtype=torch.float32)
                            if mode in ["motion", "full"]
                            else q.new_zeros((n, 12))
                        )
                        present = torch.ones(n, device="cuda")
                        if ctx == "absent":
                            v.zero_()
                            m.zero_()
                            present.zero_()
                        b, res = model(q, m, v, present, "target_only" if mode == "none" else mode)
                        pred.append(((b + res) * 8).cpu().numpy())
                raw = np.concatenate(pred)
                name = f"{mode}_s0_{ctx}"
                names.append(name)
                np.savez_compressed(
                    folder / "predictions" / f"{name}.npz", raw=raw, executed=decode(raw, task)
                )
                write(
                    folder / "predictions" / f"{name}.json",
                    dict(
                        checkpoint=str(ck),
                        checkpoint_sha=sha(ck),
                        records=len(raw),
                        zero_updates=True,
                        source_geometry_common=True,
                    ),
                )
            del model, saved
            torch.cuda.empty_cache()
        jobs = [
            (task, str(folder), name, i, min(i + 32, len(r["query"])))
            for name in names
            for i in range(0, len(r["query"]), 32)
        ]
        # One 128-worker native execution pool at a time, within the 160-core affinity.
        # Model training and prediction can proceed on the other assigned GPUs.
        with (ROOT / "execution_cpu.lock").open("a") as lock:
            write(
                folder / "progress.json",
                dict(stage="waiting_for_execution_cpu", jobs=0, total=len(jobs)),
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
            with ProcessPoolExecutor(max_workers=128, mp_context=mp.get_context("spawn")) as pool:
                fs = [pool.submit(work, j) for j in jobs]
                for i, f in enumerate(as_completed(fs)):
                    f.result()
                    if i % 8 == 0:
                        write(
                            folder / "progress.json",
                            dict(stage="execute", jobs=i + 1, total=len(jobs), workers=128),
                        )
            fcntl.flock(lock, fcntl.LOCK_UN)
        for name in names:
            paths = sorted((folder / "execution").glob(name + "_*.npz"))
            z = [dict(np.load(p)) for p in paths]
            ids = np.concatenate([x["record"] for x in z])
            assert np.array_equal(ids, np.arange(len(r["query"])))
            v = np.concatenate([x["metrics"] for x in z])
            for c, cond in enumerate(["id", "factor1", "factor2", "both"]):
                t = v[r["condition"] == c]
                results[f"{policy}/{name}/{cond}"] = dict(
                    records=len(t),
                    native_sr_pct=float(t[:, 0, 0].mean() * 100),
                    finer_sr_pct=float(t[:, 1, 0].mean() * 100),
                    native_successes=int(t[:, 0, 0].sum()),
                )
        cell_counts[policy] = len(names) * len(r["query"])
        write(
            folder / "COMPLETE.json",
            dict(
                policy_executions=cell_counts[policy],
                native_and_finer=True,
                seed=0,
                zero_updates=True,
            ),
        )
    # Identical None model, query, source geometry, and labels for both policies.
    for cond in ["id", "factor1", "factor2", "both"]:
        results[f"adaptive/none_s0_match/{cond}"] = dict(
            results[f"fixed/none_s0_match/{cond}"], shared_same_model_and_effective_inputs=True
        )
    write(
        ROOT / task / "SUMMARY.json",
        dict(
            task=task,
            training_seed=0,
            source_schema="common_source_geometry_v1",
            results=results,
            policy_executions=cell_counts,
        ),
    )
    assert sha(__file__) == evaluator_sha, "Evaluator source changed while running"
    write(
        ROOT / task / "EVALUATION_COMPLETE.json",
        dict(
            total_policy_executions=sum(cell_counts.values()),
            native_and_finer=True,
            seed=0,
            zero_updates=True,
            execution_workers=128,
            evaluator_sha256=evaluator_sha,
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1])
