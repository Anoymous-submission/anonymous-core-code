"""Paired geometry-only evaluation, frozen weights; no optimizer or training."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, time, hashlib, argparse, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "spatial_fixed"
CODE = OLD
sys.path.insert(0, str(CODE))
from task_adapter import target, source, parameters, source_video, execute, QUERY_DIMS
from action_limits import decode

TASKS = ["gate", "bank", "ramp"]
CONDS = ["id", "factor1", "factor2", "both"]
SEEDS = dict(gate=2309301000, bank=2309302000, ramp=2309303000)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(j, indent=2, allow_nan=False))
    tmp.replace(p)


def variants(task, q, rng):
    a = q.copy()
    b = q.copy()
    sgn = lambda x: 1 if x >= 0 else -1
    if task == "gate":
        a[4] = sgn(q[4]) * rng.uniform(0.36, 0.40)
        b[7] = sgn(q[7]) * rng.uniform(0.61, 0.66)
    elif task == "bank":
        a[9] = sgn(q[9]) * rng.uniform(0.305, 0.325)
        b[10] = sgn(q[10]) * rng.uniform(0.165, 0.180)
    else:
        a[3] = rng.uniform(0.365, 0.380)
        b[4] = sgn(q[4]) * rng.uniform(0.305, 0.325)
    both = q + (a - q) + (b - q)
    return [q.copy(), a, b, both]


def generate(task, out, n, smoke):
    folder = out / task
    folder.mkdir(exist_ok=False)
    rng = np.random.default_rng(SEEDS[task] + (100000 if smoke else 0))
    irng = np.random.default_rng(SEEDS[task] + 555)
    videos = np.lib.format.open_memmap(
        folder / "source_rgb.npy", mode="w+", dtype=np.uint8, shape=(2 * n, 16, 96, 96, 3)
    )
    rows = []
    motions = []
    for f in range(n):
        sq, sa, sm = source(task, rng)
        qs = [target(task, rng, ood=False) for _ in range(4)]
        ps, _ = parameters(task, rng)
        paired = [variants(task, q, irng) for q, _, _ in qs]
        for si in range(2):
            videos[2 * f + si] = source_video(task, sq, sa, ps[si], style=f % 3)[0]
            motions.append(sm)
        assert np.array_equal(videos[2 * f, 0], videos[2 * f + 1, 0])
        for cond in range(4):
            for si in range(2):
                for qi, (_, nom, _) in enumerate(qs):
                    rows.append(
                        dict(
                            query=np.pad(paired[qi][cond], (0, 16 - len(paired[qi][cond]))),
                            physics=np.pad(ps[si], (0, 4 - len(ps[si]))),
                            nominal=np.pad(nom, (0, 6 - len(nom))),
                            family=f,
                            sibling=si,
                            query_index=qi,
                            condition=cond,
                            source_id=2 * f + si,
                        )
                    )
        if f % 8 == 0 or f + 1 == n:
            write(
                folder / "generation_progress.json", dict(pid=os.getpid(), families=f + 1, total=n)
            )
    videos.flush()
    np.save(folder / "source_motion.npy", np.asarray(motions, np.float32))
    arrays = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    for f in range(n):
        ids = np.flatnonzero(arrays["family"] == f)
        for k in ["physics", "nominal", "source_id", "sibling", "query_index"]:
            x = arrays[k][ids].reshape((4, 8) + arrays[k].shape[1:])
            assert all(np.array_equal(x[0], x[c]) for c in range(4))
        q = arrays["query"][ids].reshape(4, 8, 16)
        allowed = {"gate": [4, 7], "bank": [9, 10], "ramp": [3, 4]}[task]
        keep = [k for k in range(16) if k not in allowed]
        assert all(np.array_equal(q[0][:, keep], q[c][:, keep]) for c in range(4))
    np.savez_compressed(folder / "records.npz", **arrays)
    write(
        folder / "DATA_COMPLETE.json",
        dict(
            records=len(rows),
            families=n,
            seed=SEEDS[task] + (100000 if smoke else 0),
            paired_invariants=True,
            files={
                name: sha(folder / name)
                for name in ["records.npz", "source_rgb.npy", "source_motion.npy"]
            },
            formal_updates=0,
        ),
    )


def predict(task, out):
    import torch
    from policy_moments import Policy

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    folder = out / task
    with np.load(folder / "records.npz") as f:
        r = {k: f[k] for k in f.files}
    video = np.load(folder / "source_rgb.npy", mmap_mode="r")
    motion = np.load(folder / "source_motion.npy")
    dest = folder / "predictions"
    dest.mkdir()
    for mode in ["none", "motion", "video", "full"]:
        for seed in range(3):
            ck = OLD / "runs" / f"{task}_{mode}_seed{seed}" / "final.pt"
            saved = torch.load(ck, map_location="cpu", weights_only=False)
            assert saved["step"] == 4096
            model = Policy().cuda().eval()
            model.load_state_dict(saved["model"])
            results = {ctx: [] for ctx in (["match", "absent"] if mode == "full" else ["match"])}
            with torch.inference_mode():
                for start in range(0, len(r["query"]), 64):
                    sl = slice(start, start + 64)
                    ids = r["source_id"][sl]
                    q = torch.as_tensor(r["query"][sl], dtype=torch.float32, device="cuda")
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
                        torch.as_tensor(motion[ids] / 8, device="cuda")
                        if mode in ["motion", "full"]
                        else q.new_zeros((n, 12))
                    )
                    for ctx in results:
                        present = (
                            torch.ones(n, device="cuda")
                            if ctx == "match"
                            else torch.zeros(n, device="cuda")
                        )
                        base, res = model(
                            q,
                            m if ctx == "match" else torch.zeros_like(m),
                            v if ctx == "match" else torch.zeros_like(v),
                            present,
                            "target_only" if mode == "none" else mode,
                        )
                        results[ctx].append(((base + res) * 8).cpu().numpy())
            for ctx, parts in results.items():
                raw = np.concatenate(parts)
                actions = decode(raw, task)
                name = f"{mode}_s{seed}_{ctx}"
                np.savez_compressed(dest / (name + ".npz"), raw=raw, executed=actions)
                write(
                    dest / (name + ".json"),
                    dict(
                        checkpoint=str(ck),
                        checkpoint_sha=sha(ck),
                        data_sha=sha(folder / "records.npz"),
                        prediction_sha=sha(dest / (name + ".npz")),
                        formal_updates=0,
                        records=len(raw),
                    ),
                )
            del model, saved
            torch.cuda.empty_cache()
    write(
        folder / "PREDICT_COMPLETE.json", dict(cells=15, records=len(r["query"]), formal_updates=0)
    )


def work(job):
    task, folderstr, kind, name, start, end = job
    folder = Path(folderstr)
    from finite_reference import solve

    with np.load(folder / "records.npz") as f:
        r = {k: f[k] for k in f.files}
    if kind == "policy":
        with np.load(folder / "predictions" / (name + ".npz")) as f:
            cmd = f["executed"]
    rows = []
    paths = []
    acts = []
    for i in range(start, end):
        q = r["query"][i, : QUERY_DIMS[task]]
        p = r["physics"][i]
        if kind == "policy":
            a = cmd[i]
        elif kind == "nominal":
            a = decode(r["nominal"][i : i + 1], task)[0]
        else:
            a, _, _ = solve(task, q, p, r["nominal"][i, : {"gate": 6, "bank": 3, "ramp": 2}[task]])
        acts.append(a)
        pair = []
        trace = []
        for dt in [
            dict(gate=0.004, bank=0.001, ramp=0.001)[task],
            dict(gate=0.002, bank=0.0005, ramp=0.00025)[task],
        ]:
            z = execute(task, q, a, p, dt=dt)
            pair.append(
                [
                    float(z["success"]),
                    z["endpoint_error"],
                    float(z.get("hit_panel", True)),
                    float(z.get("departed_downhill_edge", True)),
                    float(z.get("other_contact_steps", z.get("gate_contact_steps", 0))),
                ]
            )
            trace.append(z["positions"][np.linspace(0, len(z["positions"]) - 1, 17, dtype=int)])
        rows.append(pair)
        paths.append(trace)
    path = folder / "execution" / f"{name}_{start:05d}_{end:05d}.npz"
    np.savez_compressed(
        path,
        record=np.arange(start, end),
        metrics=np.asarray(rows),
        positions=np.asarray(paths),
        executed=np.asarray(acts),
    )
    return str(path)


def collect(out):
    report = {"formal_updates": 0, "tasks": {}}
    from paired_uncertainty import pack_success, paired_interval

    for task in TASKS:
        folder = out / task
        with np.load(folder / "records.npz") as f:
            r = {k: f[k] for k in f.files}
        names = [
            f"{m}_s{s}_{ctx}"
            for m in ["none", "motion", "video", "full"]
            for s in range(3)
            for ctx in (["match", "absent"] if m == "full" else ["match"])
        ] + ["nominal", "reference"]
        cells = {}
        packed = {}
        for name in names:
            chunks = []
            ids = []
            for f in sorted((folder / "execution").glob(name + "_*.npz")):
                with np.load(f) as z:
                    ids.extend(z["record"].tolist())
                    chunks.append(z["metrics"])
            assert ids == list(range(len(r["query"])))
            m = np.concatenate(chunks)
            for c, condition in enumerate(CONDS):
                mask = r["condition"] == c
                v = m[mask]
                cells[name + "_" + condition] = {
                    "records": int(mask.sum()),
                    "native_successes": int(v[:, 0, 0].sum()),
                    "finer_successes": int(v[:, 1, 0].sum()),
                    "native_sr": float(v[:, 0, 0].mean()),
                    "finer_sr": float(v[:, 1, 0].mean()),
                    "nonfinite_errors": int((~np.isfinite(v[:, 0, 1])).sum()),
                }
                packed[name + "_" + condition] = {
                    k: r[k][mask] for k in ["family", "sibling", "query_index"]
                }
                packed[name + "_" + condition]["success"] = v[:, 0, 0]
        comparisons = {}
        for c in CONDS:
            l, _ = pack_success([packed[f"full_s{s}_match_{c}"] for s in range(3)])
            rr, _ = pack_success([packed[f"full_s{s}_absent_{c}"] for s in range(3)])
            comparisons[c + "_context_gain"] = paired_interval(l, rr)
            if c != "id":
                for mode in ["none", "motion", "video", "full"]:
                    l, _ = pack_success([packed[f"{mode}_s{s}_match_{c}"] for s in range(3)])
                    rr, _ = pack_success([packed[f"{mode}_s{s}_match_id"] for s in range(3)])
                    comparisons[c + "_" + mode + "_minus_id"] = paired_interval(l, rr)
        report["tasks"][task] = {"cells": cells, "comparisons": comparisons}
    write(out / "RESULTS.json", report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = ROOT / ("smoke" if a.smoke else "formal")
    out.mkdir(exist_ok=False)
    freeze = json.loads((OLD / "FORMAL_FREEZE.json").read_text())
    assert all(sha(CODE / name) == digest for name, digest in freeze["code"].items())
    if not a.smoke:
        assert (ROOT / "smoke/COMPLETE.json").exists()
    write(
        out / "PLAN.json",
        dict(
            started=time.time(),
            pid=os.getpid(),
            families=2 if a.smoke else 128,
            smoke=a.smoke,
            formal_updates=0,
            script_sha=sha(__file__),
            protocol_sha=sha(ROOT / "PROTOCOL.md"),
            frozen_code=freeze["code"],
        ),
    )
    try:
        for task in TASKS:
            write(out / "progress.json", dict(stage="generate", task=task, pid=os.getpid()))
            generate(task, out, 2 if a.smoke else 128, a.smoke)
            write(out / "progress.json", dict(stage="predict", task=task, pid=os.getpid()))
            predict(task, out)
        jobs = []
        for task in TASKS:
            folder = out / task
            (folder / "execution").mkdir()
            n = 64 if a.smoke else 4096
            for path in sorted((folder / "predictions").glob("*.npz")):
                for start in range(0, n, 32):
                    jobs.append((task, str(folder), "policy", path.stem, start, min(start + 32, n)))
            for kind in ["nominal", "reference"]:
                for start in range(0, n, 16):
                    jobs.append((task, str(folder), kind, kind, start, min(start + 16, n)))
        write(out / "progress.json", dict(stage="execute", jobs_total=len(jobs), pid=os.getpid()))
        with ProcessPoolExecutor(max_workers=24, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(work, job) for job in jobs]
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if i % 8 == 0 or i + 1 == len(jobs):
                    write(
                        out / "progress.json",
                        dict(
                            stage="execute",
                            jobs_complete=i + 1,
                            jobs_total=len(jobs),
                            last=result,
                            pid=os.getpid(),
                        ),
                    )
        collect(out)
        write(
            out / "COMPLETE.json",
            dict(
                ended=time.time(),
                formal_updates=0,
                smoke=a.smoke,
                results_sha=sha(out / "RESULTS.json"),
            ),
        )
    except BaseException as e:
        write(out / "FAILURE.json", dict(error=repr(e), time=time.time()))
        raise


if __name__ == "__main__":
    main()
