"""Replace only source controls/RGB in the frozen three-task data."""

import os, sys, json, time, hashlib, shutil, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "spatial_fixed"
CODE = ROOT.parent / "spatial_fixed"
sys.path.insert(0, str(CODE))
from task_adapter import expert, execute, source_video, source, target, parameters, QUERY_DIMS
from action_limits import BOUNDS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    p = Path(p)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(j, indent=2))
    tmp.replace(p)


def adapt(task, q, old_action, physics, style, old_frame):
    a, r, details = expert(task, q, physics, old_action)
    lo, hi = BOUNDS[task]
    legal = lambda a: bool(np.isfinite(a).all() and np.all(a >= lo) and np.all(a <= hi))
    fallback = False
    if not r["success"] or not legal(a):
        from finite_reference import solve

        a, r, details = solve(task, q, physics, old_action)
        fallback = True
    fine = execute(task, q, a, physics, dt=dict(gate=0.002, bank=0.0005, ramp=0.00025)[task])
    proof = dict(
        action=np.asarray(a).tolist(),
        success=bool(r["success"]),
        fine_success=bool(fine["success"]),
        legal=legal(a),
        error=float(r["endpoint_error"]),
        fine_error=float(fine["endpoint_error"]),
        fallback=fallback,
    )
    if not (r["success"] and fine["success"] and legal(a)):
        raise RuntimeError(
            json.dumps(
                dict(
                    task=task,
                    source_query=np.asarray(q).tolist(),
                    physics=np.asarray(physics).tolist(),
                    source_result=proof,
                )
            )
        )
    if task == "gate":
        motion = np.pad(np.r_[q[:3], a], (0, 3))
    elif task == "bank":
        motion = np.r_[q[:3], a, q[6:12]]
    else:
        motion = np.pad(np.r_[q[:6], q[9:12], a], (0, 1))
    v, response = source_video(task, q, a, physics, style=style)
    assert np.array_equal(v[0], old_frame), "Source initial RGB changed"
    return np.asarray(a), motion, v, response, proof


def shard(job):
    task, old_path, out_path = job
    old_path = Path(old_path)
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=False)
    old_meta = json.loads((old_path / "COMPLETE.json").read_text())
    s = dict(np.load(old_path / "source_audit.npz"))
    original = np.load(old_path / "source_rgb.npy", mmap_mode="r")
    videos = np.lib.format.open_memmap(
        out / "source_rgb.npy", mode="w+", dtype=np.uint8, shape=original.shape
    )
    motions = []
    actions = []
    responses = []
    proofs = []
    try:
        for i in range(len(original)):
            a, m, v, response, p = adapt(
                task, s["query"][i], s["action"][i], s["physics"][i], (i // 2) % 3, original[i, 0]
            )
            videos[i] = v
            motions.append(m)
            actions.append(a)
            responses.append(response)
            proofs.append(p)
            if i % 8 == 0:
                write(out / "progress.json", dict(sources=i + 1, total=len(original)))
    except BaseException as e:
        write(out / "SOURCE_FAILURE.json", dict(index=i, error=repr(e), all_prior_sources=proofs))
        raise
    videos.flush()
    np.save(out / "source_motion.npy", np.asarray(motions))
    np.savez_compressed(
        out / "source_audit.npz",
        **{
            **s,
            "fixed_action": s["action"],
            "action": np.asarray(actions),
            "motion": np.asarray(motions),
            "response": np.asarray(responses),
        },
    )
    for name in ["records.npz", "generation.jsonl"]:
        os.link((old_path / name).resolve(), out / name)
    meta = {
        **old_meta,
        "source_policy": "adaptive_known_physics_expert",
        "initial_frame_and_imposed_motion_siblings_equal": False,
        "initial_frame_siblings_equal": True,
        "source_motion_contains_response": False,
        "learner_records_unchanged": True,
        "original_source_directory": str(old_path),
        "adaptive_source_goal_already_visible": True,
    }
    meta["files"] = {name: sha(out / name) for name in old_meta["files"]}
    assert sha(out / "records.npz") == sha(old_path / "records.npz")
    meta["source_policy_proofs"] = proofs
    write(out / "COMPLETE.json", meta)
    return str(out)


def eval_sources(task):
    original = ROOT.parent / "spatial_paired/formal" / task
    out = ROOT / task / "evaluation"
    out.mkdir(parents=True, exist_ok=False)
    r = dict(np.load(original / "records.npz"))
    old_video = np.load(original / "source_rgb.npy", mmap_mode="r")
    old_motion = np.load(original / "source_motion.npy")
    videos = np.lib.format.open_memmap(
        out / "source_rgb.npy", mode="w+", dtype=np.uint8, shape=old_video.shape
    )
    rng = np.random.default_rng(dict(gate=2309301000, bank=2309302000, ramp=2309303000)[task])
    motions = []
    proof = []
    for f in range(128):
        sq, sa, sm = source(task, rng)
        qs = [target(task, rng, ood=False) for _ in range(4)]
        ps, _ = parameters(task, rng)
        for si in range(2):
            sid = 2 * f + si
            ids = np.flatnonzero((r["source_id"] == sid) & (r["condition"] == 0))
            assert np.allclose(old_motion[sid], sm, atol=1e-7, rtol=1e-7)
            assert all(
                np.array_equal(r["query"][i, : QUERY_DIMS[task]], qs[int(r["query_index"][i])][0])
                for i in ids
            )
            assert all(np.array_equal(r["physics"][i, : len(ps[si])], ps[si]) for i in ids)
            a, m, v, response, p = adapt(task, sq, sa, ps[si], f % 3, old_video[sid, 0])
            videos[sid] = v
            motions.append(m)
            proof.append(p)
        write(out / "source_progress.json", dict(families=f + 1, total=128))
    videos.flush()
    np.save(out / "source_motion.npy", np.asarray(motions, dtype=np.float32))
    os.link((original / "records.npz").resolve(), out / "records.npz")
    write(
        out / "DATA_COMPLETE.json",
        dict(
            records=len(r["query"]),
            families=128,
            source_policy="adaptive",
            learner_records_unchanged=True,
            original_directory=str(original),
            source_policy_proofs=proof,
            files={n: sha(out / n) for n in ["records.npz", "source_motion.npy", "source_rgb.npy"]},
        ),
    )


def training(task, workers=6):
    freeze = json.loads((OLD / "FORMAL_FREEZE.json").read_text())
    old_paths = freeze["data"][task]
    jobs = [(task, p, str(ROOT / task / "train" / f"{i:02d}")) for i, p in enumerate(old_paths)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        fs = [pool.submit(shard, j) for j in jobs]
        for i, f in enumerate(as_completed(fs)):
            p = f.result()
            write(
                ROOT / task / "generation_progress.json",
                dict(completed_shards=i + 1, total=32, last=p),
            )
    paths = [j[2] for j in jobs]
    metas = {str(Path(p) / "COMPLETE.json"): sha(Path(p) / "COMPLETE.json") for p in paths}
    raw = {}
    for p in paths:
        meta = json.loads((Path(p) / "COMPLETE.json").read_text())
        raw.update({str((Path(p) / n).resolve()): h for n, h in meta["files"].items()})
    new = {
        "ready": True,
        "data": {task: paths},
        "data_manifests": metas,
        "raw_data_files": raw,
        "code": {p.name: sha(p) for p in CODE.glob("*.py")},
        "source_policy": "adaptive",
        "seed": 0,
        "updates_per_run": 4096,
    }
    write(ROOT / task / "FORMAL_FREEZE.json", new)
    write(
        ROOT / task / "DATA_COMPLETE.json",
        dict(shards=32, records=16384, all_learner_records_unchanged=True),
    )


if __name__ == "__main__":
    task = sys.argv[1]
    training(task)
    eval_sources(task)
