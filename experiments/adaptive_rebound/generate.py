"""Change only source demonstration policy; reuse target records exactly."""

import hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from physics import simulate, oracle

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "rebound"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(16777216), b""):
            h.update(b)
    return h.hexdigest()


def make(job):
    vi, damping, friction = job
    a, r = oracle(np.array([0.3, 0.3]), 1.0, 1.0, 0.0, damping, friction)
    error = float(np.linalg.norm(r["final"] - [0.3, 0.3]))
    assert r["hit"] and error < 0.015, (vi, error)
    v = simulate(a, 1.0, 1.0, 0.0, damping, friction, render=True, goal=[0.3, 0.3])
    assert v["video"].shape == (16, 64, 64, 3) and v["hit"]
    return vi, np.array([0, 0, 0.07, *a, 0.0]), v["video"], error


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    report = {}
    start = time.time()
    for split in ["pilot", "train", "val", "test"]:
        src = OLD / "data" / f"{split}.npz"
        source_sha = sha(src)
        z = dict(np.load(src))
        n = len(z["video"])
        indices = [int(np.flatnonzero(z["video_index"] == i)[0]) for i in range(n)]
        jobs = [
            (vi, float(z["damping"][i]), float(z["friction"][i])) for vi, i in enumerate(indices)
        ]
        # Early feasibility checks use the predetermined first pilot physics only.
        if split == "pilot":
            for job in jobs[:2]:
                make(job)
        videos = np.empty_like(z["video"])
        motions = np.empty((n, 6))
        errors = []
        with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("spawn")) as pool:
            for vi, m, v, e in pool.map(make, jobs, chunksize=8):
                videos[vi] = v
                motions[vi] = m
                errors.append(e)
                if len(errors) % 64 == 0:
                    (ROOT / "DATA_STATUS.json").write_text(
                        json.dumps(
                            dict(
                                split=split,
                                completed=len(errors),
                                total=n,
                                seconds=time.time() - start,
                            )
                        )
                    )
        assert np.array_equal(videos[:, 0], z["video"][:, 0]), "Changed initial scene"
        differing = float(
            np.mean(np.linalg.norm(motions[::2, 3:5] - motions[1::2, 3:5], axis=1) > 0)
        )
        z["video"] = videos
        z["motion"] = motions[z["video_index"]]
        out = ROOT / "data" / f"{split}.npz"
        np.savez_compressed(out, **z)
        old = np.load(src)
        kept = [k for k in old.files if k not in ["motion", "video"]]
        assert all(np.array_equal(z[k], old[k]) for k in kept)
        assert sha(src) == source_sha
        report[split] = dict(
            records=len(z["query"]),
            videos=n,
            source_sha256=source_sha,
            output_sha256=sha(out),
            max_source_error_m=max(errors),
            preserved_fields=kept,
            initial_rgb_identical=True,
            fraction_sibling_commands_different=differing,
        )
        (ROOT / "DATA_STATUS.json").write_text(
            json.dumps(
                dict(
                    completed_splits=list(report),
                    split=split,
                    completed=n,
                    total=n,
                    seconds=time.time() - start,
                )
            )
        )
    (ROOT / "DATA_COMPLETE.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
