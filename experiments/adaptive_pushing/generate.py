"""Change source policy only; retain every original learner record."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json, time, hashlib
import numpy as np
from physics import expert, simulate
from language import caption

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "pushing"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def work(job):
    i, mu, light = job
    speed, r = expert(0.3, mu, 0.2, 0.0)
    r = simulate([speed, 0], mu, render=True, light=light)
    error = float(np.linalg.norm(r["final"][:2] - [0.3, 0]))
    assert r["contact"] and error < 0.01 and np.linalg.norm(r["velocity"]) < 0.03, (i, error)
    text, tok = caption(r["motion"], r["states"])
    return i, speed, r["video"], r["motion"], text, tok, error


def main():
    (ROOT / "data").mkdir(exist_ok=False)
    proof = {}
    for split in ["pilot", "train", "val", "test"]:
        z = dict(np.load(OLD / "data" / f"{split}.npz"))
        old_video = z["video"]
        n = len(old_video)
        rows = [np.flatnonzero(z["video_index"] == i)[0] for i in range(n)]
        jobs = [
            (i, float(z["mu"][r]), ["neutral", "side", "warm"][int(z["source_light"][r])])
            for i, r in enumerate(rows)
        ]
        videos = np.empty_like(old_video)
        motions = np.empty_like(z["motion"])
        tokens = np.empty_like(z["tokens"])
        texts = [None] * n
        speeds = np.zeros(n)
        errors = np.zeros(n)
        with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as pool:
            for k, (i, speed, v, m, text, tok, error) in enumerate(
                pool.map(work, jobs, chunksize=8)
            ):
                assert np.array_equal(v[0], old_video[i, 0]), ("initial RGB", split, i)
                videos[i] = v
                motions[i] = m
                tokens[i] = tok
                texts[i] = text
                speeds[i] = speed
                errors[i] = error
                if k % 32 == 0:
                    (ROOT / "generation_progress.json").write_text(
                        json.dumps(dict(split=split, sources=k + 1, total=n))
                    )
        preserved = {k: sha(OLD / "data" / f"{split}.npz") for k in []}
        changed = {"video", "motion", "tokens", "captions", "source_speed"}
        original = {k: v for k, v in z.items() if k not in changed}
        z.update(
            video=videos,
            motion=motions,
            tokens=tokens,
            captions=np.asarray(texts),
            source_speed=speeds[z["video_index"]],
        )
        np.savez_compressed(ROOT / "data" / f"{split}.npz", **z)
        with np.load(ROOT / "data" / f"{split}.npz") as check:
            assert all(np.array_equal(check[k], v) for k, v in original.items())
        proof[split] = dict(
            records=len(z["query"]),
            sources=n,
            unchanged_learner_fields=sorted(original),
            initial_rgb_identical=True,
            source_goal=[0.3, 0],
            source_max_error=float(errors.max()),
            old_sha=sha(OLD / "data" / f"{split}.npz"),
            new_sha=sha(ROOT / "data" / f"{split}.npz"),
            distinct_sibling_commands=int(np.count_nonzero(speeds[::2] != speeds[1::2])),
        )
        (ROOT / "DATA_PROGRESS.json").write_text(json.dumps(proof, indent=2))
    (ROOT / "DATA_COMPLETE.json").write_text(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
