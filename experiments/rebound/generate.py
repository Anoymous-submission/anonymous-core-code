import os, json, time, argparse
from pathlib import Path
import numpy as np
from physics import simulate, oracle

ROOT = Path(__file__).resolve().parent


def generate(split, n, seed, part=None):
    rng = np.random.default_rng(seed)
    videos = []
    rows = []
    starttime = time.time()
    for f in range(n):
        queries = []
        for qi in range(4):
            L = rng.uniform(0.8, 1.2)
            angle = rng.uniform(-0.5, 0.5)
            normal = np.array([np.cos(angle), np.sin(angle)])
            tangent = np.array([-normal[1], normal[0]])
            goal = normal * rng.uniform(0.2, 0.65) + tangent * rng.uniform(-0.3, 0.3)
            mass = float(rng.choice([0.7, 1.4]))
            probe = normal * rng.uniform(1.8, 3) + tangent * rng.uniform(-0.5, 0.5)
            queries.append((goal, mass, L, angle, probe))
        for sibling in range(2):
            damping = rng.uniform(0.15, 0.4)
            friction = rng.uniform(0, 0.08)
            demo = simulate([2.0, 0.35], 1.0, 1.0, 0.0, damping, friction, render=True)
            assert (
                demo["video"].shape == (16, 64, 64, 3)
                and demo["hit"]
                and demo["outgoing"] is not None
            )
            assert np.linalg.norm(demo["outgoing"]) <= np.linalg.norm(demo["incoming"]) + 1e-6
            videos.append(demo["video"])
            for qi, (goal, mass, L, angle, probe) in enumerate(queries):
                action, expert = oracle(goal, mass, L, angle, damping, friction)
                error = float(np.linalg.norm(expert["final"] - goal))
                assert expert["hit"] and error < 0.015, (f, sibling, qi, error)
                future = simulate(probe, mass, L, angle, damping, friction)["positions"]
                rows.append(
                    dict(
                        family=f,
                        sibling=sibling,
                        query_index=qi,
                        video_index=len(videos) - 1,
                        query=[0, 0, 0.07, *goal, mass, L, np.cos(angle), np.sin(angle), 1.2],
                        action=action,
                        probe=probe,
                        future=future,
                        damping=damping,
                        friction=friction,
                        angle=angle,
                        oracle_error=error,
                        motion=[0, 0, 0.07, 2.0, 0.35, 0.0],
                    )
                )
        if f % 16 == 0:
            print(split, f, n, "seconds", time.time() - starttime, flush=True)
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["video"] = np.array(videos, dtype=np.uint8)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / (split + ("" if part is None else "_" + str(part)) + ".npz"), **data)
    (out / (split + ("" if part is None else "_" + str(part)) + ".json")).write_text(
        json.dumps(
            dict(
                families=n,
                records=len(rows),
                videos=len(videos),
                seed=seed,
                seconds=time.time() - starttime,
                max_oracle_error=float(data["oracle_error"].max()),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test", "pilot"])
    p.add_argument("--part", type=int)
    a = p.parse_args()
    n, seed = {
        "train": (2048, 99100000),
        "val": (128, 99200000),
        "test": (256, 99300000),
        "pilot": (16, 99000000),
    }[a.split]
    generate(a.split, n // 8 if a.part is not None else n, seed + (a.part or 0) * 10000, a.part)
