import os, json, time, argparse
from pathlib import Path
import numpy as np
from physics import flight, oracle, fast_trajectory

ROOT = Path(__file__).resolve().parent


def generate(split, n, seed, part=None):
    rng = np.random.default_rng(seed)
    videos = {p: [] for p in ["fixed", "aimed"]}
    rows = {p: [] for p in videos}
    starttime = time.time()
    for f in range(n):
        source = np.array([rng.uniform(-0.2, 0.1), rng.uniform(-0.2, 0.2), 0.6])
        goals = np.array([[0.8, -0.35], [1.4, 0.35]]) + rng.uniform(-0.1, 0.1, (1, 2))
        queries = []
        for q in range(4):
            pos = [rng.uniform(-0.15, 0.15), rng.uniform(-0.2, 0.2), 0.6]
            target = rng.uniform([0.7, -0.65], [1.5, 0.65])
            mass = float(rng.choice([0.7, 1.4]))
            probe = rng.uniform([-0.5, -1.5], [3, 1.5])
            queries.append((pos, target, mass, probe))
        for sibling in range(2):
            wind = rng.uniform([-1.5, -2], [1.5, 2])
            beta = float(rng.uniform(0.1, 1.0))
            commands = {
                "fixed": np.array([[1.5, 0, 4.5], [2, 0.3, 4.5]]),
                "aimed": np.array([[*oracle(source, g, 1.0, wind, beta), 4.5] for g in goals]),
            }
            for policy in videos:
                clips = []
                for g, cmd in zip(goals, commands[policy]):
                    demo = flight(source, cmd, 1.0, wind, beta, seconds=0.6, render=True, target=g)
                    assert demo["frames"].shape == (16, 64, 64, 3)
                    clips.append(demo["frames"])
                    if policy == "aimed":
                        assert (
                            np.linalg.norm(flight(source, cmd, 1.0, wind, beta)["landing"] - g)
                            < 1e-6
                        )
                videos[policy].append(np.concatenate(clips))
            for qi, (pos, target, mass, probe) in enumerate(queries):
                action = oracle(pos, target, mass, wind, beta)
                expert = flight(pos, [*action, 4.5], mass, wind, beta)
                assert np.linalg.norm(expert["landing"] - target) < 1e-6
                future, _ = fast_trajectory(pos, [*probe, 4.5], mass, wind, beta, steps=160)
                future = future[1:]
                if f < 8:
                    actual = flight(pos, [*probe, 4.5], mass, wind, beta, seconds=0.64)[
                        "positions"
                    ][1:]
                    assert np.max(np.abs(actual - future)) < 1e-5
                common = dict(
                    family=f,
                    sibling=sibling,
                    query_index=qi,
                    video_index=len(videos["fixed"]) - 1,
                    query=[*pos, *target, mass, 4.5, *source, *goals.flatten(), 1.0, 4.5, 0.6],
                    action=action,
                    probe=probe,
                    future=future,
                    wind=wind,
                    beta=beta,
                    source=source,
                )
                assert len(common["query"]) == 17
                for policy in videos:
                    rows[policy].append(
                        dict(
                            **common,
                            motion=np.concatenate([np.r_[source, c] for c in commands[policy]]),
                        )
                    )
        if f % 32 == 0:
            print(split, f, n, "seconds", time.time() - starttime, flush=True)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    for policy in videos:
        data = {k: np.array([r[k] for r in rows[policy]]) for k in rows[policy][0]}
        data["video"] = np.array(videos[policy], dtype=np.uint8)
        name = policy + "_" + split + ("" if part is None else "_" + str(part))
        np.savez_compressed(out / (name + ".npz"), **data)
    (out / (split + ("" if part is None else "_" + str(part)) + ".json")).write_text(
        json.dumps(
            dict(families=n, records=len(rows["fixed"]), seed=seed, seconds=time.time() - starttime)
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test", "pilot"])
    p.add_argument("--part", type=int)
    a = p.parse_args()
    n, seed = {
        "train": (2048, 98100000),
        "val": (128, 98200000),
        "test": (256, 98300000),
        "pilot": (16, 98000000),
    }[a.split]
    generate(a.split, n // 8 if a.part is not None else n, seed + (a.part or 0) * 10000, a.part)
