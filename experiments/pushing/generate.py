import json, time, argparse
from pathlib import Path
import numpy as np
from physics import simulate, expert
from language import caption

ROOT = Path(__file__).resolve().parent
SPECS = {
    "pilot": (8, 110100000),
    "train": (1024, 111000000),
    "val": (128, 112000000),
    "test": (256, 113000000),
}


def generate(split, part=None):
    n, seed = SPECS[split]
    if part is not None:
        n //= 8
        seed += 10000 * part
    rng = np.random.default_rng(seed)
    videos = []
    motions = []
    texts = []
    tokens = []
    rows = []
    starttime = time.time()
    for family in range(n):
        speed = rng.uniform(0.6, 1.2)
        light = ["neutral", "side", "warm"][(family + (part or 0)) % 3]
        queries = [
            (
                rng.uniform(-0.03, 0.04),
                rng.uniform(0.18, 0.34),
                float(rng.choice([0.15, 0.35])),
                float(rng.choice([-0.03, 0, 0.03])),
                rng.uniform(0.5, 1.3),
            )
            for _ in range(4)
        ]
        # Both materials are independent continuous draws, never selected by observed difficulty.
        for sibling in range(2):
            mu = rng.uniform(0.2, 0.6)
            demo = simulate([speed, 0], mu, render=True, light=light)
            assert demo["contact"] and demo["video"].shape == (32, 64, 64, 3)
            text, tok = caption(demo["motion"], demo["states"])
            videos.append(demo["video"])
            motions.append(demo["motion"])
            texts.append(text)
            tokens.append(tok)
            for qi, (x0, distance, mass, slope, probe) in enumerate(queries):
                goal = x0 + distance
                action, run = expert(goal, mu, mass, slope, (x0, 0))
                error = float(np.linalg.norm(run["final"][:2] - [goal, 0]))
                assert run["contact"] and error < 0.01 and np.linalg.norm(run["velocity"]) < 0.03, (
                    split,
                    family,
                    mu,
                    qi,
                    error,
                    run["velocity"],
                )
                assert abs(run["final"][2] - 0.025) < 0.003
                future = simulate([probe, 0], mu, mass, slope, (x0, 0))["states"][1::2, :2]
                rows.append(
                    dict(
                        family=family,
                        sibling=sibling,
                        query_index=qi,
                        video_index=len(videos) - 1,
                        query=[x0, 0, goal, 0, mass, slope, 1.6, 0.2, 0],
                        action=[action],
                        probe=[probe],
                        future=future,
                        mu=mu,
                        source_speed=speed,
                        source_light=["neutral", "side", "warm"].index(light),
                        oracle_error=error,
                    )
                )
        if family % 8 == 0:
            print(split, part, family, n, time.time() - starttime, flush=True)
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data.update(
        video=np.array(videos),
        motion=np.array(motions),
        tokens=np.array(tokens),
        captions=np.array(texts),
    )
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    stem = split + ("" if part is None else "_" + str(part))
    np.savez_compressed(out / (stem + ".npz"), **data)
    (out / (stem + ".json")).write_text(
        json.dumps(
            dict(
                families=n,
                records=len(rows),
                seed=seed,
                seconds=time.time() - starttime,
                max_oracle_error=float(data["oracle_error"].max()),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=list(SPECS), required=True)
    p.add_argument("--part", type=int)
    a = p.parse_args()
    if a.split in ["val", "test"] and a.part is None:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(generate, a.split, i) for i in range(8)]
            for f in futures:
                f.result()
        chunks = [dict(np.load(ROOT / "data" / f"{a.split}_{i}.npz")) for i in range(8)]
        offset = 0
        for i, z in enumerate(chunks):
            z["video_index"] += offset
            z["family"] += (SPECS[a.split][0] // 8) * i
            offset += len(z["video"])
        np.savez_compressed(
            ROOT / "data" / f"{a.split}.npz",
            **{k: np.concatenate([z[k] for z in chunks]) for k in chunks[0]},
        )
    else:
        generate(a.split, a.part)
