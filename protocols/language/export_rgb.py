"""Export raw RGB/state examples for online encoding; no latent pre-encoding."""

import json, math, argparse, hashlib

from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

import numpy as np, h5py, torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_group(item):
    path, entries, output = item
    out_rgb = np.lib.format.open_memmap(output / "robot_rgb.npy", mode="r+")
    out_state = np.lib.format.open_memmap(output / "robot_state.npy", mode="r+")
    with h5py.File(path, "r") as f:
        for i, st in entries:
            ix = np.arange(st, st + 33, 4)
            out_rgb[i] = np.clip(f["cam_data/robot_camera"][ix], 0, 255).astype(np.uint8)
            pos = f["end_position"][st : st + 33].astype(np.float32)
            grip = f["gripper_state"][st : st + 33].astype(np.float32)[:, None]
            out_state[i] = np.concatenate([pos, grip], -1)
    return len(entries)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--stats", type=Path, required=True)

    ap.add_argument("--manifest", type=Path, required=True)

    ap.add_argument("--captions", type=Path, required=True)

    ap.add_argument("--out", type=Path, required=True)

    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--workers", type=int, default=1)

    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=a.resume)

    m = json.loads(a.manifest.read_text())

    episodes = {e["path"]: e for e in m["episodes"]}

    caption = {r["uid"]: r for r in map(json.loads, a.captions.read_text().splitlines())}

    rows = []

    for s in m["segments"]:
        r = caption[s["uid"]]
        n = episodes[s["path"]]["camera_shapes"]["human"][0]
        first = math.ceil(max(s["frame_indices"]) / 4) * 4
        for offset in (0, 16, 32, 48):
            start = first + offset
            if start + 32 >= n:
                continue
            assert max(s["frame_indices"]) <= start
            rows.append(
                dict(
                    uid=f"{s['uid']}_q{start}",
                    caption_uid=s["uid"],
                    path=s["path"],
                    split=r["split"],
                    caption=r["caption"],
                    caption_last_frame=max(s["frame_indices"]),
                    query_start=start,
                )
            )

    assert len(rows) == 15335 and sum(r["split"] == "train" for r in rows) == 14634

    mode = "r+" if a.resume else "w+"

    rgb = np.lib.format.open_memmap(
        a.out / "robot_rgb.npy", mode=mode, dtype=np.uint8, shape=(len(rows), 9, 240, 426, 3)
    )

    state = np.lib.format.open_memmap(
        a.out / "robot_state.npy", mode=mode, dtype=np.float32, shape=(len(rows), 33, 7)
    )

    groups = defaultdict(list)

    for i, r in enumerate(rows):
        groups[r["path"]].append((i, r["query_start"]))

    done = 0

    if a.workers == 1:
        for item in [(path, entries, a.out) for path, entries in groups.items()]:
            done += export_group(item)
            if done % 500 < 4:
                print("EXPORTED", done, len(rows), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [
                pool.submit(export_group, (path, entries, a.out))
                for path, entries in groups.items()
            ]
            for future in as_completed(futures):
                done += future.result()
                if done % 500 < 8:
                    print("EXPORTED", done, len(rows), flush=True)

    assert done == len(rows)

    rgb.flush()

    state.flush()

    stats = json.loads(a.stats.read_text())

    (a.out / "stats.json").write_text(json.dumps(stats))

    (a.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    (a.out / "complete.json").write_text(
        json.dumps(
            dict(
                records=len(rows),
                train=14634,
                test=701,
                rgb_sha256=sha256_file(a.out / "robot_rgb.npy"),
                state_sha256=sha256_file(a.out / "robot_state.npy"),
            )
        )
    )

    print("EXPORT_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
