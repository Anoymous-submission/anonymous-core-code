"""Finalize a completed RGB export after all worker futures returned."""

import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent / "rgb_data"
rows = [json.loads(x) for x in (root / "records.jsonl").read_text().splitlines()]
assert len(rows) == 15335 and sum(r["split"] == "train" for r in rows) == 14634
assert (root / "stats.json").is_file()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


result = dict(
    records=len(rows),
    train=14634,
    test=701,
    rgb_sha256=digest(root / "robot_rgb.npy"),
    state_sha256=digest(root / "robot_state.npy"),
)
(root / "complete.json").write_text(json.dumps(result, indent=2))
print("EXPORT_COMPLETE", result, flush=True)
