"""Freeze complete, locally regenerated training shards for the core trainer."""

import argparse
import json
from pathlib import Path
from data_contract import sha, verify_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["gate", "bank", "ramp"], required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    assert len(a.data) == 32, "Formal protocol requires 32 ordered training shards"
    paths = [x.resolve() for x in a.data]
    manifests = {}
    raw = {}
    seed0 = dict(gate=2309241000, bank=2309242000, ramp=2309243000)[a.task]
    for i, path in enumerate(paths):
        m = json.loads((path / "COMPLETE.json").read_text())
        assert m["task"] == a.task and m["seed"] == seed0 + i
        assert m["stage"] == "frozen-data-generation" and m["split"] == "training"
        assert m["families"] == 64 and m["records"] == 512 and not m["failures"]
        verify_files(path, m["files"])
        manifests[str(path / "COMPLETE.json")] = sha(path / "COMPLETE.json")
        raw.update({str((path / name).resolve()): digest for name, digest in m["files"].items()})
    result = dict(
        ready=True,
        data={a.task: list(map(str, paths))},
        data_manifests=manifests,
        raw_data_files=raw,
        code={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
