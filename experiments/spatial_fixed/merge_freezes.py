"""Combine three task freezes without changing any source or data digests."""

import argparse
import json
from pathlib import Path
from data_contract import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(ready=True, data={}, data_manifests={}, raw_data_files={}, code={})
    for path in args.inputs:
        item = json.loads(path.read_text())
        assert item["ready"] and len(item["data"]) == 1
        assert not result["data"].keys() & item["data"].keys()
        if result["code"]:
            assert result["code"] == item["code"]
        result["code"] = item["code"]
        for field in ("data", "data_manifests", "raw_data_files"):
            result[field].update(item[field])
    assert set(result["data"]) == {"gate", "bank", "ramp"}
    for name, digest in result["code"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    for field in ("data_manifests", "raw_data_files"):
        for path, digest in result[field].items():
            assert sha(path) == digest
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
