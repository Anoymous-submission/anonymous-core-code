"""Run a protocol in a fresh interpreter with its matching source package."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "protocol", choices=sorted(p.name for p in (root / "protocols").iterdir() if p.is_dir())
    )
    parser.add_argument("entrypoint", help="Python file relative to the chosen protocol")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    folder = root / "protocols" / args.protocol
    entrypoint = (folder / args.entrypoint).resolve()
    if (
        not entrypoint.is_relative_to(folder)
        or not entrypoint.is_file()
        or entrypoint.suffix != ".py"
    ):
        parser.error("Entrypoint must be a Python file inside the chosen protocol")
    search = [folder / "src", root / "src"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in search] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    raise SystemExit(subprocess.call([sys.executable, str(entrypoint), *args.args], env=env))


if __name__ == "__main__":
    main()
