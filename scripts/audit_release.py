"""Audit the exact public source allowlist without printing matched secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
ANON_NAME = "Anonymous Authors"
ANON_EMAIL = ""
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".json", ".txt"}
REVIEWED_FIGURES = {
    "assets/robot_predictions.png",
    "assets/late_context_errors.png",
    "assets/late_context_predictions.png",

    "assets/overview.png",
    "assets/controlled_tasks.png",
    "assets/prediction_horizon.png",
}
# Construct patterns in fragments so the scanner's source is not a positive match.
RULES = {
    "private-home": r"/" + r"(?:Users|home|mnt/workspace/users)/[A-Za-z0-9_.-]+",
    "private-mount": r"/" + r"(?:data|mnt)/(?!v1\b)[A-Za-z0-9_.-]+/",
    "private-key": r"-----BEGIN " + r"(?:[A-Z]+ )?PRIVATE KEY-----",
    "github-token": r"\bgh" + r"[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_" + r"pat_[A-Za-z0-9_]{20,}",
    "provider-key": r"\bsk" + r"-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
    "aws-key": r"\bAK" + r"IA[A-Z0-9]{16}\b",
    "credential-url": r"https?://[^\s/]+:[^\s/]+@",
    "private-ip": r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b",
    "internal-link": r"https?://[^\s]*" + r"(?:feishu|larksuite|overleaf)" + r"[^\s]*",
}
EMAIL = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
URL = re.compile(r"https?://[^\s<>\"')]+")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL)


def inspect_text(name, text, denylist=()):
    issues = []
    for number, line in enumerate(text.splitlines(), 1):
        for category, pattern in RULES.items():
            if re.search(pattern, line):
                issues.append(dict(file=name, line=number, category=category))
        for email in EMAIL.findall(line):
            if email != ANON_EMAIL:
                issues.append(dict(file=name, line=number, category="email"))
        if URL.search(line):
            issues.append(dict(file=name, line=number, category="web-link"))
        if name.endswith(".md") and re.search(r"(?<!!)\[[^\]\n]+\]\([^\)\n]+\)", line):
            issues.append(dict(file=name, line=number, category="markdown-link"))
        if any(word.casefold() in line.casefold() for word in denylist):
            issues.append(dict(file=name, line=number, category="private-denylist"))
    if any(word.casefold() in name.casefold() for word in denylist):
        issues.append(dict(file=name, category="private-filename"))
    return issues


def inspect_data(name, data, denylist=()):
    if name not in REVIEWED_FIGURES:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return [dict(file=name, category="binary-file")]
        issues = inspect_text(name, content, denylist)
        if "\0" in content:
            issues.append(dict(file=name, category="binary-file"))
        return issues
    # Only individually reviewed raster figures are allowed. No metadata chunks,
    # embedded documents, trailing payloads, or arbitrary binary files.
    try:
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("signature")
        offset, chunks, payloads = 8, [], []
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("truncated chunk")
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            end = offset + size + 12
            if end > len(data) or kind not in {b"IHDR", b"IDAT", b"IEND"}:
                raise ValueError("unreviewed chunk")
            payload = data[offset + 8:end - 4]
            crc = struct.unpack(">I", data[end - 4:end])[0]
            if zlib.crc32(kind + payload) != crc:
                raise ValueError("CRC")
            chunks.append(kind)
            if kind == b"IHDR":
                if len(chunks) != 1 or size != 13:
                    raise ValueError("header")
                width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
                if not (0 < width <= 4096 and 0 < height <= 4096):
                    raise ValueError("dimensions")
                if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                    raise ValueError("unsupported encoding")
            elif kind == b"IDAT":
                payloads.append(payload)
            elif size != 0 or end != len(data):
                raise ValueError("trailing data")
            offset = end
        if len(chunks) < 3 or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or any(c != b"IDAT" for c in chunks[1:-1]):
            raise ValueError("chunk order")
        expected_size = height * (1 + width * 3)
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(b"".join(payloads), expected_size + 1)
        if len(pixels) != expected_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("pixel stream")
    except (ValueError, struct.error, zlib.error):
        return [dict(file=name, category="unreviewed-image-data")]
    return inspect_text(name, "", denylist)


def audit(denylist=(), require_git=False):
    errors = []
    manifest_path = ROOT / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return dict(passed=False, errors=[dict(category="missing-manifest")])
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Empty or invalid release manifest")
    for name, digest in expected.items():
        p = ROOT / name
        if Path(name).is_absolute() or ".." in Path(name).parts or p.is_symlink():
            errors.append(dict(file=name, category="unsafe-path"))
            continue
        if not p.is_file():
            errors.append(dict(file=name, category="missing-file"))
            continue
        if any(
            parent.is_symlink() for parent in p.parents if parent == ROOT or ROOT in parent.parents
        ):
            errors.append(dict(file=name, category="symlink-parent"))
            continue
        if p.suffix not in ALLOWED_SUFFIXES and name not in {"LICENSE", ".gitignore"} | REVIEWED_FIGURES:
            errors.append(dict(file=name, category="file-type"))
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            errors.append(dict(file=name, category="digest-mismatch"))
        errors.extend(inspect_data(name, data, denylist))
    errors.extend(inspect_text(MANIFEST, manifest_path.read_text(), denylist))
    declared = set(expected) | {MANIFEST}
    # All filesystem extras are reported; callers must audit a clean export.
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if rel.parts[0] == ".git":
            continue
        if p.is_symlink() or (p.is_file() and str(rel) not in declared):
            errors.append(dict(file=str(rel), category="unlisted-file-or-symlink"))
    commits = 0
    if (ROOT / ".git").exists():
        tracked = set(git("ls-files", "-z").decode().strip("\0").split("\0"))
        if tracked != declared:
            errors.append(dict(category="git-allowlist-mismatch"))
        if git("status", "--porcelain").strip():
            errors.append(dict(category="dirty-git-tree"))
        if git("remote").strip():
            # Destination identity is a separate user decision, never implicitly approved.
            errors.append(dict(category="remote-present-review-required"))
        refs = git("rev-list", "--all").decode().splitlines()
        commits = len(refs)
        if commits != 1:
            errors.append(dict(category="unexpected-history"))
        for commit in refs:
            fields = (
                git("show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", commit).decode().splitlines()
            )
            if fields[:4] != [ANON_NAME, ANON_EMAIL, ANON_NAME, ANON_EMAIL]:
                errors.append(dict(category="git-identity"))
            errors.extend(inspect_text("git-commit-message", "\n".join(fields[4:]), denylist))
            tree = set(git("ls-tree", "-r", "--name-only", commit).decode().splitlines())
            if tree != declared:
                errors.append(dict(category="historical-allowlist-mismatch"))
            for name in tree:
                data = git("show", f"{commit}:{name}")
                errors.extend(inspect_data(name, data, denylist))
        unreachable = git("fsck", "--full", "--no-reflogs", "--unreachable").strip()
        if unreachable:
            errors.append(dict(category="unreachable-git-objects"))
    elif require_git:
        errors.append(dict(category="missing-git-repository"))
    return dict(passed=not errors, files=len(declared), commits=commits, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--denylist", type=Path, help="Private newline-separated identifiers; keep outside release"
    )
    parser.add_argument("--require-git", action="store_true")
    args = parser.parse_args()
    denylist = (
        []
        if args.denylist is None
        else [s.strip() for s in args.denylist.read_text().splitlines() if s.strip()]
    )
    result = audit(denylist, args.require_git)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
