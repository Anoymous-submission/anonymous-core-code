"""The raw export hash must work without Python 3.11's file_digest API."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("size", [0, 4 * 1024 * 1024 + 31])
def test_export_hash_without_file_digest(tmp_path, monkeypatch, size):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    source = Path(__file__).resolve().parents[1] / "protocols/language/export_rgb.py"
    spec = importlib.util.spec_from_file_location("language_export_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = (b"\x00export\xff\x01" * (size // 9 + 1))[:size]
    path = tmp_path / "raw.bin"
    path.write_bytes(data)
    assert module.sha256_file(path) == hashlib.sha256(data).hexdigest()
