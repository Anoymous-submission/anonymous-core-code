"""Check modality isolation and candidate-action separation in released policies."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename="model.py"):
    folder = ROOT / "experiments" / name
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("test_" + name, folder / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(folder))


@pytest.mark.parametrize(
    "task,frames,mdim,qdim", [("throwing", 32, 12, 17), ("rebound", 16, 6, 10)]
)
def test_fixed_query_and_candidate_action_isolation(task, frames, mdim, qdim):
    torch.manual_seed(1)
    module = load(task)
    model = module.Model().eval()
    v = torch.randn(2, frames, 3, 64, 64)
    m = torch.randn(2, mdim)
    q = torch.randn(2, qdim)
    p = torch.randn(2, 2)
    with torch.no_grad():
        a, f = model(v, m, q, p, "none")
        a2, f2 = model(v + 100, m - 100, q, p, "none")
        assert torch.equal(a, a2) and torch.equal(f, f2)
        for mode in ["motion", "video", "full"]:
            a, f = model(v, m, q, p, mode)
            a2, f2 = model(v, m, q, p + 100, mode)
            assert torch.equal(a, a2), "Candidate action leaked into policy"
            assert not torch.equal(f, f2)


def test_pushing_none_hides_all_demonstration_modalities():
    torch.manual_seed(2)
    module = load("pushing")
    model = module.Model().eval()
    v = torch.randn(2, 32, 3, 64, 64)
    m = torch.randn(2, 81, 3)
    tokens = torch.zeros(2, module.MAX_TOKENS, dtype=torch.long)
    q = torch.randn(2, 9)
    p = torch.randn(2, 1)
    with torch.no_grad():
        a, f = model(v, m, tokens, q, p, "none")
        a2, f2 = model(v + 100, m - 100, tokens + 1, q, p, "none")
        assert torch.equal(a, a2) and torch.equal(f, f2)
        assert ((a >= 0.35) & (a <= 2.2)).all()


def test_spatial_absent_context_preserves_only_shared_specification():
    torch.manual_seed(3)
    model = load("spatial", "policy_moments.py").Policy().eval()
    q = torch.randn(2, 32)
    m = torch.randn(2, 12)
    v = torch.rand(2, 16, 3, 96, 96)
    absent = torch.zeros(2)
    with torch.no_grad():
        base, residual = model(q, m, v, absent, "full")
        b2, r2 = model(q, m + 100, 1 - v, absent, "full")
        assert torch.equal(base, b2) and torch.equal(residual, r2)
        q2 = q.clone()
        q2[:, 16:] += 10
        b3, r3 = model(q2, m, v, absent, "full")
        assert torch.equal(base, b3), "Source specification leaked into target-only base"
        assert not torch.equal(residual, r3), "Shared source specification unexpectedly masked"
