"""Float64 CPU population experiments, with fixed-budget projected Adam."""

import os

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"

import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
G = torch.eye(4)[:2]
STEPS = 4000


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(case):
    if case == "missing":
        return torch.stack([torch.eye(4)[0], torch.zeros(4)])
    return torch.eye(4)[{"s2": [0, 1], "s1": [0, 2], "s0": [2, 3]}[case]]


def configs():
    result = []
    for case in ["s2", "s1", "s0", "missing"]:
        for width in [1, 2, 3, 4]:
            for seed in [0, 1, 2]:
                result.append(
                    dict(
                        case=case,
                        width=width,
                        cap=1,
                        seed=seed,
                        block="B3" if case == "missing" else "B1",
                    )
                )
    for cap in [10, 100]:
        for width in [2, 4]:
            for seed in [0, 1, 2]:
                result.append(dict(case="s0", width=width, cap=cap, seed=seed, block="B2"))
    assert len(result) == 60
    return result


def run_id(c):
    return f"{c['case']}_m{c['width']}_C{c['cap']}_seed{c['seed']}"


def initialize(width, seed):
    gen = torch.Generator(device="cpu").manual_seed(997000 + width * 100 + seed)
    raw = torch.randn(4, width, generator=gen).requires_grad_()
    weight = (0.1 * torch.randn(width, width, generator=gen) / math.sqrt(width)).requires_grad_()
    # Preserve the original version's initial attention kernel exactly while
    # expressing W in the canonical positive-diagonal QR basis.
    with torch.no_grad():
        _, r = torch.linalg.qr(raw, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        weight = (signs[:, None] * weight * signs[None, :]).requires_grad_()
    return raw, weight


def basis(raw):
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    return (q * signs).T


def loss(raw, weight, f):
    b = basis(raw)
    residual = G - f @ b.T @ weight @ b
    return 0.5 * residual.square().sum()


def update(raw, weight, opt, f, cap, step):
    lr = 0.0001 + 0.5 * (0.01 - 0.0001) * (1 + math.cos(math.pi * (step - 1) / (STEPS - 1)))
    for group in opt.param_groups:
        group["lr"] = lr
    opt.zero_grad(set_to_none=True)
    objective = loss(raw, weight, f)
    objective.backward()
    if not all(torch.isfinite(p.grad).all() for p in [raw, weight]):
        raise FloatingPointError("Nonfinite gradient")
    grad_norms = [float(p.grad.norm()) for p in [raw, weight]]
    opt.step()
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        weight.copy_((u * s.clamp(max=cap)) @ vh)
    return lr, grad_norms


def metrics(raw, weight, f):
    with torch.no_grad():
        b = basis(raw)
        a = f @ b.T
        residual = G - a @ weight @ b
        singular = torch.linalg.svdvals(a)
        return dict(
            risk=float(0.5 * residual.square().sum()),
            W_spectral=float(torch.linalg.matrix_norm(weight, ord=2)),
            W_frobenius=float(weight.norm()),
            B_orthogonality=float((b @ b.T - torch.eye(b.shape[0])).abs().max()),
            A_sigma_max=float(singular.max()),
            A_sigma_min=float(singular.min()),
            raw_sigma_min=float(torch.linalg.svdvals(raw).min()),
        )


def checkpoint(path, c, raw, weight, opt, step):
    torch.save(
        dict(
            config=c,
            raw=raw.detach().clone(),
            W=weight.detach().clone(),
            optimizer=opt.state_dict(),
            step=step,
            torch_rng=torch.get_rng_state(),
            script_sha256=sha(__file__),
        ),
        path,
    )


def preflight(out):
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    # Independently evaluate the actual attention formula over all covariates
    # and an exact covariance-I sigma-point distribution of task coefficients.
    rng = np.random.default_rng(7722)
    grad_checks = []
    explicit_errors = []
    for width in [1, 2, 3, 4]:
        raw, weight = initialize(width, 83)
        f = source("s1")
        grad_checks.append(
            torch.autograd.gradcheck(
                lambda r, w: loss(r, w, f), (raw, weight), eps=1e-6, atol=2e-5, rtol=1e-4
            )
        )
        b = basis(raw).detach().numpy()
        w = weight.detach().numpy()
        fn = f.numpy()
        gn = G.numpy()
        phi = 2 * np.eye(4)
        beta_points = np.sqrt(2) * np.concatenate([np.eye(2), -np.eye(2)])
        explicit = 0.0
        for beta in beta_points:
            values = beta @ fn @ phi
            for q in range(4):
                predicted = (
                    sum(values[i] * (b @ phi[:, i]) @ w @ (b @ phi[:, q]) for i in range(4)) / 4
                )
                truth = beta @ gn @ phi[:, q]
                explicit += 0.5 * (truth - predicted) ** 2 / 16
        explicit_errors.append(abs(explicit - float(loss(raw, weight, f).detach())))
    # A healthy full-rank matrix must not produce an O(1) loss jump when its
    # first entry crosses zero. Raw LAPACK QR changes a column sign here.
    example_w = torch.tensor([[0.2, 0.7], [-0.4, 0.3]])
    continuity = []
    for delta in [-1e-9, 1e-9]:
        example_r = torch.tensor([[delta, 0.2], [1.0, 0.3], [0.5, 1.0], [0.1, 0.7]])
        continuity.append(float(loss(example_r, example_w, source("s0"))))
    checks["canonical_qr_loss_change_for_2e9_input_change"] = abs(continuity[1] - continuity[0])
    assert checks["canonical_qr_loss_change_for_2e9_input_change"] < 1e-7
    checks["gradient_checks_all_widths"] = all(grad_checks)
    checks["explicit_attention_max_error"] = max(explicit_errors)
    assert checks["explicit_attention_max_error"] < 1e-12
    recovery = []
    for cap in [1, 100]:
        c = dict(case="s0", width=2, seed=91, cap=cap)
        raw, weight = initialize(2, 91)
        opt = torch.optim.Adam([raw, weight], lr=0.01)
        for step in range(1, 8):
            update(raw, weight, opt, source("s0"), cap, step)
        file = out / f"resume_C{cap}_step7.pt"
        checkpoint(file, c, raw, weight, opt, 7)
        update(raw, weight, opt, source("s0"), cap, 8)
        loaded = torch.load(file, weights_only=False, map_location="cpu")
        rr = loaded["raw"].clone().requires_grad_()
        ww = loaded["W"].clone().requires_grad_()
        oo = torch.optim.Adam([rr, ww], lr=0.01)
        oo.load_state_dict(loaded["optimizer"])
        update(rr, ww, oo, source("s0"), cap, 8)
        equal = torch.equal(raw, rr) and torch.equal(weight, ww)
        recovery.append(equal)
        assert equal
    checks["exact_save_reload_next_update"] = all(recovery)
    raw, weight = initialize(2, 92)
    opt = torch.optim.Adam([raw, weight], lr=0.01)
    t0 = time.monotonic()
    for step in range(1, 201):
        update(raw, weight, opt, source("s2"), 1, step)
        metrics(raw, weight, source("s2"))
    duration = time.monotonic() - t0
    checks.update(
        benchmark_updates=200,
        total_preflight_optimizer_updates=218,
        benchmark_seconds=duration,
        estimated_core_seconds=duration * 1200,
        benchmark_final=metrics(raw, weight, source("s2")),
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        device=str(raw.device),
        dtype=str(raw.dtype),
        threads=torch.get_num_threads(),
        script_sha256=sha(__file__),
        pass_all=True,
    )
    dump(out / "preflight.json", checks)
    print("PREFLIGHT_PASS " + json.dumps(checks), flush=True)


def train_one(c):
    ident = run_id(c)
    out = ROOT / "runs" / ident
    if (out / "complete.json").exists():
        record = json.loads((out / "complete.json").read_text())
        assert record["updates"] == STEPS and record["checkpoint_sha256"] == sha(out / "final.pt")
        assert record["script_sha256"] == sha(__file__)
        print("EXISTING_COMPLETE " + ident, flush=True)
        return record
    out.mkdir(parents=True, exist_ok=True)
    if (out / "initial.pt").exists():
        raise RuntimeError(f"Incomplete existing run requires explicit recovery: {out}")
    f = source(c["case"])
    raw, weight = initialize(c["width"], c["seed"])
    opt = torch.optim.Adam([raw, weight], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    checkpoint(out / "initial.pt", c, raw, weight, opt, 0)
    dump(
        out / "config.json",
        dict(
            c,
            steps=STEPS,
            device="cpu",
            dtype="float64",
            optimizer="projected_Adam",
            lr_start=0.01,
            lr_end=0.0001,
            numpy_version=np.__version__,
            torch_version=torch.__version__,
        ),
    )
    rows = []
    t0 = time.monotonic()
    start = metrics(raw, weight, f)
    rows.append(dict(step=0, lr=0.01, grad_raw=0.0, grad_W=0.0, **start))
    first_grads = None
    step = 0
    try:
        for step in range(1, STEPS + 1):
            lr, grads = update(raw, weight, opt, f, c["cap"], step)
            if first_grads is None:
                first_grads = grads
            met = metrics(raw, weight, f)
            assert all(math.isfinite(v) for v in met.values())
            assert met["W_spectral"] <= c["cap"] * (1 + 1e-10)
            assert met["B_orthogonality"] < 1e-10
            if c["case"] == "missing":
                assert met["risk"] >= 0.5 - 1e-12
            rows.append(dict(step=step, lr=lr, grad_raw=grads[0], grad_W=grads[1], **met))
            if step in [20, 1000, 2000, 3000, 4000]:
                checkpoint(out / f"step{step:04d}.pt", c, raw, weight, opt, step)
            if step == 20:
                dump(
                    out / "launch_verified.json",
                    dict(
                        actual_updates=20,
                        first_gradients=first_grads,
                        metrics=met,
                        script_sha256=sha(__file__),
                        optimization_only=True,
                    ),
                )
            if time.monotonic() - FORMAL_START > 7200:
                raise TimeoutError("Core two-hour CPU wall-clock budget reached")
    except Exception:
        checkpoint(out / "interrupted.pt", c, raw, weight, opt, step)
        dump(out / "failure.json", dict(step=step, traceback=traceback.format_exc()))
        raise
    finally:
        if rows:
            names = list(rows[0])
            np.savez_compressed(
                out / "history.npz", **{key: np.array([r[key] for r in rows]) for key in names}
            )
    final = metrics(raw, weight, f)
    checkpoint(out / "final.pt", c, raw, weight, opt, STEPS)
    b = basis(raw).detach().numpy()
    np.savez(
        out / "final_arrays.npz",
        B=b,
        W=weight.detach().numpy(),
        raw=raw.detach().numpy(),
        F=f.numpy(),
        G=G.numpy(),
    )
    record = dict(
        c,
        run_id=ident,
        updates=STEPS,
        initial_risk=start["risk"],
        final=final,
        best_logged_risk=min(r["risk"] for r in rows),
        first_gradients=first_grads,
        elapsed_seconds=time.monotonic() - t0,
        checkpoint_sha256=sha(out / "final.pt"),
        initial_checkpoint_sha256=sha(out / "initial.pt"),
        script_sha256=sha(__file__),
    )
    dump(out / "complete.json", record)
    print("COMPLETE " + json.dumps(record), flush=True)
    return record


def formal(stage):
    evidence = ROOT / "evidence"
    assert (evidence / "preflight" / "preflight.json").exists(), "Run preflight first"
    check = json.loads((evidence / "preflight" / "preflight.json").read_text())
    assert check["pass_all"] and check["script_sha256"] == sha(__file__)
    saved_code = evidence / "formal_code" / "run.py"
    if saved_code.exists():
        assert sha(saved_code) == sha(__file__), "Do not change formal training source"
    else:
        saved_code.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, saved_code)
        shutil.copyfile(ROOT / "PROTOCOL.md", evidence / "original_plan.md")
        dump(evidence / "matrix.json", configs())
    all_c = configs()
    sanity_ids = ["s2_m2_C1_seed0", "s0_m4_C1_seed0", "missing_m2_C1_seed0"]
    ordered = sorted(
        all_c,
        key=lambda c: (
            run_id(c) not in sanity_ids,
            sanity_ids.index(run_id(c)) if run_id(c) in sanity_ids else all_c.index(c),
        ),
    )
    if stage == "sanity":
        ordered = [c for c in ordered if run_id(c) in sanity_ids]
    records = [train_one(c) for c in ordered]
    dump(
        evidence / f"{stage}_complete.json",
        dict(
            count=len(records),
            updates=sum(r["updates"] for r in records),
            records=records,
            script_sha256=sha(__file__),
        ),
    )
    print(
        f"{stage.upper()}_COMPLETE runs={len(records)} actual_updates={sum(r['updates'] for r in records)}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, default=ROOT / "evidence/preflight")
    parser.add_argument("--stage", choices=["sanity", "all"])
    args = parser.parse_args()
    FORMAL_START = time.monotonic()
    if args.preflight:
        preflight(args.preflight_dir)
    elif args.stage:
        formal(args.stage)
    else:
        parser.error("Choose --preflight or --stage")
