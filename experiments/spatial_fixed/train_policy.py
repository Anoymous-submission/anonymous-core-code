"""Fixed-budget training; same target-only base plus optional demonstration.

No simulator, source response coordinates, or physical coefficients are imported
or read here. A separately trained None control learns an unrestricted target-only
residual from actual-world action labels, with the matched exposure budget.
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from policy_moments import Policy
from data_contract import sha, verify_files

ADIMS = dict(gate=6, bank=3, ramp=2)


def state_sha(state):
    return hashlib.sha256(
        b"".join(t.detach().cpu().numpy().tobytes() for t in state.values())
    ).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


class Dataset:
    def __init__(self, paths, task, frozen_files=None):
        self.videos = []
        self.motions = []
        self.meta = []
        self.paths = paths
        queries = []
        actions = []
        nominals = []
        families = []
        sources = []
        shards = []
        offset = 0
        for j, path in enumerate(paths):
            meta = json.loads((path / "COMPLETE.json").read_text())
            assert meta["task"] == task and not meta["failures"]
            if frozen_files is not None:
                assert all(
                    frozen_files[str((path / name).resolve())] == digest
                    for name, digest in meta["files"].items()
                )
            verify_files(path, meta["files"])
            r = np.load(path / "records.npz")
            queries.append(r["query"])
            actions.append(r["action"])
            nominals.append(r["nominal_action"])
            families.append(r["family"] + offset)
            sources.append(r["source_id"])
            shards.append(np.full(len(r["query"]), j))
            self.videos.append(np.load(path / "source_rgb.npy", mmap_mode="r"))
            self.motions.append(np.load(path / "source_motion.npy"))
            self.meta.append(meta)
            offset += meta["families"]
        self.query = np.concatenate(queries).astype(np.float32)
        self.action = np.concatenate(actions).astype(np.float32) / 8
        self.nominal = np.concatenate(nominals).astype(np.float32) / 8
        self.family = np.concatenate(families)
        self.source = np.concatenate(sources)
        self.shard = np.concatenate(shards)

    def batch(self, ids, mode, device):
        if mode in ("video", "full"):
            rgb = np.stack([self.videos[j][s] for j, s in zip(self.shard[ids], self.source[ids])])
            video = torch.as_tensor(rgb, device=device).permute(0, 1, 4, 2, 3).float() / 255
        else:
            video = torch.zeros((len(ids), 16, 3, 96, 96), device=device)
        if mode in ("motion", "full"):
            movement = np.stack(
                [self.motions[j][s] for j, s in zip(self.shard[ids], self.source[ids])]
            )
            motion = torch.as_tensor(movement / 8, dtype=torch.float32, device=device)
        else:
            motion = torch.zeros((len(ids), 12), device=device)
        query = torch.as_tensor(self.query[ids], device=device)
        labels = torch.as_tensor(self.action[ids], device=device)
        nominal = torch.as_tensor(self.nominal[ids], device=device)
        return query, motion, video, labels, nominal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(ADIMS), required=True)
    parser.add_argument("--mode", choices=["none", "motion", "video", "full"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--development-steps", type=int, default=0)
    parser.add_argument(
        "--resume",
        help="Full checkpoint to continue in a new output directory; original run remains immutable",
    )
    parser.add_argument(
        "--freeze", help="Required immutable protocol/data manifest for formal training"
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    development = args.development_steps > 0
    frozen_files = None
    if not development:
        assert args.freeze, "Formal training requires frozen data/code provenance"
        freeze = json.loads(Path(args.freeze).read_text())
        assert freeze["ready"]
        assert args.data == freeze["data"][args.task]
        for path, digest in freeze["data_manifests"].items():
            assert sha(path) == digest
        for name, digest in freeze["code"].items():
            assert sha(Path(__file__).with_name(name)) == digest
        frozen_files = freeze["raw_data_files"]
    data = Dataset([Path(p) for p in args.data], args.task, frozen_files)
    n = len(data.query)
    adim = ADIMS[args.task]
    if not development:
        assert len(data.paths) == 32 and n == 16384 and len(np.unique(data.family)) == 2048
        assert all(
            m["stage"] == "frozen-data-generation" and m.get("split", "training") == "training"
            for m in data.meta
        )
        base_seed = dict(gate=2309241000, bank=2309242000, ramp=2309243000)[args.task]
        assert [m["seed"] for m in data.meta] == list(range(base_seed, base_seed + 32))
    torch.set_num_threads(4)
    torch.manual_seed(2309246000 + args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Policy().cuda().train()
    model_mode = "target_only" if args.mode == "none" else args.mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Prespecified schedule shared by every mode and seed; no validation tuning.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if step < 3277 else 0.2
    )
    rng = np.random.default_rng(2309247000 + args.seed)
    coverage = np.zeros(n, np.uint16)
    retained = np.zeros(n, np.uint16)
    total = args.development_steps if development else 4096
    initial_sha = state_sha(model.state_dict())
    started = time.time()
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    step = 0
    epoch = 0
    cursor = 0
    order = np.array([], dtype=int)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous = saved["args"]
        for key in ("task", "mode", "seed", "data"):
            assert previous[key] == getattr(args, key), (key, "resume mismatch")
        assert bool(previous.get("development_steps", 0)) == development
        if not development:
            assert saved["step"] < 4096, "Formal completed checkpoints cannot be extended"
        previous_plan = json.loads((Path(args.resume).parent / "PLAN.json").read_text())
        assert previous_plan["initial_sha"] == initial_sha
        assert previous_plan["data_manifests"] == data.meta, "Training input manifest changed"
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step = int(saved["step"])
        epoch = int(saved["epoch"])
        cursor = int(saved["cursor"])
        order = saved["order"]
        coverage = saved["coverage"].copy()
        retained = saved["retained"].copy()
        rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        assert step < total and len(coverage) == n and len(order) == n

    def checkpoint(name):
        tmp = out / (name + ".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                step=step,
                epoch=epoch,
                cursor=cursor,
                order=order,
                coverage=coverage,
                retained=retained,
                sampler_rng=rng.bit_generator.state,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                args=vars(args),
            ),
            tmp,
        )
        tmp.replace(out / name)

    checkpoint("resumed_state.pt" if args.resume else "initial.pt")
    dump(
        out / "PLAN.json",
        dict(
            **vars(args),
            stage="development" if development else "formal",
            initial_sha=initial_sha,
            parameters=sum(p.numel() for p in model.parameters()),
            records=n,
            starting_update=step,
            total_updates=total,
            nominal_supervision=True,
            none_control_learns_actual_residual=True,
            source_inputs=["raw_rgb", "imposed_motion"],
            data_manifests=data.meta,
        ),
    )
    while step < total and not stop:
        if cursor >= len(order):
            if len(order):
                epoch += 1
            order = rng.permutation(n)
            cursor = 0
        ids = order[cursor : cursor + 64]
        cursor += len(ids)
        q, m, v, label, nominal = data.batch(ids, args.mode, "cuda")
        present = torch.as_tensor((data.family[ids] + epoch) % 2 == 0, device="cuda")
        b, r = model(q, m, v, present, model_mode)
        base_loss = F.mse_loss(b[:, :adim], nominal[:, :adim])
        actual_loss = (
            F.mse_loss((b.detach() + r)[present, :adim], label[present, :adim])
            if present.any()
            else base_loss * 0
        )
        loss = base_loss + actual_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 19:
            groups = ["base", "residual"] + (["motion"] if args.mode in ("motion", "full") else [])
            if args.mode in ("video", "full"):
                groups += ["cnn", "temporal", "ordered_readout", "change_readout"]
            gradients = {
                g: sum(
                    float(p.grad.norm())
                    for name, p in model.named_parameters()
                    if name.startswith(g + ".") and p.grad is not None
                )
                for g in groups
            }
            assert all(np.isfinite(x) and x > 0 for x in gradients.values()), gradients
        torch.nn.utils.clip_grad_norm_(model.base.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(
            [p for name, p in model.named_parameters() if not name.startswith("base.")], 1.0
        )
        optimizer.step()
        scheduler.step()
        coverage[ids] += 1
        retained[ids] += present.cpu().numpy().astype(np.uint16)
        step += 1
        if step == 20:
            with torch.inference_mode():
                model.eval()
                off = torch.zeros(len(q), device="cuda")
                a, ar = model(q, m, v, off, model_mode)
                po, pr = model(q, torch.randn_like(m), torch.randn_like(v), off, model_mode)
                assert torch.equal(a, po) and torch.equal(ar, pr)
                model.train()
            proof = dict(
                actual_updates=20,
                formal_updates=0 if development else 20,
                diagnostic_updates=20 if development else 0,
                mode=args.mode,
                seed=args.seed,
                optimizer_steps=sorted(
                    set(int(s["step"]) for s in optimizer.state.values() if "step" in s)
                ),
                covered_records=int(np.count_nonzero(coverage)),
                sample_visits=int(coverage.sum()),
                initial_sha=initial_sha,
                base_sha=state_sha(model.base.state_dict()),
                gradients=gradients,
                no_demo_poison_invariance=True,
            )
            dump(out / "launch_verified.json", proof)
            checkpoint("latest.pt")
        if step % 64 == 0 or step == total:
            progress = dict(
                pid=os.getpid(),
                step=step,
                total=total,
                epoch=epoch,
                cursor=cursor,
                loss=float(loss),
                base_loss=float(base_loss),
                actual_loss=float(actual_loss),
                elapsed=time.time() - started,
                visits=int(coverage.sum()),
                covered=int(np.count_nonzero(coverage)),
            )
            dump(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if cursor == n:
            checkpoint("latest.pt")
    checkpoint("final.pt" if step == total else "paused.pt")
    if not development and step == total:
        assert np.all(coverage == 16) and np.all(retained == 8)
    dump(
        out / "COMPLETE.json" if step == total else out / "PAUSED.json",
        dict(
            step=step,
            formal_updates=0 if development else step,
            development_updates=step if development else 0,
            coverage_min=int(coverage.min()),
            coverage_max=int(coverage.max()),
            retained_min=int(retained.min()),
            retained_max=int(retained.max()),
            initial_sha=initial_sha,
            final_sha=state_sha(model.state_dict()),
            base_sha=state_sha(model.base.state_dict()),
        ),
    )


if __name__ == "__main__":
    main()
