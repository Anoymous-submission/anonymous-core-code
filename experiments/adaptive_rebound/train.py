import os, json, time, hashlib, random, argparse
from pathlib import Path
import numpy as np
import torch
from model import Model

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)


def sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Data:
    def __init__(self, split):
        self.z = dict(np.load(ROOT / "data" / f"{split}.npz"))

    def __len__(self):
        return len(self.z["query"])

    def batch(self, ids, wrong=False):
        z = self.z
        vi = z["video_index"][ids].copy()
        if wrong:
            vi = vi ^ 1
        video = torch.from_numpy(z["video"][vi]).cuda().float().permute(0, 1, 4, 2, 3) / 255
        motion = z["motion"][np.asarray(ids) ^ 4 if wrong else ids]
        return (
            video,
            torch.tensor(motion, device="cuda", dtype=torch.float32),
            *[
                torch.tensor(z[k][ids], device="cuda", dtype=torch.float32)
                for k in ["query", "probe", "action", "future"]
            ],
        )


def evaluate(model, data, mode, path, wrong=False):
    model.eval()
    pred = []
    futures = []
    with torch.no_grad():
        for off in range(0, len(data), 64):
            v, m, q, p, a, f = data.batch(np.arange(off, min(off + 64, len(data))), wrong)
            ah, fh = model(v, m, q, p, mode)
            pred.append(ah.cpu().numpy())
            futures.append(fh.cpu().numpy())
    actions = np.concatenate(pred)
    future = np.concatenate(futures)
    result = dict(
        action_mse=float(np.mean((actions - data.z["action"]) ** 2)),
        future_ade_m=float(np.linalg.norm(future - data.z["future"], axis=-1).mean()),
    )
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "predictions.npz", action=actions, future=future)
    (path / "metrics.json").write_text(json.dumps(result))
    return result


def main(mode, seed, smoke=False):
    if smoke:
        prior = ROOT / "preflight" / f"{mode}_seed{seed}"
        if (prior / "smoke.json").exists():
            started = json.loads((prior / "started.json").read_text())
            for name, expected in started["code"].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
            assert json.loads((prior / "smoke.json").read_text())["updates"] == 8192
            print("SMOKE_ALREADY_COMPLETE", prior, flush=True)
            return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    data = Data("pilot" if smoke else "train")
    val = Data("pilot" if smoke else "val")
    model = Model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    out = ROOT / ("preflight" if smoke else "runs") / f"{mode}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "started.json").exists()
    (out / "started.json").write_text(
        json.dumps(
            dict(
                initial_sha=sha(model),
                parameters=sum(p.numel() for p in model.parameters()),
                code={
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ["model.py", "train.py", "physics.py", "generate.py"]
                },
            )
        )
    )
    rng = np.random.default_rng(8100 + seed)
    coverage = np.zeros(len(data), np.int32)
    step = 0
    start = time.time()
    epochs = 8192 if smoke else 16
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(32 if smoke else len(data))
        acc = []
        for off in range(0, len(order), 64):
            ids = order[off : off + 64]
            v, m, q, p, a, f = data.batch(ids)
            opt.zero_grad(set_to_none=True)
            ah, fh = model(v, m, q, p, mode)
            la = (ah - a).square().mean()
            lf = (fh - f).square().mean()
            loss = la + lf
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            coverage[ids] += 1
            step += 1
            acc.append([la.item(), lf.item()])
            if step == 20 and not smoke:
                groups = {}
                for key, param in model.named_parameters():
                    group = key.split(".")[0]
                    groups[group] = groups.get(group, 0.0) + (
                        0.0 if param.grad is None else float(param.grad.detach().abs().sum())
                    )
                assert groups["fuse"] > 0 and groups["action"] > 0 and groups["future"] > 0
                if mode in ["video", "full"]:
                    assert groups["cnn"] > 0 and groups["video"] > 0
                if mode in ["motion", "full"]:
                    assert groups["motion"] > 0
                optim_steps = sorted(set(int(x["step"]) for x in opt.state.values()))
                assert optim_steps == [20]
                (out / "formal_verified.json").write_text(
                    json.dumps(
                        dict(
                            actual_updates=20,
                            optimizer_steps=optim_steps,
                            gradient_l1=groups,
                            current_sha=sha(model),
                            coverage_sum=int(coverage.sum()),
                        )
                    )
                )
            if step % 32 == 0:
                (out / "progress.json").write_text(
                    json.dumps(
                        dict(
                            step=step,
                            epoch=epoch + 1,
                            loss=loss.item(),
                            seconds=time.time() - start,
                        )
                    )
                )
        with (out / "history.jsonl").open("a") as h:
            h.write(
                json.dumps(
                    dict(
                        epoch=epoch + 1,
                        step=step,
                        action_loss=float(np.mean(acc, 0)[0]),
                        future_loss=float(np.mean(acc, 0)[1]),
                    )
                )
                + "\n"
            )
        if not smoke and (epoch + 1) % 4 == 0:
            evaluate(model, val, mode, out / f"val_epoch{epoch+1:02d}")
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    step=step,
                    epoch=epoch + 1,
                    coverage=coverage,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    sampler_rng=rng.bit_generator.state,
                ),
                out / "checkpoint.pt",
            )
    if smoke:
        with torch.no_grad():
            v, m, q, p, a, f = data.batch(np.arange(32))
            ah, fh = model(v, m, q, p, mode)
            result = dict(
                action_rmse=float((ah - a).square().mean().sqrt()),
                future_ade=float((fh - f).norm(dim=-1).mean()),
                updates=step,
            )
        (out / "smoke.json").write_text(json.dumps(result))
        print(result)
        return
    assert step == 4096 and np.all(coverage == 16)
    results = {"val": evaluate(model, val, mode, out / "val")}
    test = Data("test")
    results["test"] = evaluate(model, test, mode, out / "test")
    if mode in ["video", "full"]:
        results["wrong"] = evaluate(model, test, mode, out / "wrong", True)
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                updates=step,
                coverage_min=int(coverage.min()),
                coverage_max=int(coverage.max()),
                results=results,
                seconds=time.time() - start,
                final_sha=sha(model),
            )
        )
    )
    print("COMPLETE", mode, seed, results, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "motion", "video", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    main(a.mode, a.seed, a.smoke)
