"""Fresh 0.6B pretraining with caption-only demo and matched no-demo arm."""

import argparse, json, math, random, re, time, hashlib
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel
from fasterwam.models.hr_vae import FrozenWan22VideoEncoder

ROOT = Path(__file__).resolve().parent
RGB = ROOT / "rgb_data/robot_rgb.npy"
STATES = ROOT / "rgb_data/robot_state.npy"
VAE = Path(__import__("os").environ.get("VAE_PATH", "checkpoints/Wan2.2_VAE_bf16.safetensors"))
STEPS = 10000
BATCH = 32
MAX_TEXT = 128


def tokenize(texts, vocab, visible):
    lookup = {w: i for i, w in enumerate(vocab)}
    ids = torch.zeros(len(texts), MAX_TEXT, device="cuda", dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, t in enumerate(texts):
        if not visible[i]:
            continue
        words = ["<cls>"] + re.findall(r"\w+|[^\w\s]", t.casefold())
        assert len(words) <= MAX_TEXT
        nums = [2] + [lookup.get(w, 1) for w in words[1:]]
        ids[i, : len(nums)] = torch.tensor(nums, device="cuda")
        valid[i, : len(nums)] = True
    return ids, valid


class Records(Dataset):
    def __init__(self, rows, stats):
        self.rows = rows
        self.indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
        self.rgb = np.load(RGB, mmap_mode="r")
        self.state = np.load(STATES, mmap_mode="r")
        self.mean = np.array(stats["robot_mean"], np.float32)
        self.std = np.array(stats["robot_std"], np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        r = self.rows[i]
        video = np.array(self.rgb[i], copy=True)
        state = (np.array(self.state[i], copy=True) - self.mean) / self.std
        return {
            "rgb": torch.from_numpy(video),
            "current": torch.from_numpy(state[0]),
            "target": torch.from_numpy(state[1:]),
            "text": r["caption"],
            "index": j,
        }


def model_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["language", "none"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs")
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    np.random.seed(20260923)
    random.seed(20260923)
    assert json.loads((ROOT / "rgb_data/complete.json").read_text())["train"] == 14634
    rows = [json.loads(x) for x in (ROOT / "rgb_data/records.jsonl").read_text().splitlines()]
    stats = json.loads((ROOT / "rgb_data/stats.json").read_text())
    vocab = ["<pad>", "<unk>", "<cls>"] + sorted(
        {
            w
            for r in rows
            if r["split"] == "train"
            for w in re.findall(r"\w+|[^\w\s]", r["caption"].casefold())
        }
    )
    model = HRMoTFlowModel(HRMoTConfig(num_text_tokens=MAX_TEXT, text_vocab_size=len(vocab))).cuda()
    initial = model_hash(model)
    vae = FrozenWan22VideoEncoder.from_pretrained(
        VAE, device=torch.device("cuda"), clips_per_call=8
    )
    ds = Records(rows, stats)
    rng = np.random.default_rng(123000923)
    order = np.concatenate(
        [rng.permutation(len(ds)) for _ in range(math.ceil(a.steps * BATCH / len(ds)))]
    )[: a.steps * BATCH]
    loader = DataLoader(
        ds, batch_size=BATCH, sampler=order.tolist(), num_workers=4, pin_memory=True
    )
    out = a.out_root / a.mode
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(
        json.dumps(
            dict(
                mode=a.mode,
                from_scratch=True,
                train_records=len(ds),
                test_records=701,
                steps=a.steps,
                batch=BATCH,
                caption_dropout=0.5,
                initial_sha=initial,
                vae=str(VAE),
            ),
            indent=2,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    coverage = np.zeros(len(ds), np.int32)
    log = (out / "metrics.jsonl").open("w", buffering=1)
    start = time.monotonic()
    for step, batch in enumerate(loader, 1):
        b = len(batch["rgb"])
        rgb = (
            batch["rgb"]
            .cuda(non_blocking=True)
            .permute(0, 1, 4, 2, 3)
            .reshape(b * 9, 3, 240, 426)
            .float()
        )
        rgb = (
            F.interpolate(rgb, size=(224, 384), mode="bilinear", align_corners=False)
            .reshape(b, 9, 3, 224, 384)
            .permute(0, 2, 1, 3, 4)
            / 127.5
            - 1
        )
        robot = vae.encode_video(rgb).permute(0, 2, 1, 3, 4)
        keep = (
            torch.rand(
                b,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(130000000 + step),
            )
            >= 0.5
        )
        visible = keep if a.mode == "language" else torch.zeros_like(keep)
        ids, valid = tokenize(batch["text"], vocab, visible)
        inputs = dict(
            robot_current_latent=robot[:, :1],
            human_latents=torch.zeros_like(robot),
            robot_current_state=batch["current"].cuda(),
            human_future_action=torch.zeros((b, 32, 84), device="cuda"),
            robot_future_latents=robot[:, 1:],
            robot_future_action=batch["target"].cuda(),
            human_mask=torch.zeros((b, 3), device="cuda", dtype=torch.bool),
            human_motion_mask=torch.zeros((b, 3), device="cuda", dtype=torch.bool),
            text_token_ids=ids,
            text_valid=valid,
        )
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(124000000 + step)
        for group in opt.param_groups:
            group["lr"] = 1e-4 * min(1.0, step / 500)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model(**inputs)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(loss) and torch.isfinite(gn)
        text_grad = (
            sum(
                float(p.grad.abs().sum())
                for n, p in model.named_parameters()
                if ".text_" in n and p.grad is not None
            )
            if step == 20
            else None
        )
        opt.step()
        np.add.at(coverage, batch["index"].numpy(), 1)
        item = dict(
            step=step,
            loss=float(loss),
            grad_norm=float(gn),
            caption_keep=float(visible.float().mean()),
            seconds=time.monotonic() - start,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        log.write(json.dumps(item) + "\n")
        if step <= 3 or step % 20 == 0:
            (out / "progress.json").write_text(json.dumps(item))
            print(json.dumps(item), flush=True)
        if step == 20:
            assert (text_grad > 0) == (a.mode == "language")
            (out / "formal_verified.json").write_text(
                json.dumps(
                    dict(
                        actual_updates=20,
                        text_gradient_l1=text_grad,
                        coverage_sum=int(coverage.sum()),
                        initial_sha=initial,
                    )
                )
            )
        if step in (1000, 5000, 10000):
            ck = dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                config=model.config.__dict__,
                stats=stats,
                vocabulary=vocab,
                coverage=coverage,
                initial_sha=initial,
                mode=a.mode,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                next_sample=step * BATCH,
            )
            torch.save(ck, out / f"step{step:05d}.pt")
    assert coverage.sum() == a.steps * BATCH
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                updates=a.steps,
                visits=int(coverage.sum()),
                records=len(ds),
                coverage_min=int(coverage.min()),
                coverage_max=int(coverage.max()),
            )
        )
    )
    print("TRAIN_COMPLETE", a.mode, flush=True)


if __name__ == "__main__":
    main()
