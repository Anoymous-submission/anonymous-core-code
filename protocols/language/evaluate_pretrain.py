"""Fixed final-checkpoint evaluation on held-out, causally paired records."""

import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel
from fasterwam.models.hr_vae import FrozenWan22VideoEncoder
from train_pretrain import ROOT, RGB, STATES, VAE, tokenize
from visual_metrics import gaussian_ssim


class TestRecords(Dataset):
    def __init__(self, rows, stats):
        self.rows = [(i, r) for i, r in enumerate(rows) if r["split"] == "test"]
        self.rgb = np.load(RGB, mmap_mode="r")
        self.state = np.load(STATES, mmap_mode="r")
        self.mean = np.array(stats["robot_mean"], np.float32)
        self.std = np.array(stats["robot_std"], np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        i, r = self.rows[j]
        return dict(
            rgb=torch.from_numpy(np.array(self.rgb[i], copy=True)),
            target=torch.from_numpy(
                (np.array(self.state[i, 1:], copy=True) - self.mean) / self.std
            ),
            text=r["caption"],
            index=j,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["language", "none"], required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(ROOT / "runs" / a.mode / "step10000.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 10000 and ck["mode"] == a.mode
    model = HRMoTFlowModel(HRMoTConfig(**ck["config"])).cuda().eval()
    model.load_state_dict(ck["model"])
    stats, vocab = ck["stats"], ck["vocabulary"]
    del ck
    rows = [json.loads(x) for x in (ROOT / "rgb_data/records.jsonl").read_text().splitlines()]
    data = TestRecords(rows, stats)
    assert len(data) == 701
    vae = FrozenWan22VideoEncoder.from_pretrained(
        VAE, device=torch.device("cuda"), clips_per_call=4
    )
    mean = torch.tensor(stats["robot_mean"], device="cuda")
    std = torch.tensor(stats["robot_std"], device="cuda")
    summaries = {}
    for condition in (["correct", "no_text"] if a.mode == "language" else ["no_text"]):
        for seed in (0, 1, 2):
            out = ROOT / "evaluation" / a.mode / condition / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            scores = []
            loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
            for batch in loader:
                n = len(batch["rgb"])
                index = batch["index"].numpy().tolist()
                rgb = (
                    batch["rgb"]
                    .cuda(non_blocking=True)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(n * 9, 3, 240, 426)
                    .float()
                )
                rgb = (
                    F.interpolate(rgb, size=(224, 384), mode="bilinear", align_corners=False)
                    .reshape(n, 9, 3, 224, 384)
                    .permute(0, 2, 1, 3, 4)
                    / 127.5
                    - 1
                )
                z = vae.encode_video(rgb).permute(0, 2, 1, 3, 4)
                visible = torch.full(
                    (n,),
                    a.mode == "language" and condition == "correct",
                    device="cuda",
                    dtype=torch.bool,
                )
                tid, tvalid = tokenize(batch["text"], vocab, visible)
                inputs = dict(
                    robot_current_latent=z[:, :1],
                    human_latents=torch.zeros_like(z),
                    robot_current_state=torch.zeros(n, 7, device="cuda"),
                    human_future_action=torch.zeros(n, 32, 84, device="cuda"),
                    human_mask=torch.zeros(n, 3, device="cuda", dtype=torch.bool),
                    human_motion_mask=torch.zeros(n, 3, device="cuda", dtype=torch.bool),
                    text_token_ids=tid,
                    text_valid=tvalid,
                )
                # Current state is part of the query, identical to training.
                state = np.stack(
                    [np.array(data.state[data.rows[j][0], 0], copy=True) for j in index]
                )
                inputs["robot_current_state"] = (torch.from_numpy(state).cuda() - mean) / std
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pz, ps = model.generate(
                        **inputs,
                        num_inference_steps=20,
                        generator=torch.Generator(device="cuda").manual_seed(
                            124000000 + seed * 10000 + index[0]
                        ),
                    )
                truth = batch["target"].cuda() * std + mean
                pred = ps.float() * std + mean
                dist = torch.linalg.vector_norm(pred[:, :, :3] - truth[:, :, :3], dim=-1)
                decoded = vae.decode_video(
                    torch.cat([z[:, :1], pz], 1).permute(0, 2, 1, 3, 4)
                ).float()
                output = ((decoded[:, :, 1:] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
                target = ((rgb[:, :, 1:] + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
                x = output.permute(0, 2, 1, 3, 4).flatten(0, 1).double() / 255
                y = target.permute(0, 2, 1, 3, 4).flatten(0, 1).double() / 255
                mse = (x - y).square().flatten(1).mean(1).reshape(n, 8)
                ssim = gaussian_ssim(x, y).reshape(n, 8)
                for k, j in enumerate(index):
                    r = data.rows[j][1]
                    assert r["caption_last_frame"] <= r["query_start"]
                    scores.append(
                        dict(
                            uid=r["uid"],
                            caption_uid=r["caption_uid"],
                            path=r["path"],
                            query_start=r["query_start"],
                            caption_last_frame=r["caption_last_frame"],
                            seed=seed,
                            condition=condition,
                            ade_mm=float(dist[k].mean()),
                            fde_mm=float(dist[k, -1]),
                            state_mse=float(
                                (ps[k].float() - batch["target"][k].cuda()).square().mean()
                            ),
                            latent_mse=float((pz[k].float() - z[k, 1:]).square().mean()),
                            pixel_mse=float(mse[k].mean()),
                            ssim=float(ssim[k].mean()),
                        )
                    )
                (out / "progress.json").write_text(json.dumps(dict(records=len(scores))))
            (out / "scores.json").write_text(json.dumps(scores))
            sm = {
                k: float(np.mean([r[k] for r in scores]))
                for k in ("ade_mm", "fde_mm", "state_mse", "latent_mse", "pixel_mse", "ssim")
            }
            sm["psnr_db"] = float(-10 * np.log10(sm["pixel_mse"]))
            sm["n"] = len(scores)
            (out / "summary.json").write_text(json.dumps(sm, indent=2))
            summaries[f"{condition}/seed{seed}"] = sm
            print("EVAL_COMPLETE", a.mode, condition, seed, json.dumps(sm), flush=True)
    (ROOT / "evaluation" / a.mode / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
