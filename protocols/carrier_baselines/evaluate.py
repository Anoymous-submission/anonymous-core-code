"""Frozen Full human-video appearance/time interventions; zero parameter updates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from lpips import LPIPS
from pytorch_msssim import ssim
from fasterwam.datasets.hr_raw_stream import (
    RawHRStreamingDataset,
    StateStats,
    discover_raw_v1_episodes,
)
from fasterwam.models.hr_eval import (
    dataset_geometry,
    select_record_ids,
    validate_record_files,
    validate_checkpoint_run_pair,
    validate_runtime_assets,
)
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel
from fasterwam.models.hr_vae import FrozenWan22VideoEncoder
from transforms import SPECS, replace_context


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def thash(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def dark(x):
    # Fixed before this evaluation; preserves geometry/time/labels, no test tuning.
    return (((x + 1) * 0.5).clamp(0, 1).pow(1.5) * 0.65) * 2 - 1


@torch.inference_mode()
def main(args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible in ["0", "1", "2", "3"], visible
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output / (args.domain + ("_smoke" if args.smoke else ""))
    output.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    checkpoint = args.run / "epoch_003_step_0010000.pt"
    expected = args.checkpoint_sha
    assert sha(checkpoint) == expected
    run = json.loads((args.run / "config.json").read_text())
    artifact = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    validate_checkpoint_run_pair(artifact, run)
    validate_runtime_assets(run)
    cfg = HRMoTConfig(**artifact["config"])
    assert cfg.use_robot_state_context and not cfg.use_human_video_context
    assert cfg.use_human_motion_context == (args.baseline_type == "Motion-only")
    assert args.baseline_type != "Robot-only" or args.domain == "ID"
    assert cfg.human_action_dim == 84
    model = HRMoTFlowModel(cfg)
    model.load_state_dict(artifact["model"], strict=True, assign=True)
    model = model.cuda().eval().requires_grad_(False)
    del artifact
    stats = StateStats(**{k: tuple(v) for k, v in run["state_stats"].items()})
    contract = run["evaluation_contract"]
    records = discover_raw_v1_episodes(
        workspace_root=Path(contract["workspace_root"]),
        data_root=Path(contract["data_root"]),
        annotation_paths=[Path(x["path"]) for x in contract["annotations"]],
    )
    assert [str(r.annotation_path) for r in records] == contract["inventory_record_ids"]
    validate_record_files(records, contract)
    dataset = RawHRStreamingDataset(
        select_record_ids(records, contract["holdout_record_ids"]),
        stats=stats,
        **dataset_geometry(cfg),
        image_height=contract["image_height"],
        image_width=contract["image_width"],
        seed=contract["seed"] + 1,
        context=contract["context"],
        deterministic=True,
        windows_per_episode=contract["eval_windows_per_episode"],
    )
    metadata = json.loads(args.queries.read_text())["scores_all64"]
    pairing = json.loads((Path(__file__).parent / "pairing.json").read_text())
    eligible = [r for r in metadata if r["sample_index"] in pairing["selected_indices"]]
    assert len(eligible) == len(pairing["selected_indices"]) and len(eligible) > 0
    if args.smoke:
        eligible = eligible[:4]
    vae = FrozenWan22VideoEncoder.from_pretrained(
        contract["vae"]["path"], device=torch.device("cuda"), dtype=torch.bfloat16, clips_per_call=4
    )
    perceptual = LPIPS(net="alex").cuda().eval().requires_grad_(False)
    mean = torch.tensor(stats.robot_mean, device="cuda")
    std = torch.tensor(stats.robot_std, device="cuda")
    conditions = ["correct"]
    save(
        output / "start.json",
        dict(
            pid=os.getpid(),
            physical_gpu=visible,
            domain=args.domain,
            checkpoint=str(checkpoint),
            checkpoint_sha256=expected,
            script_sha256=sha(__file__),
            queries_sha256=sha(args.queries),
            config_sha256=sha(args.run / "config.json"),
            parameter_updates=0,
            generation_seed=7000,
            inference_steps=20,
            n=len(eligible),
            intervention=SPECS[args.domain],
            conditions=conditions,
            unseen_tasks=False,
            baseline_type=args.baseline_type,
            independently_trained=True,
            natural_background_transfer=False,
            human_motion_unchanged=args.domain in ["ID", "video_shift_+8"],
            robot_rgb_and_targets_unchanged=True,
            transforms_sha256=sha(Path(__file__).with_name("transforms.py")),
            started_unix=begin,
        ),
    )
    rows, arrays = [], {}
    for bi, begin_index in enumerate(range(0, len(eligible), 4)):
        metas = eligible[begin_index : begin_index + 4]
        items = [dataset[r["sample_index"]] for r in metas]
        for r, item in zip(metas, items):
            assert int(item["robot_start_frame"]) == r["robot_start_frame"]
            assert str(item["target_path"]) == r["record_path"]
            assert int(item["overflow_padded_frames"]) == 0
        robot = torch.stack([x["robot_video"] for x in items]).cuda()
        human = torch.stack([x["human_video"] for x in items]).cuda()
        raw_robot = robot.clone()
        raw_human = human.clone()
        changed_human, changed_motion, transform_info = replace_context(
            dataset, items, metas, args.domain, pairing
        )
        human = changed_human.cuda()
        assert torch.equal(robot, raw_robot)
        if args.smoke:
            checks = {}
            for name in SPECS:
                a, am, ai = replace_context(dataset, items, metas, name, pairing)
                b, bm, bi_info = replace_context(dataset, items, metas, name, pairing)
                assert torch.equal(a, b) and torch.equal(am, bm) and ai == bi_info
                checks[name] = dict(shape=list(a.shape), deterministic=True, metadata=ai)
            save(output / "transform_checks.json", checks)
        current = torch.stack([x["robot_current_state"] for x in items]).cuda().float()
        original_motion = torch.stack([x["human_future_action"] for x in items]).cuda().float()
        motion = changed_motion.cuda().float()
        # Generator only ever receives R0. Future RGB/state are scored afterwards.
        rz0 = vae.encode_video(robot[:, :, :1]).float().permute(0, 2, 1, 3, 4)
        hz = vae.encode_video(human).float().permute(0, 2, 1, 3, 4)
        target = torch.stack([x["robot_future_state"] for x in items]).cuda().float()
        raw_target = target * std + mean
        if args.smoke:
            reference_z0 = vae.encode_video(robot).float().permute(0, 2, 1, 3, 4)[:, :1]
            assert torch.equal(rz0, reference_z0), "R0-only vs training clip encoder differs"
        cache = {}
        for condition in conditions:
            m = torch.full(
                (len(items), cfg.num_latent_steps),
                cfg.use_human_video_context,
                device="cuda",
                dtype=torch.bool,
            )
            mm = torch.full_like(m, cfg.use_human_motion_context)
            inputs = dict(
                robot_current_latent=rz0,
                robot_current_state=current,
                human_latents=hz,
                human_future_action=motion,
                human_mask=m,
                human_motion_mask=mm,
                num_inference_steps=20,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent, state = model.generate(
                    **inputs, generator=torch.Generator(device="cuda").manual_seed(7000 + bi)
                )
            assert torch.isfinite(latent).all() and torch.isfinite(state).all()
            if args.smoke:
                cache[condition] = (latent.clone(), state.clone())
            decoded = vae.decode_video(torch.cat([rz0, latent.float()], 1).permute(0, 2, 1, 3, 4))[
                :, :, 1:
            ].float()
            predicted = decoded.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 384)
            truth = robot[:, :, 1:].permute(0, 2, 1, 3, 4).reshape_as(predicted)
            lp = torch.cat(
                [
                    perceptual(predicted[i : i + 8], truth[i : i + 8], normalize=False).flatten()
                    for i in range(0, len(truth), 8)
                ]
            ).reshape(len(items), 8)
            ss = ssim(
                (predicted + 1) * 0.5, (truth + 1) * 0.5, data_range=1.0, size_average=False
            ).reshape(len(items), 8)
            mse = (predicted - truth).square().flatten(1).mean(1).reshape(len(items), 8)
            psnr = 10 * torch.log10(4 / mse.clamp_min(1e-12))
            raw_state = state.float() * std + mean
            distances = (raw_state[..., :3] - raw_target[..., :3]).norm(dim=-1)
            for i, meta in enumerate(metas):
                idx = meta["sample_index"]
                record = dict(
                    sample_index=idx,
                    instruction=meta["instruction"],
                    domain=args.domain,
                    condition=condition,
                    robot_start=meta["robot_start_frame"],
                    record_path=meta["record_path"],
                    intervention=transform_info[i],
                    predicted_state_raw=raw_state[i].cpu().tolist(),
                    target_state_raw=raw_target[i].cpu().tolist(),
                    translation_by_step=distances[i].cpu().tolist(),
                    ade_mm=float(distances[i].mean()),
                    threshold_all32_20mm=bool((distances[i] <= 20).all()),
                    state_mse=float((state[i].float() - target[i]).square().mean()),
                    gripper_mae=float((raw_state[i, :, 6] - raw_target[i, :, 6]).abs().mean()),
                    rgb_mse=float(mse[i].mean()),
                    lpips_by_frame=lp[i].cpu().tolist(),
                    ssim_by_frame=ss[i].cpu().tolist(),
                    psnr_by_frame=psnr[i].cpu().tolist(),
                    query_rgb_sha256=thash(robot[i, :, :1]),
                    original_query_rgb_sha256=thash(raw_robot[i, :, :1]),
                    robot_state_sha256=thash(current[i]),
                    human_motion_sha256=thash(motion[i]),
                    original_human_motion_sha256=thash(original_motion[i]),
                    original_human_rgb_sha256=thash(raw_human[i]),
                    human_rgb_sha256=thash(human[i]),
                    target_rgb_sha256=thash(robot[i, :, 1:]),
                    human_video_mask=m[i].cpu().tolist(),
                    human_motion_mask=mm[i].cpu().tolist(),
                    initial_noise_seed=7000 + bi,
                    within_batch_index=i,
                    latent_sha256=thash(latent[i]),
                    state_sha256=thash(state[i]),
                )
                rows.append(record)
                with (output / "samples.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                arrays[f"{idx}_{condition}_latent"] = latent[i].float().cpu().numpy()
                arrays[f"{idx}_{condition}_state"] = state[i].float().cpu().numpy()
                if idx in [eligible[0]["sample_index"], 47, 28]:
                    arrays[f"{idx}_{condition}_decoded"] = decoded[i].cpu().numpy()
                    arrays[f"{idx}_truth_rgb"] = robot[i, :, 1:].cpu().numpy()
                    arrays[f"{idx}_human_rgb"] = human[i].cpu().numpy()
                    arrays[f"{idx}_original_human_rgb"] = raw_human[i].cpu().numpy()
        if args.smoke:
            inputs.update(
                human_latents=hz * 2 + 1,
                human_future_action=motion if cfg.use_human_motion_context else motion * 3 + 1,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                altered = model.generate(
                    **inputs, generator=torch.Generator(device="cuda").manual_seed(7000 + bi)
                )
            assert all(
                torch.equal(x, y) for x, y in zip(altered, cache["correct"])
            ), "Disabled modality leak"
            save(
                output / "smoke_checks.json",
                dict(
                    robot_r0_only_matches_clip=True,
                    disabled_modality_invariance=True,
                    all_transforms_deterministic=True,
                    robot_future_not_in_generator_inputs=True,
                    all_model_parameters_frozen=not any(
                        p.requires_grad for p in model.parameters()
                    ),
                ),
            )
        save(
            output / "progress.json",
            dict(batches=bi + 1, rows=len(rows), elapsed_seconds=time.time() - begin),
        )
        print(
            json.dumps(
                dict(
                    domain=args.domain,
                    batches=bi + 1,
                    rows=len(rows),
                    elapsed_seconds=time.time() - begin,
                )
            ),
            flush=True,
        )
    assert len(rows) == len(eligible) * len(conditions)
    np.savez_compressed(output / "predictions.npz", **arrays)
    summary = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        summary[condition] = {
            k: float(np.mean([r[k] for r in subset]))
            for k in ["ade_mm", "state_mse", "gripper_mae", "rgb_mse", "threshold_all32_20mm"]
        }
        for k in ["lpips", "ssim", "psnr"]:
            summary[condition][k] = float(np.mean([r[k + "_by_frame"] for r in subset]))
    save(
        output / "complete.json",
        dict(
            passed=True,
            smoke=args.smoke,
            domain=args.domain,
            n=len(eligible),
            rows=len(rows),
            summary=summary,
            parameter_updates=0,
            elapsed_seconds=time.time() - begin,
            samples_sha256=sha(output / "samples.jsonl"),
            predictions_sha256=sha(output / "predictions.npz"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domain", choices=list(SPECS), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--checkpoint-sha", required=True)
    p.add_argument("--baseline-type", choices=["Motion-only", "Robot-only"], required=True)
    main(p.parse_args())
