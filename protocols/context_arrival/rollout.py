"""Four autoregressive 32-step blocks; demonstration arrives at physical horizon32."""

import argparse, json, os, time, sys
from pathlib import Path
import numpy as np
import torch
# Select the checkpoint's implementation before importing any fasterwam modules.
_package_parser = argparse.ArgumentParser(add_help=False)
_package_parser.add_argument("--model-package", choices=["main", "carrier_baselines"], default="main")
_package = _package_parser.parse_known_args()[0].model_package
_release = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_release / "src" if _package == "main" else
                       _release / "protocols" / _package / "src"))
from controls import resolve_domains, visible_at
from support import (
    sha,
    save,
    thash,
    LPIPS,
    ssim,
    RawHRStreamingDataset,
    StateStats,
    discover_raw_v1_episodes,
    dataset_geometry,
    select_record_ids,
    validate_record_files,
    validate_checkpoint_run_pair,
    validate_runtime_assets,
    HRMoTConfig,
    HRMoTFlowModel,
    FrozenWan22VideoEncoder,
)


@torch.inference_mode()
def main(a):
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    assert gpu in ["0", "1", "2", "3"]
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    video_domain, motion_domain = resolve_domains(a)
    root = a.output / (f"{a.baseline_type}_video_{video_domain}_motion_{motion_domain}"
                       + ("_smoke" if a.smoke else ""))
    root.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    checkpoint = a.run / "epoch_003_step_0010000.pt"
    assert sha(checkpoint) == a.checkpoint_sha
    run = json.loads((a.run / "config.json").read_text())
    artifact = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    validate_checkpoint_run_pair(artifact, run)
    validate_runtime_assets(run)
    cfg = HRMoTConfig(**artifact["config"])
    assert cfg.use_state_context and cfg.human_action_dim == 84
    if video_domain != "none" and not getattr(cfg, "use_human_video_context", True):
        raise ValueError("Checkpoint was trained with video disabled")
    if motion_domain != "none" and not cfg.use_human_motion_context:
        raise ValueError("Checkpoint was trained with motion disabled")
    model = HRMoTFlowModel(cfg)
    model.load_state_dict(artifact["model"], strict=True, assign=True)
    model = model.cuda().eval().requires_grad_(False)
    del artifact
    st = StateStats(**{k: tuple(v) for k, v in run["state_stats"].items()})
    ct = run["evaluation_contract"]
    records = discover_raw_v1_episodes(
        workspace_root=Path(ct["workspace_root"]),
        data_root=Path(ct["data_root"]),
        annotation_paths=[Path(x["path"]) for x in ct["annotations"]],
    )
    validate_record_files(records, ct)
    ds = RawHRStreamingDataset(
        select_record_ids(records, ct["holdout_record_ids"]),
        stats=st,
        **dataset_geometry(cfg),
        image_height=ct["image_height"],
        image_width=ct["image_width"],
        seed=ct["seed"] + 1,
        context=ct["context"],
        deterministic=True,
        windows_per_episode=16,
    )
    assert ds.context == "aligned" and cfg.action_horizon == 32
    metas = json.loads(a.queries.read_text())["scores_all64"]
    B = len(metas)
    assert 1 <= B <= 4
    recs = [ds.records[m["sample_index"] % len(ds.records)] for m in metas]
    for m, r in zip(metas, recs):
        assert str(r.annotation_path) == m["record_path"]
        assert m["robot_start_frame"] + 128 < min(r.robot_frames, r.human_frames)
    vae = FrozenWan22VideoEncoder.from_pretrained(
        ct["vae"]["path"], device=torch.device("cuda"), dtype=torch.bfloat16, clips_per_call=4
    )
    lpips = LPIPS(net="alex").cuda().eval().requires_grad_(False)
    mean = torch.tensor(st.robot_mean, device="cuda")
    std = torch.tensor(st.robot_std, device="cuda")
    spans = [8, 24] if a.protocol == "8plus24" else [32, 32, 32, 32]
    if a.smoke:
        spans = spans[:2]
    nblocks = len(spans)
    cut = spans[0]
    offsets = np.cumsum([0] + spans[:-1]).tolist()
    seedbase = 9000 + 100 * a.batch_index
    arrays = {}
    rows = []
    hist = [[] for m in metas]
    predstates = []
    truthstates = []
    predvideos = []
    truthvideos = []
    current_rgb = current_state = None
    save(
        root / "start.json",
        dict(
            pid=os.getpid(),
            physical_gpu=gpu,
            domain=a.domain,
            video_domain=video_domain, motion_domain=motion_domain,
            baseline_type=a.baseline_type, model_package=a.model_package,
            checkpoint=str(checkpoint),
            checkpoint_sha256=a.checkpoint_sha,
            config_sha256=sha(a.run / "config.json"),
            queries_sha256=sha(a.queries),
            script_sha256=sha(__file__),
            parameter_updates=0,
            blocks=nblocks,
            block_horizon=32,
            physical_intervention_horizon=cut,
            inference_steps_per_block=20,
            seeds=[seedbase + i for i in range(nblocks)],
            batch_size=B,
            robot_feedback="predicted decoded last RGB re-encoded as T1 plus predicted normalized last7D state; no ground-truth reset",
            started_unix=begin,
        ),
    )
    for block in range(nblocks):
        keep = spans[block]
        offset = offsets[block]
        starts = [m["robot_start_frame"] + offset for m in metas]
        targetparts = [ds._read_target(r, t) for r, t in zip(recs, starts)]
        contextparts = [ds._read_context(r, t) for r, t in zip(recs, starts)]
        assert all(x[3] == 0 for x in targetparts) and all(x[2] == 0 for x in contextparts)
        gtvideo = torch.stack([x[0].permute(1, 0, 2, 3) for x in targetparts]).cuda()
        gtcurrent = torch.stack([x[1] for x in targetparts]).cuda().float()
        target = torch.stack([x[2] for x in targetparts]).cuda().float()
        human = torch.stack([x[0].permute(1, 0, 2, 3) for x in contextparts]).cuda()
        motion = torch.stack([x[1] for x in contextparts]).cuda().float()
        if block == 0:
            current_rgb = gtvideo[:, :, :1].clone()
            current_state = gtcurrent.clone()
            arrays["initial_robot_rgb"] = current_rgb.cpu().numpy()
            arrays["initial_robot_state"] = current_state.cpu().numpy()
        else:
            assert torch.equal(current_rgb, previous_video[:, :, -1:])
            assert torch.equal(current_state, previous_state[:, -1])
        rz0 = vae.encode_video(current_rgb).float().permute(0, 2, 1, 3, 4)
        hz = vae.encode_video(human).float().permute(0, 2, 1, 3, 4)
        video_visible = visible_at(video_domain, block)
        motion_visible = visible_at(motion_domain, block)
        mask = torch.full((B, cfg.num_latent_steps), video_visible, device="cuda", dtype=torch.bool)
        motion_mask = torch.full_like(mask, motion_visible)
        inputs = dict(
            robot_current_latent=rz0,
            robot_current_state=current_state,
            human_latents=hz,
            human_future_action=motion,
            human_mask=mask,
            human_motion_mask=motion_mask,
            num_inference_steps=20,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, state = model.generate(
                **inputs, generator=torch.Generator(device="cuda").manual_seed(seedbase + block)
            )
        assert torch.isfinite(latent).all() and torch.isfinite(state).all()
        if a.smoke and block == 0 and not (video_visible and motion_visible):
            poisoned = dict(inputs)
            if not video_visible:
                poisoned["human_latents"] = hz * 3 + 1
            if not motion_visible:
                poisoned["human_future_action"] = motion * 3 + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                other = model.generate(
                    **poisoned, generator=torch.Generator(device="cuda").manual_seed(seedbase)
                )
            assert torch.equal(latent, other[0]) and torch.equal(state, other[1])
            del other, poisoned
        decoded = vae.decode_video(torch.cat([rz0, latent.float()], 1).permute(0, 2, 1, 3, 4))[
            :, :, 1:
        ].float()
        assert torch.isfinite(decoded).all()
        decoded = decoded[:, :, : keep // 4]
        state = state[:, :keep]
        target = target[:, :keep]
        for i, m in enumerate(metas):
            hist[i].append(
                dict(
                    block=block,
                    relative_anchor=offset,
                    retained_steps=keep,
                    human_last_relative_frame=offset + 32,
                    absolute_anchor=starts[i],
                    video_visible=video_visible,
                    motion_visible=motion_visible,
                    noise_seed=seedbase + block,
                    input_source="recorded_R0" if block == 0 else "previous_prediction",
                    input_rgb_sha256=thash(current_rgb[i]),
                    input_state_sha256=thash(current_state[i]),
                    input_latent_sha256=thash(rz0[i]),
                    gt_current_rgb_sha256=thash(gtvideo[i, :, :1]),
                    gt_current_state_sha256=thash(gtcurrent[i]),
                    gt_future_rgb_sha256=thash(gtvideo[i, :, 1 : 1 + keep // 4]),
                    human_rgb_sha256=thash(human[i]),
                    human_motion_sha256=thash(motion[i]),
                    output_last_rgb_sha256=thash(decoded[i, :, -1:]),
                    output_last_state_sha256=thash(state[i, -1]),
                    output_rgb_sha256=thash(decoded[i]),
                    output_state_sha256=thash(state[i]),
                )
            )
        arrays[f"block{block}_latent"] = latent.float().cpu().numpy()
        arrays[f"block{block}_input_rgb"] = current_rgb.cpu().numpy()
        arrays[f"block{block}_input_state"] = current_state.cpu().numpy()
        arrays[f"block{block}_human_rgb"] = human.cpu().numpy()
        predstates.append(state.float())
        truthstates.append(target)
        predvideos.append(decoded)
        truthvideos.append(gtvideo[:, :, 1 : 1 + keep // 4])
        previous_video = decoded
        previous_state = state
        current_rgb = decoded[:, :, -1:].clone()
        current_state = state[:, -1].clone()
        save(
            root / "progress.json",
            dict(
                blocks_completed=block + 1,
                physical_horizon=offset + keep,
                elapsed_seconds=time.time() - begin,
            ),
        )
        print(json.dumps(dict(domain=a.domain,
            video_domain=video_domain, motion_domain=motion_domain,
            baseline_type=a.baseline_type, model_package=a.model_package, block=block, horizon=offset + keep)), flush=True)
    state = torch.cat(predstates, 1)
    target = torch.cat(truthstates, 1)
    video = torch.cat(predvideos, 2)
    truth = torch.cat(truthvideos, 2)
    raw = state * std + mean
    gt = target * std + mean
    dist = (raw[..., :3] - gt[..., :3]).norm(dim=-1)
    pf = video.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 384)
    tf = truth.permute(0, 2, 1, 3, 4).reshape_as(pf)
    lp = torch.cat(
        [lpips(pf[i : i + 8], tf[i : i + 8]).flatten() for i in range(0, len(pf), 8)]
    ).reshape(B, -1)
    ss = ssim((pf + 1) * 0.5, (tf + 1) * 0.5, data_range=1.0, size_average=False).reshape(B, -1)
    mse = (pf - tf).square().flatten(1).mean(1).reshape(B, -1)
    psnr = 10 * torch.log10(4 / mse.clamp_min(1e-12))
    for i, m in enumerate(metas):
        rows.append(
            dict(
                **m,
                domain=a.domain,
            video_domain=video_domain, motion_domain=motion_domain,
            baseline_type=a.baseline_type, model_package=a.model_package,
                block_history=hist[i],
                translation_by_step=dist[i].cpu().tolist(),
                ade_mm=float(dist[i].mean()),
                post_intervention_ade_mm=float(dist[i, cut:].mean()),
                max_mm=float(dist[i].max()),
                post_intervention_max_mm=float(dist[i, cut:].max()),
                fde_mm=float(dist[i, -1]),
                all_scored_le20=bool((dist[i] <= 20).all()),
                post_intervention_all_le20=bool((dist[i, cut:] <= 20).all()),
                predicted_state_raw=raw[i].cpu().tolist(),
                target_state_raw=gt[i].cpu().tolist(),
                state_mse=float((state[i] - target[i]).square().mean()),
                gripper_mae=float((raw[i, :, 6] - gt[i, :, 6]).abs().mean()),
                gripper_by_step=raw[i, :, 6].cpu().tolist(),
                gt_gripper_by_step=gt[i, :, 6].cpu().tolist(),
                lpips_by_frame=lp[i].cpu().tolist(),
                ssim_by_frame=ss[i].cpu().tolist(),
                psnr_by_frame=psnr[i].cpu().tolist(),
                rgb_mse=float(mse[i].mean()),
                post_intervention_lpips=float(lp[i, cut // 4 :].mean()),
                initial_noise_seeds=[seedbase + j for j in range(nblocks)],
                within_batch_index=i,
            )
        )
    arrays.update(
        predicted_rgb=video.cpu().numpy(),
        truth_rgb=truth.cpu().numpy(),
        predicted_state=state.cpu().numpy(),
        target_state=target.cpu().numpy(),
    )
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    np.savez_compressed(root / "predictions.npz", **arrays)
    if a.smoke:
        save(
            root / "smoke_checks.json",
            dict(
                hidden_modalities_poison_invariance=(True if not (visible_at(video_domain, 0) and visible_at(motion_domain, 0)) else None),
                two_block_prediction_feedback=True,
                explicit_anchors_no_padding=True,
                all_model_parameters_frozen=True,
            ),
        )
    save(
        root / "complete.json",
        dict(
            rows=B,
            blocks=nblocks,
            parameter_updates=0,
            samples_sha256=sha(root / "samples.jsonl"),
            predictions_sha256=sha(root / "predictions.npz"),
            completed_unix=time.time(),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-sha", required=True)
    p.add_argument("--domain", choices=["none", "arrival", "full"], default="arrival")
    p.add_argument("--baseline-type", choices=["None", "Action", "Video", "Full"], default="Full",
                   help="Input preset; does not change the trained checkpoint architecture")
    p.add_argument("--video-domain", choices=["none", "arrival", "full"])
    p.add_argument("--motion-domain", choices=["none", "arrival", "full"])
    p.add_argument("--model-package", choices=["main", "carrier_baselines"], default="main")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--protocol", choices=["8plus24", "32plus96"], required=True)
    p.add_argument("--batch-index", type=int, default=0)
    main(p.parse_args())
