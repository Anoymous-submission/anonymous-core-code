"""Does the human context actually matter? Generation-from-noise evaluation.

A teacher-forced denoising metric cannot answer this. Feeding
`(1-sigma)*true_future + sigma*noise` at sigma=0.2 hands the model 80% of the
answer, so it can denoise from the noisy future plus the robot's own current
frame and motion priors without ever consulting the human stream. Any such
metric is blind to whether the human context is used.

This script instead runs the real inference path: robot future latents and
states start as pure Gaussian noise and are produced by the full flow
integration, conditioned only on the robot z0 latent step and the human stream.
The generated trajectory is then compared against ground truth.

Controls, in increasing strength:

  real        the human episode the sample was paired with (same instruction)
  other_task  a human episode from a DIFFERENT instruction group -- the real
              negative. A permutation within the same instruction group is still
              a legitimate context, so leaving the loss unchanged there proves
              nothing. Replacements come from the full held-out set and every
              sample is verified to change instruction.
  none        human future steps masked out entirely

If `real` does not beat `other_task`, the human content is not being used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lpips import LPIPS
from pytorch_msssim import ssim

from fasterwam.datasets.hr_raw_stream import (
    RawHRStreamingDataset,
    StateStats,
    discover_raw_v1_episodes,
)
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel
from fasterwam.models.hr_eval import (
    dataset_geometry,
    different_instruction_indices,
    future_pixel_mse,
    select_record_ids,
    validate_checkpoint_run_pair,
    validate_record_files,
    validate_runtime_assets,
)
from fasterwam.models.hr_vae import FrozenWan22VideoEncoder
from fasterwam.models.hr_inference import denormalize_state


def offline_success_proxy_counts(
    translation_mm: torch.Tensor,
    thresholds_mm: tuple[float, ...],
    horizon_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts for the explicitly offline trajectory-error success proxy."""

    selected = translation_mm.index_select(1, horizon_indices)
    by_horizon = torch.stack([(selected <= threshold).sum(0) for threshold in thresholds_mm])
    final = torch.stack([(translation_mm[:, -1] <= threshold).sum() for threshold in thresholds_mm])
    all_steps = torch.stack(
        [(translation_mm.amax(dim=1) <= threshold).sum() for threshold in thresholds_mm]
    )
    return by_horizon, final, all_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=7000,
        help="Base seed for matched Gaussian starts; batch index is added.",
    )
    parser.add_argument(
        "--cross-modal",
        action="store_true",
        help=(
            "Separate human-video and 84-D human-motion interventions instead "
            "of replacing both carriers together."
        ),
    )
    parser.add_argument(
        "--availability-grid",
        action="store_true",
        help=(
            "Evaluate all eight keep/drop patterns over human latent steps "
            "z0/z1/z2. Motion tokens mapped to a dropped step are masked too."
        ),
    )
    parser.add_argument(
        "--carrier-availability-grid",
        action="store_true",
        help=(
            "Evaluate full, motion-only, video-only, and no-human conditions "
            "using separate hard attention masks for the two carriers."
        ),
    )
    parser.add_argument(
        "--video-availability-grid",
        action="store_true",
        help=(
            "Evaluate all eight keep/drop patterns over human-video z0/z1/z2 "
            "while always retaining the complete 84-D human-motion context."
        ),
    )
    parser.add_argument("--decode-pixels", action="store_true")
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Evaluate only the checkpoint's native, correctly aligned context.",
    )
    parser.add_argument(
        "--success-threshold-mm",
        type=float,
        action="append",
        default=None,
        help="Pre-registered offline trajectory-error proxy threshold; repeatable.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output is None:
        raise ValueError("This frozen ablation evaluator requires --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_output = args.output.with_suffix(".samples.jsonl")
    if args.output.exists() or sample_output.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation")
    modes = (
        args.cross_modal,
        args.availability_grid,
        args.carrier_availability_grid,
        args.video_availability_grid,
    )
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError(
            "--cross-modal, --availability-grid, --carrier-availability-grid, "
            "and --video-availability-grid are mutually exclusive"
        )
    device = torch.device("cuda")
    run_config = json.loads((args.run_dir / "config.json").read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_run_pair(checkpoint, run_config)
    validate_runtime_assets(run_config)
    contract = run_config["evaluation_contract"]
    raw_stats = run_config["state_stats"]
    stats = StateStats(
        robot_mean=tuple(raw_stats["robot_mean"]),
        robot_std=tuple(raw_stats["robot_std"]),
        human_action_mean=tuple(raw_stats["human_action_mean"]),
        human_action_std=tuple(raw_stats["human_action_std"]),
    )

    config = HRMoTConfig(**checkpoint["config"])
    model = HRMoTFlowModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if config.direct_fastwam_robot_only and not args.real_only:
        raise ValueError(
            "direct FastWAM is robot-only; use --real-only so the evaluator does "
            "not construct meaningless human-context interventions"
        )
    success_thresholds = tuple(args.success_threshold_mm or (10.0, 20.0, 50.0))
    encoder = FrozenWan22VideoEncoder.from_pretrained(
        contract["vae"]["path"], device=device, dtype=torch.bfloat16
    )
    perceptual = LPIPS(net="alex").to(device).eval().requires_grad_(False)

    records = discover_raw_v1_episodes(
        workspace_root=Path(contract["workspace_root"]),
        data_root=Path(contract["data_root"]),
        annotation_paths=[Path(item["path"]) for item in contract["annotations"]],
    )
    seed = int(contract["seed"])
    inventory_ids = [str(record.annotation_path) for record in records]
    if inventory_ids != contract["inventory_record_ids"]:
        raise ValueError("current raw-data inventory differs from the training contract")
    validate_record_files(records, contract)
    eval_records = select_record_ids(records, contract["holdout_record_ids"])
    dataset = RawHRStreamingDataset(
        eval_records,
        stats=stats,
        **dataset_geometry(config),
        image_height=int(contract["image_height"]),
        image_width=int(contract["image_width"]),
        seed=seed + 1,
        # A checkpoint trained with cross_episode context must be scored the same
        # way, or the ablation reads as a null result for the wrong reason.
        context=contract["context"],
        windows_per_episode=int(contract["eval_windows_per_episode"]),
        deterministic=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    other_task_index = different_instruction_indices(
        [record.group_key for record in dataset.records]
    )

    availability_patterns = {
        "full_z0_z1_z2": (True, True, True),
        "drop_z0": (False, True, True),
        "drop_z1": (True, False, True),
        "drop_z2": (True, True, False),
        "only_z0": (True, False, False),
        "only_z1": (False, True, False),
        "only_z2": (False, False, True),
        "none": (False, False, False),
    }
    carrier_patterns = {
        "full_video_motion": {"video": True, "motion": True},
        "motion_only_no_video": {"video": False, "motion": True},
        "video_only_no_motion": {"video": True, "motion": False},
        "none": {"video": False, "motion": False},
    }
    video_availability_patterns = {
        "video_full_z0_z1_z2": (True, True, True),
        "video_drop_z0": (False, True, True),
        "video_drop_z1": (True, False, True),
        "video_drop_z2": (True, True, False),
        "video_only_z0": (True, False, False),
        "video_only_z1": (False, True, False),
        "video_only_z2": (False, False, True),
        "video_none": (False, False, False),
    }
    if args.carrier_availability_grid:
        variants = tuple(carrier_patterns)
    elif args.video_availability_grid:
        if config.num_latent_steps != 3:
            raise ValueError(
                "--video-availability-grid requires exactly three human latent "
                f"steps z0/z1/z2, got {config.num_latent_steps}"
            )
        variants = tuple(video_availability_patterns)
    elif args.availability_grid:
        if config.num_latent_steps != 3:
            raise ValueError(
                "--availability-grid requires exactly three human latent steps "
                f"z0/z1/z2, got {config.num_latent_steps}"
            )
        if not config.droppable_human_current:
            raise ValueError(
                "--availability-grid requires a checkpoint trained with droppable human z0"
            )
        variants = tuple(availability_patterns)
    elif args.cross_modal:
        variants = (
            "real_video_real_motion",
            "real_video_wrong_motion",
            "wrong_video_real_motion",
            "wrong_video_wrong_motion",
            "none",
        )
    elif args.real_only:
        variants = ("real",)
    else:
        variants = ("real", "other_task", "none")
    state_horizons = (1, 4, 8, 16, 32)
    state_horizon_indices = torch.tensor(
        [step - 1 for step in state_horizons], dtype=torch.long, device=device
    )
    video_horizons = tuple(
        int(config.frame_stride) * step for step in range(1, int(config.num_rgb_frames))
    )
    totals = {
        v: {
            "latent": torch.zeros(config.num_horizons, dtype=torch.float64, device=device),
            "state": torch.zeros((), dtype=torch.float64, device=device),
            "pixel": torch.zeros((), dtype=torch.float64, device=device),
            "translation_ade": torch.zeros((), dtype=torch.float64, device=device),
            "translation_horizon": torch.zeros(
                len(state_horizons), dtype=torch.float64, device=device
            ),
            "state_mse_horizon": torch.zeros(
                len(state_horizons), dtype=torch.float64, device=device
            ),
            "lpips": torch.zeros((), dtype=torch.float64, device=device),
            "ssim": torch.zeros((), dtype=torch.float64, device=device),
            "psnr": torch.zeros((), dtype=torch.float64, device=device),
            "lpips_horizon": torch.zeros(len(video_horizons), dtype=torch.float64, device=device),
            "ssim_horizon": torch.zeros(len(video_horizons), dtype=torch.float64, device=device),
            "psnr_horizon": torch.zeros(len(video_horizons), dtype=torch.float64, device=device),
            "latent_cosine": torch.zeros(config.num_horizons, dtype=torch.float64, device=device),
            "success_horizon": torch.zeros(
                (len(success_thresholds), len(state_horizons)), dtype=torch.float64, device=device
            ),
            "success_final": torch.zeros(
                len(success_thresholds), dtype=torch.float64, device=device
            ),
            "success_all32": torch.zeros(
                len(success_thresholds), dtype=torch.float64, device=device
            ),
        }
        for v in variants
    }
    swapped_groups = 0
    count = 0.0

    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        robot = encoder.encode_video(batch["robot_video"].to(device)).float()
        robot = robot.permute(0, 2, 1, 3, 4).contiguous()
        if config.direct_fastwam_robot_only:
            # Do not even encode human pixels for the physically robot-only model.
            human = torch.zeros_like(robot)
        else:
            human = encoder.encode_video(batch["human_video"].to(device)).float()
            human = human.permute(0, 2, 1, 3, 4).contiguous()
        target_latent = robot[:, 1:]
        target_state = batch["robot_future_state"].to(device, torch.float32)
        robot_now = batch["robot_current_state"].to(device, torch.float32)
        human_now = batch["human_future_action"].to(device, torch.float32)
        size = robot.shape[0]

        other_human = None
        other_human_now = None
        negative_metadata = []
        if not (
            args.availability_grid
            or args.carrier_availability_grid
            or args.video_availability_grid
            or args.real_only
        ):
            # Construct the negative from the entire held-out record set, not
            # from this batch. Every replacement is checked to have a different
            # target instruction; there is deliberately no same-task fallback.
            other_videos = []
            other_states = []
            for sample_index, target_index in zip(
                batch["sample_index"].tolist(), batch["target_index"].tolist()
            ):
                replacement = other_task_index[int(target_index)]
                target_group = dataset.records[int(target_index)].group_key
                replacement_record = dataset.records[replacement]
                if replacement_record.group_key == target_group:
                    raise RuntimeError("other_task replacement has the same instruction")
                negative_rng = np.random.default_rng((seed + 17) * 1_000_003 + int(sample_index))
                negative_start = dataset._start_frame(negative_rng, replacement_record.human_frames)
                negative_video, negative_state, _ = dataset._read_context(
                    replacement_record, negative_start
                )
                negative_metadata.append(
                    {
                        "path": str(replacement_record.annotation_path),
                        "group": replacement_record.group_key,
                        "start_frame": int(negative_start),
                    }
                )
                other_videos.append(negative_video.permute(1, 0, 2, 3).contiguous())
                other_states.append(negative_state)
                swapped_groups += 1
            other_human_video = torch.stack(other_videos).to(device)
            other_human = encoder.encode_video(other_human_video).float()
            other_human = other_human.permute(0, 2, 1, 3, 4).contiguous()
            other_human_now = torch.stack(other_states).to(device, torch.float32)

        full_mask = torch.ones((size, config.num_latent_steps), dtype=torch.bool, device=device)
        none_mask = torch.zeros_like(full_mask)
        # A model trained with --pin-human-current never saw the all-missing
        # pattern. Preserve that legacy contract; the default droppable model
        # gets the true no-human condition it was trained on.
        if not config.droppable_human_current:
            none_mask[:, 0] = True
        availability_masks = (
            {
                name: torch.tensor(pattern, dtype=torch.bool, device=device)[None, :]
                .expand(size, -1)
                .contiguous()
                for name, pattern in availability_patterns.items()
            }
            if args.availability_grid
            else None
        )
        video_availability_masks = (
            {
                name: torch.tensor(pattern, dtype=torch.bool, device=device)[None, :]
                .expand(size, -1)
                .contiguous()
                for name, pattern in video_availability_patterns.items()
            }
            if args.video_availability_grid
            else None
        )

        for variant in variants:
            human_motion_mask = None
            if args.carrier_availability_grid:
                video_available = carrier_patterns[variant]["video"]
                motion_available = carrier_patterns[variant]["motion"]
                mask = full_mask if video_available else none_mask
                human_motion_mask = full_mask if motion_available else none_mask
                context, context_state = human, human_now
            elif args.video_availability_grid:
                context, context_state = human, human_now
                mask = video_availability_masks[variant]  # type: ignore[index]
                human_motion_mask = full_mask
            elif args.availability_grid:
                context, context_state, mask = (
                    human,
                    human_now,
                    availability_masks[variant],  # type: ignore[index]
                )
            elif variant in ("real", "real_video_real_motion"):
                context, context_state, mask = human, human_now, full_mask
            elif variant in ("other_task", "wrong_video_wrong_motion"):
                context, context_state, mask = other_human, other_human_now, full_mask
            elif variant == "real_video_wrong_motion":
                context, context_state, mask = human, other_human_now, full_mask
            elif variant == "wrong_video_real_motion":
                context, context_state, mask = other_human, human_now, full_mask
            else:
                context, context_state, mask = human, human_now, none_mask
            # Pure Gaussian start, full flow integration -- the real inference path.
            generator = torch.Generator(device=device).manual_seed(args.generation_seed + index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents, state = model.generate(
                    robot_current_latent=robot[:, :1],
                    human_latents=context,
                    robot_current_state=robot_now,
                    # Human hand motion is part of the context: swapping the
                    # demonstration must swap it too, or the ablation leaks the
                    # correct human trajectory into the wrong condition.
                    human_future_action=context_state,
                    num_inference_steps=args.inference_steps,
                    generator=generator,
                    human_mask=mask,
                    human_motion_mask=human_motion_mask,
                )
            latents = latents.float()
            totals[variant]["latent"] += (
                (latents - target_latent).pow(2).flatten(2).mean(2).sum(0).double()
            )
            totals[variant]["latent_cosine"] += (
                F.cosine_similarity(latents.flatten(2), target_latent.flatten(2), dim=2)
                .sum(0)
                .double()
            )
            totals[variant]["state"] += (
                (state.float() - target_state).pow(2).mean((1, 2)).sum().double()
            )
            selected_pred = state.float().index_select(1, state_horizon_indices)
            selected_true = target_state.index_select(1, state_horizon_indices)
            totals[variant]["state_mse_horizon"] += (
                (selected_pred - selected_true).pow(2).mean(2).sum(0).double()
            )
            state_raw = denormalize_state(state.float(), stats)
            target_raw = denormalize_state(target_state, stats)
            translation = torch.linalg.vector_norm(state_raw[..., :3] - target_raw[..., :3], dim=-1)
            totals[variant]["translation_ade"] += translation.mean(1).sum().double()
            totals[variant]["translation_horizon"] += (
                translation.index_select(1, state_horizon_indices).sum(0).double()
            )
            success_horizon, success_final, success_all32 = offline_success_proxy_counts(
                translation, success_thresholds, state_horizon_indices
            )
            totals[variant]["success_horizon"] += success_horizon.double()
            totals[variant]["success_final"] += success_final.double()
            totals[variant]["success_all32"] += success_all32.double()
            predicted = encoder.decode_video(
                torch.cat((robot[:, :1], latents), dim=1).permute(0, 2, 1, 3, 4)
            ).float()
            truth = batch["robot_video"].to(device).float()
            totals[variant]["pixel"] += future_pixel_mse(predicted, truth).sum().double()
            pred_future = predicted[:, :, 1:].clamp(-1, 1)
            true_future = truth[:, :, 1:].clamp(-1, 1)
            batch_size, _, future_frames, height, width = pred_future.shape
            pred_images = pred_future.permute(0, 2, 1, 3, 4).reshape(
                batch_size * future_frames, 3, height, width
            )
            true_images = true_future.permute(0, 2, 1, 3, 4).reshape(
                batch_size * future_frames, 3, height, width
            )
            lpips_values = []
            for image_start in range(0, pred_images.shape[0], 8):
                lpips_values.append(
                    perceptual(
                        pred_images[image_start : image_start + 8],
                        true_images[image_start : image_start + 8],
                        normalize=False,
                    ).flatten()
                )
            lpips_values = torch.cat(lpips_values).reshape(batch_size, future_frames)
            # SSIM conventionally consumes [0, 1]; shifting both inputs matters
            # because its luminance term is not invariant to a constant offset.
            ssim_values = ssim(
                (pred_images + 1.0) * 0.5,
                (true_images + 1.0) * 0.5,
                data_range=1.0,
                size_average=False,
            ).reshape(batch_size, future_frames)
            frame_mse = F.mse_loss(pred_images, true_images, reduction="none").flatten(1).mean(1)
            psnr_values = (10.0 * torch.log10(4.0 / frame_mse.clamp_min(1e-12))).reshape(
                batch_size, future_frames
            )
            totals[variant]["lpips"] += lpips_values.mean(1).sum().double()
            totals[variant]["ssim"] += ssim_values.mean(1).sum().double()
            totals[variant]["psnr"] += psnr_values.mean(1).sum().double()
            totals[variant]["lpips_horizon"] += lpips_values.sum(0).double()
            totals[variant]["ssim_horizon"] += ssim_values.sum(0).double()
            totals[variant]["psnr_horizon"] += psnr_values.sum(0).double()
            if not torch.isfinite(state).all() or not torch.isfinite(latents).all():
                raise ValueError("Non-finite generated predictions")
            sample_state_mse = (state.float() - target_state).pow(2).mean((1, 2))
            with sample_output.open("a") as handle:
                for row in range(size):
                    sample = {
                        "generation_seed": args.generation_seed,
                        "batch_index": index,
                        "variant": variant,
                        "sample_index": int(batch["sample_index"][row]),
                        "target_index": int(batch["target_index"][row]),
                        "target_path": str(batch["target_path"][row]),
                        "context_path": str(batch["context_path"][row]),
                        "negative_context": negative_metadata[row] if negative_metadata else None,
                        "target_group": batch["target_group"][row],
                        "robot_start_frame": int(batch["robot_start_frame"][row]),
                        "human_start_frame": int(batch["human_start_frame"][row]),
                        "overflow_padded_frames": batch["overflow_padded_frames"][row].tolist(),
                        "state": float(sample_state_mse[row]),
                        "predicted_state_normalized": state[row].float().cpu().tolist(),
                        "target_state_normalized": target_state[row].cpu().tolist(),
                        "translation_by_step": translation[row].cpu().tolist(),
                        "lpips_by_frame": lpips_values[row].cpu().tolist(),
                        "ssim_by_frame": ssim_values[row].cpu().tolist(),
                        "psnr_by_frame_db": psnr_values[row].cpu().tolist(),
                    }
                    handle.write(json.dumps(sample, allow_nan=False) + "\n")
        count += size
        print(
            json.dumps(
                {
                    "event": "batch_complete",
                    "batch": index + 1,
                    "samples": count,
                    "variants": len(variants),
                }
            ),
            flush=True,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "protocol": "generation from pure Gaussian noise, full flow integration",
        "inference_steps": args.inference_steps,
        "generation_seed": args.generation_seed,
        "real_only": args.real_only,
        "cross_modal": args.cross_modal,
        "availability_grid": args.availability_grid,
        "carrier_availability_grid": args.carrier_availability_grid,
        "video_availability_grid": args.video_availability_grid,
        "availability_patterns": (
            {
                name: [int(value) for value in pattern]
                for name, pattern in availability_patterns.items()
            }
            if args.availability_grid
            else None
        ),
        "carrier_availability_patterns": (
            carrier_patterns if args.carrier_availability_grid else None
        ),
        "video_availability_patterns": (
            {
                name: [int(value) for value in pattern]
                for name, pattern in video_availability_patterns.items()
            }
            if args.video_availability_grid
            else None
        ),
        "human_motion_mask": ([1, 1, 1] if args.video_availability_grid else None),
        "samples": count,
        "context_swaps_to_a_different_instruction": (
            None
            if (
                args.availability_grid
                or args.carrier_availability_grid
                or args.video_availability_grid
                or args.real_only
            )
            else int(swapped_groups)
        ),
        "all_other_task_samples_have_different_instruction": (
            None
            if (
                args.availability_grid
                or args.carrier_availability_grid
                or args.video_availability_grid
                or args.real_only
            )
            else swapped_groups == count
        ),
        "none_context_keeps_human_z0": not config.droppable_human_current,
        "architecture": (
            "direct_fastwam_robot_only"
            if config.direct_fastwam_robot_only
            else "hr_mot_human_carrier_sequence"
        ),
        "translation_units": "native end_position units (dataset inspection indicates millimetres)",
        "offline_rollout_success_proxy": {
            "accepted_definition": "translation trajectory error below a fixed threshold",
            "thresholds_mm": list(success_thresholds),
            "not_real_environment_success": True,
            "all32_definition": (
                "every one of the 32 predicted control steps is at or below threshold"
            ),
        },
        "state_horizons": list(state_horizons),
        "video_horizons": list(video_horizons),
        "video_metrics_exclude_known_r0": True,
        "variants": {},
    }
    for variant in variants:
        latent = (totals[variant]["latent"] / count).tolist()
        report["variants"][variant] = {
            **{f"latent_z{i + 1}": v for i, v in enumerate(latent)},
            "latent_mean": sum(latent) / len(latent),
            "latent_cosine_by_z": {
                f"z{i + 1}": value
                for i, value in enumerate((totals[variant]["latent_cosine"] / count).tolist())
            },
            "state": float(totals[variant]["state"] / count),
            "state_mse_by_horizon": {
                str(step): value
                for step, value in zip(
                    state_horizons,
                    (totals[variant]["state_mse_horizon"] / count).tolist(),
                )
            },
            "translation_ade": float(totals[variant]["translation_ade"] / count),
            "translation_by_horizon": {
                str(step): value
                for step, value in zip(
                    state_horizons,
                    (totals[variant]["translation_horizon"] / count).tolist(),
                )
            },
            "offline_success_proxy_by_horizon": {
                str(threshold): {
                    str(step): value
                    for step, value in zip(
                        state_horizons,
                        (totals[variant]["success_horizon"][threshold_index] / count).tolist(),
                    )
                }
                for threshold_index, threshold in enumerate(success_thresholds)
            },
            "offline_success_proxy_final_t32": {
                str(threshold): float(totals[variant]["success_final"][threshold_index] / count)
                for threshold_index, threshold in enumerate(success_thresholds)
            },
            "offline_success_proxy_all_32_steps": {
                str(threshold): float(totals[variant]["success_all32"][threshold_index] / count)
                for threshold_index, threshold in enumerate(success_thresholds)
            },
            "pixel_mse_future_only": float(totals[variant]["pixel"] / count),
            "lpips_future_mean": float(totals[variant]["lpips"] / count),
            "ssim_future_mean": float(totals[variant]["ssim"] / count),
            "psnr_future_mean_db": float(totals[variant]["psnr"] / count),
            "lpips_by_horizon": {
                str(step): value
                for step, value in zip(
                    video_horizons,
                    (totals[variant]["lpips_horizon"] / count).tolist(),
                )
            },
            "ssim_by_horizon": {
                str(step): value
                for step, value in zip(
                    video_horizons,
                    (totals[variant]["ssim_horizon"] / count).tolist(),
                )
            },
            "psnr_by_horizon_db": {
                str(step): value
                for step, value in zip(
                    video_horizons,
                    (totals[variant]["psnr_horizon"] / count).tolist(),
                )
            },
        }
    if args.carrier_availability_grid:
        real_name = "full_video_motion"
    elif args.video_availability_grid:
        real_name = "video_full_z0_z1_z2"
    elif args.availability_grid:
        real_name = "full_z0_z1_z2"
    elif args.cross_modal:
        real_name = "real_video_real_motion"
    else:
        real_name = "real"
    real = report["variants"][real_name]
    for variant in variants:
        if variant == real_name:
            continue
        other = report["variants"][variant]
        report["variants"][variant]["latent_delta_vs_real"] = (
            other["latent_mean"] - real["latent_mean"]
        )
        report["variants"][variant]["state_delta_vs_real"] = other["state"] - real["state"]
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
