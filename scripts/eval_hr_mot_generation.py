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
from torch.utils.data import DataLoader

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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
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
    encoder = FrozenWan22VideoEncoder.from_pretrained(
        contract["vae"]["path"], device=device, dtype=torch.bfloat16
    )

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
    else:
        variants = ("real", "other_task", "none")
    totals = {
        v: {
            "latent": torch.zeros(config.num_horizons, dtype=torch.float64, device=device),
            "state": torch.zeros((), dtype=torch.float64, device=device),
            "pixel": torch.zeros((), dtype=torch.float64, device=device),
        }
        for v in variants
    }
    swapped_groups = 0
    count = 0.0

    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        robot = encoder.encode_video(batch["robot_video"].to(device)).float()
        human = encoder.encode_video(batch["human_video"].to(device)).float()
        robot = robot.permute(0, 2, 1, 3, 4).contiguous()
        human = human.permute(0, 2, 1, 3, 4).contiguous()
        target_latent = robot[:, 1:]
        target_state = batch["robot_future_state"].to(device, torch.float32)
        robot_now = batch["robot_current_state"].to(device, torch.float32)
        human_now = batch["human_future_action"].to(device, torch.float32)
        size = robot.shape[0]

        other_human = None
        other_human_now = None
        if not (
            args.availability_grid or args.carrier_availability_grid or args.video_availability_grid
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
            totals[variant]["state"] += (
                (state.float() - target_state).pow(2).mean((1, 2)).sum().double()
            )
            if args.decode_pixels:
                predicted = encoder.decode_video(
                    torch.cat((robot[:, :1], latents), dim=1).permute(0, 2, 1, 3, 4)
                ).float()
                truth = batch["robot_video"].to(device).float()
                totals[variant]["pixel"] += future_pixel_mse(predicted, truth).sum().double()
        count += size

    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "protocol": "generation from pure Gaussian noise, full flow integration",
        "inference_steps": args.inference_steps,
        "generation_seed": args.generation_seed,
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
            )
            else int(swapped_groups)
        ),
        "all_other_task_samples_have_different_instruction": (
            None
            if (
                args.availability_grid
                or args.carrier_availability_grid
                or args.video_availability_grid
            )
            else swapped_groups == count
        ),
        "none_context_keeps_human_z0": not config.droppable_human_current,
        "variants": {},
    }
    for variant in variants:
        latent = (totals[variant]["latent"] / count).tolist()
        report["variants"][variant] = {
            **{f"latent_z{i + 1}": v for i, v in enumerate(latent)},
            "latent_mean": sum(latent) / len(latent),
            "state": float(totals[variant]["state"] / count),
            **({"pixel": float(totals[variant]["pixel"] / count)} if args.decode_pixels else {}),
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
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
