"""Train the ~0.6B H&R flow model from scratch or strict model-only initialization.

Frames are encoded by the frozen Wan2.2 VAE (the one pretrained component; it is
a fixed tokenizer, holds no trainable parameters, and is never checkpointed). The
MoT defaults to random initialization; --init-checkpoint starts a new stage
from all model tensors, with a fresh optimizer, scheduler, and sampler.

Deliberate departures from the previous, audit-failed trainer:
  * Parameters, gradients and optimizer moments stay FP32; compute runs under
    bf16 autocast. The old run kept everything in BF16, which quantized away
    small updates and left all 86,400 RMSNorm scales at exactly 1.0.
  * Held-out episodes with a fixed-noise, fixed-timestep, per-horizon evaluation.
    The old run reported training-batch loss only and had zero held-out data.
  * Provenance (git commit, source hashes, argv, GPU model, versions) is written
    into every run directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fasterwam.datasets.hr_raw_stream import (
    ACTION_HORIZON,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    FRAME_STRIDE,
    HUMAN_ACTION_DIM,
    NUM_RGB_FRAMES,
    STATE_DIM,
    RawHRStreamingDataset,
    StateStats,
    compute_state_stats,
    discover_raw_v1_episodes,
)
from fasterwam.models.hr_mot import (
    HRMoTConfig,
    HRMoTFlowModel,
    count_parameters,
    parameter_breakdown,
)
from fasterwam.models.hr_eval import (
    HR_MOT_SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
    human_context_metadata,
)
from fasterwam.models.hr_vae import FrozenWan22VideoEncoder
from fasterwam.models.hr_initialization import load_initial_model, validate_initialization_data

DEFAULT_VAE_PATH = "checkpoints/Wan2.2_VAE_bf16.safetensors"
# Fixed sigmas for held-out evaluation. Averaging a few points across the flow
# schedule gives a number comparable step-to-step, unlike a random training draw.
EVAL_SIGMAS = (0.2, 0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ~0.6B H&R MoT flow model (FastWAM-style) from scratch."
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--annotations", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint-sha256")
    parser.add_argument("--vae-path", type=Path, default=Path(DEFAULT_VAE_PATH))

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=256,
        help="windows sampled per episode per epoch. len(dataset) = episodes x this; "
        "an episode holds ~len-8 distinct windows, so this sets how much of the "
        "window space one epoch covers.",
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="optional hard cap; 0 = run all epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="extra step-interval checkpoints; 0 = checkpoint once per epoch only",
    )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="holdout batch size. Must NOT reuse the training batch size: the "
        "holdout set is small, and a large batch with drop_last=True yields "
        "zero batches and silently produces no eval at all.",
    )
    parser.add_argument(
        "--eval-windows-per-episode",
        type=int,
        default=4,
        help="windows sampled per holdout episode, so the eval set is not one "
        "window per episode",
    )
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help="retain only the N most recent checkpoints (0 = keep all). A 0.6B "
        "checkpoint carries FP32 weights plus two Adam moments (~7 GB), so "
        "keeping every save costs ~70 GB for a 50k-step run.",
    )

    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--num-rgb-frames", type=int, default=NUM_RGB_FRAMES)
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument(
        "--no-state-context",
        action="store_true",
        help="drop the clean action-side anchors (robot state at t0 + human hand "
        "state); reproduces the anchorless baseline",
    )
    parser.add_argument(
        "--no-human-motion-context",
        action="store_true",
        help="hard-mask all 84-D human-motion tokens while retaining human video "
        "and the robot-current state anchor; model size is unchanged",
    )
    parser.add_argument(
        "--no-human-video-context",
        action="store_true",
        help="hard-mask all human video latent slots without changing model size",
    )
    parser.add_argument(
        "--no-robot-state-context",
        action="store_true",
        help="hard-mask the robot-current state slot without changing model size",
    )
    parser.add_argument(
        "--direct-fastwam-robot-only",
        action="store_true",
        help=(
            "use the native robot-only FastWAM sequence and visibility graph: "
            "physically omit every human video/motion slot and projection while "
            "retaining robot z0 and the robot-current 7-D proprioceptive anchor"
        ),
    )
    parser.add_argument(
        "--pin-human-current",
        action="store_true",
        help="never drop the human z0 step; leaves the no-human path untrained",
    )
    parser.add_argument(
        "--context",
        choices=("aligned", "cross_episode"),
        default="aligned",
        help="aligned: human stream from the same episode at the same t0, which "
        "is how the two cameras are recorded. cross_episode: ablation using "
        "another episode of the same instruction (no index correspondence).",
    )

    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--num-heads", type=int, default=10)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--video-hidden-dim", type=int, default=1280)
    parser.add_argument("--video-ffn-dim", type=int, default=5120)
    parser.add_argument("--action-hidden-dim", type=int, default=640)
    parser.add_argument("--action-ffn-dim", type=int, default=2560)
    parser.add_argument(
        "--full-human-context-prob",
        type=float,
        default=0.9,
        help="fraction of samples trained with all human latent steps; the "
        "remainder is the context-dropout regularizer (default 10%%)",
    )
    parser.add_argument("--loss-lambda-video", type=float, default=1.0)
    parser.add_argument(
        "--none-human-context-prob",
        type=float,
        default=None,
        help="explicit probability of dropping all human context; partial patterns share the remaining mass",
    )
    parser.add_argument("--loss-lambda-action", type=float, default=1.0)
    parser.add_argument(
        "--mixed-attention-type",
        choices=("softmax", "linear"),
        default="softmax",
        help="mixed MoT self-attention kernel; linear uses normalized ELU+1 "
        "attention while preserving the H&R condition/prediction mask",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--print-model-only", action="store_true")
    parser.add_argument("--disable-future-coupling", action="store_true")
    return parser.parse_args()


def build_model_config(args: argparse.Namespace, latent_h: int, latent_w: int) -> HRMoTConfig:
    direct = bool(args.direct_fastwam_robot_only)
    return HRMoTConfig(
        latent_height=latent_h,
        latent_width=latent_w,
        num_rgb_frames=args.num_rgb_frames,
        frame_stride=args.frame_stride,
        action_horizon=args.action_horizon,
        use_state_context=not args.no_state_context,
        use_robot_state_context=not args.no_robot_state_context,
        use_human_motion_context=(not args.no_human_motion_context) and not direct,
        use_human_video_context=(not args.no_human_video_context) and not direct,
        direct_fastwam_robot_only=direct,
        future_video_action_coupling=not args.disable_future_coupling,
        droppable_human_current=not args.pin_human_current,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        video_hidden_dim=args.video_hidden_dim,
        video_ffn_dim=args.video_ffn_dim,
        action_hidden_dim=args.action_hidden_dim,
        action_ffn_dim=args.action_ffn_dim,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        full_human_context_prob=args.full_human_context_prob,
        none_human_context_prob=args.none_human_context_prob,
        loss_lambda_video=args.loss_lambda_video,
        loss_lambda_action=args.loss_lambda_action,
        mixed_attention_type=args.mixed_attention_type,
    )


def holdout_split_size(num_episodes: int, requested: int) -> int:
    """Single definition of the holdout size, imported by the eval scripts.

    Duplicating `order[:requested]` in the eval scripts silently scores training
    episodes whenever `requested` exceeds this cap.
    """

    return min(int(requested), max(num_episodes // 10, 0))


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("H&R MoT training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    # default=str so Paths (including lists of them, e.g. --annotations) survive.
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def reduce_mean(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return values


def learning_rate_multiplier(
    step: int, *, total_steps: int, warmup_steps: int, min_ratio: float
) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


# Size comes from the StateStats contract, never a hard-coded shape: the stats
# vector contains robot-state and raw-human-motion mean/std. Deriving this
# avoids silently allocating the wrong distributed broadcast buffer.
STATS_VECTOR_SIZE = 2 * STATE_DIM + 2 * HUMAN_ACTION_DIM


def broadcast_stats(
    stats: StateStats | None, *, device: torch.device, world_size: int
) -> StateStats:
    if stats is None:
        array = torch.empty(STATS_VECTOR_SIZE, device=device, dtype=torch.float32)
    else:
        array = torch.as_tensor(stats.as_array(), device=device, dtype=torch.float32)
        if array.numel() != STATS_VECTOR_SIZE:
            raise RuntimeError(
                f"stats vector is {array.numel()} floats, expected {STATS_VECTOR_SIZE}"
            )
    if world_size > 1:
        dist.broadcast(array, src=0)
    return StateStats.from_array(array.cpu().numpy())


def source_provenance(project_root: Path) -> dict[str, object]:
    """Record exactly which code produced a run. The old runs recorded none."""

    tracked = [
        "src/fasterwam/models/hr_mot.py",
        "src/fasterwam/models/hr_vae.py",
        "src/fasterwam/models/hr_eval.py",
        "src/fasterwam/models/hr_inference.py",
        "src/fasterwam/models/hr_initialization.py",
        "src/fasterwam/models/wan22/mot.py",
        "src/fasterwam/models/wan22/wan_video_vae.py",
        "src/fasterwam/models/wan22/wan_video_dit.py",
        "src/fasterwam/models/wan22/schedulers/scheduler_continuous.py",
        "src/fasterwam/datasets/hr_raw_stream.py",
        "scripts/train_hr_mot_600m.py",
    ]
    hashes: dict[str, str] = {}
    for relative in tracked:
        path = project_root / relative
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    try:
        commit = (
            subprocess.run(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or None
        )
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except Exception:
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256_16": hashes,
        "argv": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count_visible": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": "not-recorded",
    }


def evaluation_contract(
    *,
    args: argparse.Namespace,
    config: HRMoTConfig,
    stats: StateStats,
    inventory_records,
    train_records,
    eval_records,
    source: dict[str, object],
) -> dict[str, object]:
    """Immutable inputs required to reproduce a checkpoint evaluation."""

    annotations = [{"path": str(path), "sha256": file_sha256(path)} for path in args.annotations]
    vae_path = Path(args.vae_path)
    return {
        "schema_version": HR_MOT_SCHEMA_VERSION,
        "model": config.to_dict(),
        "state_stats": stats.to_dict(),
        "workspace_root": str(args.workspace_root),
        "data_root": str(args.data_root),
        "annotations": annotations,
        "vae": {
            "path": str(vae_path),
            "size_bytes": vae_path.stat().st_size,
            "sha256": file_sha256(vae_path),
        },
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "seed": int(args.seed),
        "context": args.context,
        "eval_windows_per_episode": int(args.eval_windows_per_episode),
        "inventory_record_ids": [str(record.annotation_path) for record in inventory_records],
        "inventory_files": [
            {
                "id": str(record.annotation_path),
                "path": str(record.path),
                "size_bytes": record.path.stat().st_size,
                "mtime_ns": record.path.stat().st_mtime_ns,
            }
            for record in inventory_records
        ],
        "inventory_fingerprint": "path+size_bytes+mtime_ns",
        "train_record_ids": [str(record.annotation_path) for record in train_records],
        "holdout_record_ids": [str(record.annotation_path) for record in eval_records],
        "source_provenance": source,
    }


def encode_batch(
    encoder: FrozenWan22VideoEncoder,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_human_video_context: bool = True,
    use_human_motion_context: bool = True,
    use_robot_state_context: bool = True,
) -> dict[str, torch.Tensor]:
    """RGB segments -> frozen VAE temporal latents, one whole pass per stream."""

    # One VAE pass per stream over the whole ordered segment, so the causal
    # temporal convolutions fold each run of 4 RGB frames into one latent step.
    robot = encoder.encode_video(batch["robot_video"].to(device=device, non_blocking=True)).float()
    if use_human_video_context:
        human = encoder.encode_video(
            batch["human_video"].to(device=device, non_blocking=True)
        ).float()
    else:
        # Do not even send human pixels through the frozen VAE in the strict
        # robot-video-only control.  Slots remain as zero tensors downstream.
        human = torch.zeros_like(robot)
    # [B, C, Tz, h, w] -> [B, Tz, C, h, w]
    robot = robot.permute(0, 2, 1, 3, 4).contiguous()
    human = human.permute(0, 2, 1, 3, 4).contiguous()
    return {
        # robot z0 is restored as the clean condition, as in FastWAM
        "robot_current_latent": robot[:, :1],
        "human_latents": human,
        "robot_future_latents": robot[:, 1:],
        "robot_future_state": batch["robot_future_state"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "robot_current_state": (
            batch["robot_current_state"].to(device=device, dtype=torch.float32, non_blocking=True)
            if use_robot_state_context
            else torch.zeros(
                batch["robot_current_state"].shape,
                device=device,
                dtype=torch.float32,
            )
        ),
        "human_future_action": (
            batch["human_future_action"].to(device=device, dtype=torch.float32, non_blocking=True)
            if use_human_motion_context
            else torch.zeros(
                batch["human_future_action"].shape,
                device=device,
                dtype=torch.float32,
            )
        ),
    }


@torch.no_grad()
def evaluate(
    model: HRMoTFlowModel,
    loader: DataLoader,
    encoder: FrozenWan22VideoEncoder,
    *,
    device: torch.device,
    max_batches: int,
    step_labels: tuple[str, ...],
) -> dict[str, float]:
    """Fixed-noise, fixed-sigma, per-horizon held-out loss.

    Random training draws cannot be compared across steps; this can.
    """

    model.eval()
    num_horizons = len(step_labels)
    video_totals = torch.zeros(num_horizons, dtype=torch.float64, device=device)
    state_totals = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        encoded = encode_batch(
            encoder,
            batch,
            device,
            use_human_video_context=model.config.use_human_video_context,
            use_human_motion_context=model.config.use_human_motion_context,
            use_robot_state_context=model.config.use_robot_state_context,
        )
        robot_future = encoded["robot_future_latents"]
        robot_state = encoded["robot_future_state"]
        batch_size = robot_future.shape[0]
        # Deterministic noise: same realization for the same batch at every step.
        generator = torch.Generator(device=device).manual_seed(1234 + index)
        noise_video = torch.randn(
            robot_future.shape, device=device, dtype=robot_future.dtype, generator=generator
        )
        noise_state = torch.randn(
            robot_state.shape, device=device, dtype=robot_state.dtype, generator=generator
        )
        for sigma in EVAL_SIGMAS:
            timestep = torch.full(
                (batch_size,),
                sigma * model.config.num_train_timesteps,
                device=device,
                dtype=robot_future.dtype,
            )
            noisy_video = model.train_video_scheduler.add_noise(robot_future, noise_video, timestep)
            noisy_state = model.train_action_scheduler.add_noise(robot_state, noise_state, timestep)
            target_video = noise_video - robot_future
            target_state = noise_state - robot_state
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_video, pred_state = model.predict_velocity(
                    robot_current_latent=encoded["robot_current_latent"],
                    human_latents=encoded["human_latents"],
                    noisy_robot_future_latents=noisy_video,
                    noisy_robot_future_action=noisy_state,
                    timestep_video=timestep,
                    timestep_action=timestep,
                    robot_current_state=encoded["robot_current_state"],
                    human_future_action=encoded["human_future_action"],
                )
            # Per-horizon means, so a horizon that never learns cannot hide behind
            # the others.
            video_totals += (
                (pred_video.float() - target_video.float())
                .pow(2)
                .flatten(2)
                .mean(dim=2)
                .sum(dim=0)
                .double()
            )
            state_totals += (
                (pred_state.float() - target_state.float()).pow(2).mean(dim=(1, 2)).sum().double()
            )
            count += batch_size
    model.train()

    if dist.is_initialized():
        dist.all_reduce(video_totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(state_totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if float(count) == 0.0:
        # Never return silently: an empty holdout is a configuration fault, and
        # the last time it happened it cost a full 20-epoch run of blind training.
        raise RuntimeError(
            "holdout evaluation produced zero samples -- check eval batch size, "
            "drop_last, and the size of the holdout split"
        )
    metrics: dict[str, float] = {}
    for position, label in enumerate(step_labels):
        metrics[f"eval_video_{label}"] = float(video_totals[position] / count)
    metrics["eval_video_mean"] = float(video_totals.sum() / (count * num_horizons))
    metrics["eval_state_mean"] = float(state_totals.sum() / count)
    return metrics


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    if not args.print_model_only:
        missing = [
            name
            for name in ("workspace_root", "data_root", "output_dir")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(
                "missing required arguments: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )

    if args.print_model_only:
        config = build_model_config(args, DEFAULT_IMAGE_HEIGHT // 16, DEFAULT_IMAGE_WIDTH // 16)
        model = HRMoTFlowModel(config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str))
        print(json.dumps(parameter_breakdown(model), indent=2, sort_keys=True))
        print(
            f"sequence: {config.num_video_tokens + config.num_action_sequence} tokens\n"
            f"  video  {config.num_video_tokens:>4} "
            f"({config.num_condition_tokens} condition + "
            f"{config.num_video_tokens - config.num_condition_tokens} prediction)\n"
            f"  action {config.num_action_sequence:>4} "
            f"({config.num_action_condition_tokens} condition + "
            f"{config.num_action_tokens} prediction)\n"
            f"  {config.num_rgb_frames} frames @stride {config.frame_stride} "
            f"= {config.video_span} control steps, "
            f"{config.num_latent_steps} latent steps at {config.latent_step_frames}"
        )
        return

    rank, world_size, local_rank, device = setup_distributed()
    try:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        output_dir = args.output_dir
        run_id = uuid.uuid4().hex
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        records = discover_raw_v1_episodes(
            workspace_root=args.workspace_root,
            data_root=args.data_root,
            annotation_paths=args.annotations,
        )
        # Split BEFORE computing normalization statistics. Deriving mean/std from
        # all episodes leaks holdout information into every training sample.
        order = np.random.default_rng(args.seed).permutation(len(records))
        holdout_count = holdout_split_size(len(records), args.holdout_episodes)
        holdout_indices = set(int(x) for x in order[:holdout_count])
        train_records = [r for i, r in enumerate(records) if i not in holdout_indices]
        eval_records = [r for i, r in enumerate(records) if i in holdout_indices]

        stats = compute_state_stats(train_records) if rank == 0 else None
        stats = broadcast_stats(stats, device=device, world_size=world_size)

        dataset = RawHRStreamingDataset(
            train_records,
            stats=stats,
            num_rgb_frames=args.num_rgb_frames,
            frame_stride=args.frame_stride,
            action_horizon=args.action_horizon,
            image_height=args.image_height,
            image_width=args.image_width,
            seed=args.seed,
            windows_per_episode=args.windows_per_episode,
            context=args.context,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
        )

        eval_loader = None
        eval_dataset = None
        # NOT gated on eval_every: 0 means "no step-interval eval", and the
        # per-epoch evaluation must still run. Gating the loader here once made
        # --eval-every 0 silently disable holdout evaluation altogether.
        if eval_records:
            eval_dataset = RawHRStreamingDataset(
                eval_records,
                stats=stats,
                num_rgb_frames=args.num_rgb_frames,
                frame_stride=args.frame_stride,
                action_horizon=args.action_horizon,
                image_height=args.image_height,
                image_width=args.image_width,
                seed=args.seed + 1,
                context=args.context,
                windows_per_episode=args.eval_windows_per_episode,
                # Pin the window start and the context pairing so the holdout set
                # is the same batch at every evaluation; combined with the fixed
                # noise and sigmas in evaluate(), the curve is comparable.
                deterministic=True,
            )
            # drop_last=False on both: the holdout set is small, and dropping a
            # partial batch here can silently leave zero batches per rank.
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                sampler=DistributedSampler(
                    eval_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                ),
                num_workers=2,
                pin_memory=True,
                drop_last=False,
            )
            per_rank = len(eval_dataset) // world_size
            if per_rank < args.eval_batch_size and rank == 0:
                print(
                    f"[rank0] note: {per_rank} holdout samples per rank < "
                    f"eval batch {args.eval_batch_size}; batches will be partial",
                    flush=True,
                )

        # Frozen pretrained VAE: fixed tokenizer, no gradients, not in the optimizer.
        encoder = FrozenWan22VideoEncoder.from_pretrained(
            args.vae_path, device=device, dtype=torch.bfloat16
        )
        latent_h, latent_w = encoder.latent_size(args.image_height, args.image_width)
        config = build_model_config(args, latent_h, latent_w)

        # FP32 master weights: parameters, gradients and Adam moments all stay
        # FP32; only the compute is bf16, under autocast.
        with torch.device(device):
            raw_model = HRMoTFlowModel(config)
        initialization = None
        source_contract = None
        if args.init_checkpoint is not None:
            initialization, source_contract = load_initial_model(
                raw_model,
                args.init_checkpoint,
                args.init_checkpoint_sha256 or "",
                stats.to_dict(),
                verify_hash=(rank == 0),
            )
        elif args.init_checkpoint_sha256 is not None:
            raise ValueError("--init-checkpoint-sha256 requires --init-checkpoint")
        barrier()
        source_steps = 0 if initialization is None else initialization["source_cumulative_steps"]
        parameter_count = count_parameters(raw_model)
        model: torch.nn.Module = raw_model
        if world_size > 1:
            model = DistributedDataParallel(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )

        steps_per_epoch = max(len(dataset) // (args.batch_size * world_size), 1)
        total_steps = args.steps if args.steps else steps_per_epoch * args.epochs

        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: learning_rate_multiplier(
                step,
                total_steps=total_steps,
                warmup_steps=args.warmup_steps,
                min_ratio=args.min_learning_rate / args.learning_rate,
            ),
        )

        contract = None
        contract_hash = None
        if rank == 0:
            source = source_provenance(project_root)
            contract = evaluation_contract(
                args=args,
                config=config,
                stats=stats,
                inventory_records=records,
                train_records=dataset.records,
                eval_records=[] if eval_dataset is None else eval_dataset.records,
                source=source,
            )
            if source_contract is not None:
                validate_initialization_data(source_contract, contract)
            contract_hash = canonical_sha256(contract)
            configuration = {
                "schema_version": HR_MOT_SCHEMA_VERSION,
                "run_id": run_id,
                "evaluation_contract": contract,
                "evaluation_contract_sha256": contract_hash,
                "args": vars(args),
                "initialization": initialization,
                "schedule": {
                    "epochs": args.epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "source_training_steps": source_steps,
                    "cumulative_training_steps_planned": source_steps + total_steps,
                    "global_batch": args.batch_size * world_size,
                    "samples_per_epoch": len(dataset),
                },
                "model": config.to_dict(),
                "architecture": (
                    "direct_fastwam_robot_only"
                    if config.direct_fastwam_robot_only
                    else "hr_mot_human_carrier_sequence"
                ),
                "parameters": parameter_breakdown(raw_model),
                "sequence": {
                    "action_condition_tokens": config.num_action_condition_tokens,
                    "action_sequence": config.num_action_sequence,
                    "video_tokens": config.num_video_tokens,
                    "condition_tokens": config.num_condition_tokens,
                    "prediction_video_tokens": config.num_video_tokens
                    - config.num_condition_tokens,
                    "action_tokens": config.num_action_tokens,
                    "tokens_per_frame": config.tokens_per_frame,
                    "frame_roles": list(config.frame_roles),
                    "frame_rope_positions": list(config.frame_rope_positions),
                },
                "attention": (
                    {
                        "name": "native_fastwam_first_frame_causal",
                        "video_current_queries": "robot_z0_and_robot_current_state",
                        "video_future_queries": "all_robot_video_and_robot_current_state",
                        "robot_state_query": "robot_z0_and_self",
                        "action_future_queries": (
                            "robot_z0_robot_current_state_and_all_future_action_tokens"
                        ),
                        "video_action_future_coupling": False,
                        "layout": [
                            "robot_z0_latent_step",
                            "robot_z1_z2_noise_tokens",
                            "robot_current_state_condition",
                            "robot_future_state_noise_tokens",
                        ],
                    }
                    if config.direct_fastwam_robot_only
                    else {
                        "name": "hr_condition_prediction_factorial",
                        "video_action_future_coupling": config.future_video_action_coupling,
                        "condition_queries": "condition_only",
                        "prediction_queries": (
                            "all_condition_and_prediction_tokens"
                            if config.future_video_action_coupling
                            else "all_conditions_and_same_modality_predictions"
                        ),
                        "layout": [
                            "robot_z0_latent_step",
                            (
                                "human_ground_truth_latent_steps"
                                if config.use_human_video_context
                                else "human_video_latent_slots_hard_masked_out"
                            ),
                            "robot_future_latent_step_noise_tokens",
                            (
                                "robot_current_state_condition"
                                if config.use_robot_state_context
                                else "robot_current_state_slot_hard_masked_out"
                            ),
                            (
                                "human_hand_motion_ground_truth_condition"
                                if config.use_human_motion_context
                                else "human_hand_motion_slots_hard_masked_out"
                            ),
                            "robot_future_state_noise_tokens",
                        ],
                    }
                ),
                "vae": {
                    "path": str(args.vae_path),
                    "class": "WanVideoVAE38",
                    "latent_channels": encoder.latent_channels,
                    "latent_size": [latent_h, latent_w],
                    "temporal_compression": 4,
                    "encoding": "whole ordered segment in one pass (FastWAM-style)",
                    "frozen": True,
                    "trainable_parameters": 0,
                    "note": (
                        "pretrained Wan2.2 VAE used as a fixed frame tokenizer, "
                        "exactly as FastWAM does; the MoT is random-initialized"
                    ),
                },
                "data": {
                    "selection": (
                        "all discovered raw-v1 episodes; quality flags are recorded, "
                        "not filtered, per the full-data/no-cache experiment contract"
                    ),
                    "episodes_total": len(records),
                    "episodes_train": len(dataset.records),
                    "normalization_stats_from": "train split only",
                    "episodes_holdout": 0 if eval_dataset is None else len(eval_dataset.records),
                    "episodes_explicitly_non_clean": sum(
                        record.clean_behavior_eligible is False for record in records
                    ),
                    "episodes_needs_review": sum(record.needs_review for record in records),
                    "singleton_group_episodes_dropped": dataset.singleton_group_episodes,
                    "num_rgb_frames": args.num_rgb_frames,
                    "frame_stride": args.frame_stride,
                    "video_span_control_steps": config.video_span,
                    "action_horizon": args.action_horizon,
                    "human_action_source": (
                        None
                        if config.direct_fastwam_robot_only
                        else "HDF5 transformed_hand_frames(4x3) + " "transformed_hand_coords(24x3)"
                    ),
                    "human_action_dimension": (
                        None if config.direct_fastwam_robot_only else config.human_action_dim
                    ),
                    "human_action_control_steps": (
                        None if config.direct_fastwam_robot_only else [1, config.num_action_tokens]
                    ),
                    "human_action_context_enabled": config.use_human_motion_context,
                    "human_video_context_enabled": config.use_human_video_context,
                    "robot_state_context_enabled": config.use_robot_state_context,
                    "mixed_attention_type": config.mixed_attention_type,
                    "latent_steps_per_stream": config.num_latent_steps,
                    "windows_per_episode": args.windows_per_episode,
                    "image_size": [args.image_height, args.image_width],
                    "cache": None,
                    "vae_encoding_time": "online_per_training_batch",
                    "overflow": "repeat_last_frame",
                    "window_start": "uniform per item, capped at len - max_offset",
                    "context": args.context,
                    "context_note": (
                        "human_camera and robot_camera are the same episode, same "
                        "length, aligned frame by frame; aligned mode reads both at "
                        "the same t0"
                    ),
                    "holdout_sampling": "deterministic",
                },
                "state_stats": stats.to_dict(),
                "precision": {
                    "parameters": "fp32",
                    "optimizer_moments": "fp32",
                    "compute": "bf16_autocast",
                    "vae": "bf16",
                },
                "human_context": human_context_metadata(config),
                "provenance": source,
            }
            atomic_json(output_dir / "config.json", configuration)
            print(
                f"[rank0] {parameter_count:,} trainable parameters | "
                f"architecture={'direct FastWAM robot-only' if config.direct_fastwam_robot_only else 'H&R MoT'} | "
                f"{config.num_video_tokens + config.num_action_sequence} tokens/sample "
                f"({config.num_video_tokens} video + {config.num_action_sequence} action) | "
                f"{config.num_latent_steps} latent steps from {args.num_rgb_frames} frames "
                f"@stride {config.frame_stride} spanning {config.video_span} control steps | "
                f"human motion context={'on' if config.use_human_motion_context else 'off'} | "
                f"human video context={'on' if config.use_human_video_context else 'off'} | "
                f"robot state context={'on' if config.use_robot_state_context else 'off'} | "
                f"mixed attention={config.mixed_attention_type} | "
                f"{len(train_records)} train / {len(eval_records)} holdout episodes | "
                f"{steps_per_epoch:,} steps/epoch x {args.epochs} epochs = {total_steps:,} steps",
                flush=True,
            )

        metrics_path = output_dir / "metrics.jsonl"
        metrics_file = metrics_path.open("a") if rank == 0 else None
        start_time = time.perf_counter()
        last_metrics: dict[str, object] = {}
        step_labels = tuple(f"z{index + 1}" for index in range(config.num_horizons))

        def run_eval(step: int, epoch: int) -> None:
            if eval_loader is None:
                return
            eval_metrics = evaluate(
                raw_model,
                eval_loader,
                encoder,
                device=device,
                max_batches=args.eval_batches,
                step_labels=step_labels,
            )
            if rank == 0 and eval_metrics:
                row = {"step": step, "epoch": epoch, "split": "holdout", **eval_metrics}
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(row, sort_keys=True), flush=True)

        def save_checkpoint(step: int, epoch: int) -> None:
            if args.skip_checkpoint or rank != 0:
                return
            checkpoint = {
                "schema_version": HR_MOT_SCHEMA_VERSION,
                "run_id": run_id,
                "evaluation_contract_sha256": contract_hash,
                "step": step,
                "initialization": initialization,
                "cumulative_training_steps": source_steps + step,
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(),
                "state_stats": stats.to_dict(),
            }
            target = output_dir / f"epoch_{epoch:03d}_step_{step:07d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(target)
            if args.keep_last_checkpoints > 0:
                saved = sorted(output_dir.glob("epoch_*_step_*.pt"))
                for stale in saved[: -args.keep_last_checkpoints]:
                    stale.unlink()

        model.train()
        step = 0
        context_fraction_sum = torch.zeros(2, device=device)
        stop = False
        last_epoch_index = -1
        for epoch in range(args.epochs):
            last_epoch_index = epoch
            sampler.set_epoch(epoch)
            for batch in loader:
                encoded = encode_batch(
                    encoder,
                    batch,
                    device,
                    use_human_video_context=config.use_human_video_context,
                    use_human_motion_context=config.use_human_motion_context,
                    use_robot_state_context=config.use_robot_state_context,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Through the DDP wrapper, never model.module: DDP arms its
                    # gradient reducer inside its own forward().
                    loss, parts = model(
                        robot_current_latent=encoded["robot_current_latent"],
                        human_latents=encoded["human_latents"],
                        robot_future_latents=encoded["robot_future_latents"],
                        robot_future_action=encoded["robot_future_state"],
                        robot_current_state=encoded["robot_current_state"],
                        human_future_action=encoded["human_future_action"],
                    )
                loss.backward()
                context_fraction_sum += torch.stack(
                    (parts["human_full_fraction"], parts["human_none_fraction"])
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                step += 1

                if step % args.log_every == 0 or step == 1:
                    values = torch.stack(
                        (
                            loss.detach().float(),
                            parts["loss_video"].float(),
                            parts["loss_action"].float(),
                            parts["loss_video_unweighted"].float(),
                            parts["loss_action_unweighted"].float(),
                            grad_norm.detach().float(),
                            parts["human_frames_kept"].float(),
                            context_fraction_sum[0] / step,
                            context_fraction_sum[1] / step,
                        )
                    )
                    values = reduce_mean(values, world_size)
                    if rank == 0:
                        elapsed = time.perf_counter() - start_time
                        last_metrics = {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(values[0]),
                            "loss_video": float(values[1]),
                            "loss_state": float(values[2]),
                            "loss_video_unweighted": float(values[3]),
                            "loss_state_unweighted": float(values[4]),
                            "grad_norm": float(values[5]),
                            "human_steps_kept": float(values[6]),
                            "human_full_fraction_cumulative": float(values[7]),
                            "human_none_fraction_cumulative": float(values[8]),
                            "learning_rate": float(scheduler.get_last_lr()[0]),
                            "elapsed_seconds": elapsed,
                            "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                            "samples_per_second": step
                            * args.batch_size
                            * world_size
                            / max(elapsed, 1e-9),
                        }
                        metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
                        metrics_file.flush()
                        print(json.dumps(last_metrics, sort_keys=True), flush=True)

                if args.eval_every > 0 and step % args.eval_every == 0:
                    run_eval(step, epoch)
                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(step, epoch)
                    barrier()
                if args.steps and step >= args.steps:
                    stop = True
                    break

            # One checkpoint and one holdout evaluation per epoch.
            run_eval(step, epoch)
            save_checkpoint(step, epoch)
            barrier()
            if stop:
                break

        # evaluate() all-reduces, so every rank must enter it or the run deadlocks.
        final_eval: dict[str, float] = {}
        if eval_loader is not None:
            final_eval = evaluate(
                raw_model,
                eval_loader,
                encoder,
                device=device,
                max_batches=args.eval_batches,
                step_labels=step_labels,
            )
        if rank == 0:
            atomic_json(
                output_dir / "summary.json",
                {
                    "completed_steps": step,
                    "initialization": initialization,
                    "cumulative_training_steps": source_steps + step,
                    "completed_epochs": min(step // steps_per_epoch, args.epochs),
                    "epochs_started": last_epoch_index + 1,
                    "configured_epochs": args.epochs,
                    "stopped_by_step_cap": bool(stop),
                    "parameters": parameter_count,
                    "final_train": last_metrics,
                    "final_holdout": final_eval,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )
            metrics_file.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
