"""End-to-end H&R inference: RGB in, RGB + real-world state out.

`HRMoTFlowModel.generate` is deliberately latent-only -- the frozen VAE lives
outside the trainable model so DDP never syncs its 705M frozen parameters and
checkpoints stay DiT-only. That leaves two steps that have to happen somewhere:
decoding predicted latents back to frames, and undoing the per-dimension
normalization on the predicted robot state. This module is that layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..datasets.hr_raw_stream import StateStats
from .hr_mot import HRMoTFlowModel
from .hr_vae import FrozenWan22VideoEncoder


@dataclass
class HRRollout:
    """One generated rollout, in the units a caller actually wants."""

    robot_future_frames: torch.Tensor  # [B, 3, T-1, H, W] future RGB only, in [-1, 1]
    robot_future_state: torch.Tensor  # [B, T-1, 7] end_position(6)+gripper(1), raw units
    robot_future_latents: torch.Tensor  # [B, Kz, C, h, w] pre-decode, for diagnostics
    human_steps_used: torch.Tensor  # [B] human latent steps that conditioned each sample


def denormalize_state(state: torch.Tensor, stats: StateStats) -> torch.Tensor:
    """Undo (x - mean) / std applied by the dataset, back to raw robot units."""

    mean = torch.as_tensor(stats.robot_mean, dtype=state.dtype, device=state.device)
    std = torch.as_tensor(stats.robot_std, dtype=state.dtype, device=state.device)
    return state * std + mean


def build_human_mask(
    num_human_steps: int,
    *,
    batch: int,
    keep: Optional[Sequence[int]] = None,
    pin_current: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """[B, Tz] mask selecting which human latent steps to condition on.

    `keep` lists step indices: 0 is the human z0 step (RGB frame 0), 1 covers
    RGB frames 1-4, 2 covers 5-8, and so on. `None` keeps all of them. An explicit
    list is exact, so `keep=[]` means no human context for models trained with a
    droppable current step. `pin_current=True` preserves the contract of legacy
    models trained with ``droppable_human_current=False``.
    """

    mask = torch.zeros((batch, num_human_steps), dtype=torch.bool, device=device)
    if keep is None:
        mask[:] = True
        return mask
    for slot in keep:
        if not 0 <= int(slot) < num_human_steps:
            raise ValueError(f"human latent step {slot} out of range [0, {num_human_steps})")
        mask[:, int(slot)] = True
    if pin_current:
        mask[:, 0] = True
    return mask


@torch.no_grad()
def generate_rollout(
    model: HRMoTFlowModel,
    encoder: FrozenWan22VideoEncoder,
    stats: StateStats,
    *,
    human_video: torch.Tensor,
    robot_current_frame: Optional[torch.Tensor] = None,
    robot_video: Optional[torch.Tensor] = None,
    robot_current_state: Optional[torch.Tensor] = None,
    human_future_action: Optional[torch.Tensor] = None,
    keep_human_steps: Optional[Sequence[int]] = None,
    num_inference_steps: int = 20,
    sigma_shift: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    decode: bool = True,
) -> HRRollout:
    """Encode -> joint flow generation -> VAE decode + state un-normalization.

    Args:
        robot_current_frame: the only robot image available online, either
            [B, 3, H, W] or [B, 3, 1, H, W], in [-1, 1].
        robot_video: deprecated compatibility input. It may contain one or many
            frames, but only frame 0 is encoded; future robot frames are never
            required or observed.
        human_video: [B, 3, T, H, W] ordered ground-truth human segment.
        robot_current_state: [B, 7] normalized robot state at t0. Required when
            the model was trained with use_state_context.
        human_future_action: [B, action_horizon, 84] normalized raw human wrist
            frame and articulated hand coordinates over the predicted steps.
        keep_human_steps: exact human latent steps to condition on. Defaults to
            all. An empty sequence supplies no human context when the model was
            trained with a droppable current step.
    """

    config = model.config
    if config.use_state_context and (robot_current_state is None or human_future_action is None):
        raise ValueError(
            "this model was trained with use_state_context; pass "
            "robot_current_state [B, 7] and human_future_action "
            f"[B, {config.num_action_tokens}, {config.human_action_dim}]"
        )
    if (robot_current_frame is None) == (robot_video is None):
        raise ValueError("pass exactly one of robot_current_frame or robot_video")
    if robot_current_frame is None:
        if robot_video is None or robot_video.ndim != 5 or robot_video.shape[1] != 3:
            shape = None if robot_video is None else tuple(robot_video.shape)
            raise ValueError(f"robot_video must be [B, 3, T, H, W], got {shape}")
        robot_current_frame = robot_video[:, :, :1]
    elif robot_current_frame.ndim == 4 and robot_current_frame.shape[1] == 3:
        robot_current_frame = robot_current_frame.unsqueeze(2)
    elif not (
        robot_current_frame.ndim == 5
        and robot_current_frame.shape[1] == 3
        and robot_current_frame.shape[2] == 1
    ):
        raise ValueError(
            "robot_current_frame must be [B, 3, H, W] or [B, 3, 1, H, W], got "
            f"{tuple(robot_current_frame.shape)}"
        )
    if human_video.ndim != 5 or human_video.shape[1] != 3:
        raise ValueError(f"human_video must be [B, 3, T, H, W], got {tuple(human_video.shape)}")
    if human_video.shape[2] != config.num_rgb_frames:
        raise ValueError(
            f"human_video must carry {config.num_rgb_frames} frames, " f"got {human_video.shape[2]}"
        )
    if robot_current_frame.shape[0] != human_video.shape[0]:
        raise ValueError("robot and human batch sizes must match")

    batch = robot_current_frame.shape[0]
    if robot_current_state is not None and robot_current_state.shape != (
        batch,
        config.action_dim,
    ):
        raise ValueError(
            f"robot_current_state must be [B, {config.action_dim}], got "
            f"{tuple(robot_current_state.shape)}"
        )
    if human_future_action is not None and human_future_action.shape != (
        batch,
        config.num_action_tokens,
        config.human_action_dim,
    ):
        raise ValueError(
            "human_future_action must be "
            f"[B, {config.num_action_tokens}, {config.human_action_dim}], got "
            f"{tuple(human_future_action.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        # Online inference observes and encodes R0 only. Human ground truth still
        # goes through one temporal VAE pass so its latent-step contract is kept.
        robot = encoder.encode_video(robot_current_frame).float().permute(0, 2, 1, 3, 4)
        human = encoder.encode_video(human_video).float().permute(0, 2, 1, 3, 4)
        robot_current_latent = robot[:, :1].contiguous()
        human_latents = human.contiguous()
        # Dataset callers naturally provide CPU FP32 state tensors while the VAE
        # and MoT live on CUDA. Images are moved inside the encoder; state must be
        # moved explicitly or the first action-side Linear raises a device error.
        state_device = robot_current_latent.device
        robot_current_state = (
            None
            if robot_current_state is None
            else robot_current_state.to(device=state_device, dtype=torch.float32)
        )
        human_future_action = (
            None
            if human_future_action is None
            else human_future_action.to(device=state_device, dtype=torch.float32)
        )
        human_mask = build_human_mask(
            config.num_latent_steps,
            batch=robot_current_frame.shape[0],
            keep=keep_human_steps,
            pin_current=not config.droppable_human_current,
            device=robot_current_latent.device,
        )
        latents, state = model.generate(
            robot_current_latent=robot_current_latent,
            human_latents=human_latents,
            robot_current_state=robot_current_state,
            human_future_action=human_future_action,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            generator=generator,
            human_mask=human_mask,
        )
        # The causal decoder needs clean z0 plus generated z1.., but the public
        # field is explicitly future-only and must not masquerade R0 as a target.
        full = torch.cat((robot_current_latent, latents), dim=1)
        frames = (
            encoder.decode_video(full.permute(0, 2, 1, 3, 4))[:, :, 1:]
            if decode
            else torch.empty(0, device=latents.device)
        )
        return HRRollout(
            robot_future_frames=frames,
            robot_future_state=denormalize_state(state.float(), stats),
            robot_future_latents=latents,
            human_steps_used=human_mask.sum(dim=1),
        )
    finally:
        if was_training:
            model.train()
