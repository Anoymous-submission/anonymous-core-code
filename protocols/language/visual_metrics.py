"""Future-only physical-state and RGB metrics; no task-success surrogate.

Bridge author transformation_utils.state2transform uses Rz(yaw)Ry(pitch)Rx(roll)
times a fixed default rotation. That common right factor cancels in SO(3)
geodesic error. Angles are radians, translation meters, gripper native units.
"""

import math

import torch
import torch.nn.functional as F


def future_frames(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 5 or predicted.shape[1:3] != (3, 9):
        raise ValueError("Expected matching [B,3,9,H,W] complete decoded clips")
    # Never score the known current frame, even if its reconstruction is bad.
    return (
        predicted[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1),
        target[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1),
    )


def rotation_matrices(euler):
    x, y, z = euler.double().unbind(-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    return torch.stack(
        (
            cy * cz,
            sx * sy * cz - cx * sz,
            cx * sy * cz + sx * sz,
            cy * sz,
            sx * sy * sz + cx * cz,
            cx * sy * sz - sx * cz,
            -sy,
            sx * cy,
            cx * cy,
        ),
        -1,
    ).reshape(*euler.shape[:-1], 3, 3)


def state_scores(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 7:
        raise ValueError("State must be matching [B,H,7], no current timestep")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite state prediction/target")
    translation = (predicted[..., :3].double() - target[..., :3].double()).norm(dim=-1) * 1000
    a, b = rotation_matrices(predicted[..., 3:6]), rotation_matrices(target[..., 3:6])
    relative = a.transpose(-2, -1) @ b
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    # atan2 formulation avoids acos instability near identity while retaining pi.
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = skew.norm(dim=-1) / 2
    rotation = torch.atan2(sine, cosine) * 180 / math.pi
    gripper = (predicted[..., 6].double() - target[..., 6].double()).abs()
    return dict(
        translation_mm=translation,
        rotation_deg=rotation,
        gripper_abs=gripper,
        ade_mm=translation.mean(-1),
        fde_mm=translation[:, -1],
    )


def gaussian_ssim(x, y):
    """Per-frame SSIM, 11x11 Gaussian sigma1.5, valid border, population moments.

    RGB float in[0,1]; channel scores averaged. No skimage dependency or padding.
    """
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 11:
        raise ValueError("Invalid RGB shape for SSIM")
    # Local variance subtracts similar moments. Float32/TF32 convolution on
    # channels-last video tensors caused ~0.009 SSIM bias in the real-batch
    # witness despite passing CPU/random-image tests. Use float64 moments and
    # canonical layout; metric precision must not depend on training kernels.
    x, y = x.double().contiguous(), y.double().contiguous()
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-axis.square() / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, 11, 11).contiguous()
    conv = lambda t: F.conv2d(t, kernel, groups=x.shape[1])
    mx, my = conv(x), conv(y)
    vx = (conv(x * x) - mx * mx).clamp_min(0)
    vy = (conv(y * y) - my * my).clamp_min(0)
    cov = conv(x * y) - mx * my
    value = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
    )
    return value.flatten(1).mean(1)


@torch.no_grad()
def visual_scores(x, y, lpips_model):
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite RGB")
    if x.min() < 0 or y.min() < 0 or x.max() > 1 or y.max() > 1:
        raise ValueError("RGB outside [0,1]")
    with (
        torch.autocast(device_type=x.device.type, enabled=False),
        torch.backends.cudnn.flags(allow_tf32=False),
    ):
        x, y = x.float().contiguous(), y.float().contiguous()
        mse = (x.double() - y.double()).square().flatten(1).mean(1)
        # Zero-error frames explicitly counted; no silent epsilon cap on PSNR.
        psnr = -10 * torch.log10(mse)
        return dict(
            pixel_mse=mse,
            psnr_db=psnr,
            ssim=gaussian_ssim(x, y),
            lpips=lpips_model(x * 2 - 1, y * 2 - 1).flatten(),
        )
