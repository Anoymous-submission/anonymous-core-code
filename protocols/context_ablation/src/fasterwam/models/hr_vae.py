"""Frozen Wan2.2 VAE used as the H&R video tokenizer, FastWAM-style.

FastWAM does not encode frames one at a time. It pushes the whole ordered
segment through the VAE in a single pass, so the causal 3D convolutions apply
*both* compressions:

    [B, 3, T, H, W]  ->  [B, 48, 1 + (T-1)/4, H/16, W/16]

    z0 <- RGB frame 0
    z1 <- RGB frames 1..4
    z2 <- RGB frames 5..8
    ...

The temporal axis is causal: verified on this checkpoint, perturbing RGB frames
5..8 changes z2 only and leaves z0 and z1 bitwise identical. Encoding each frame
separately as a T=1 clip -- what this module used to do -- throws that temporal
compression away and keeps only the 16x spatial one.

The VAE is a fixed feature extractor: no trainable parameters, kept out of the
optimizer and out of DDP, never written to a run checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .wan22.helpers.io import load_state_dict
from .wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
from .wan22.wan_video_vae import WanVideoVAE38

# Properties of the released Wan2.2 checkpoint, not knobs.
VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_COMPRESSION = 16
VAE_TEMPORAL_COMPRESSION = 4


def latent_steps_for(num_rgb_frames: int) -> int:
    """1 + (T-1)/4, the VAE's temporal contract."""

    if num_rgb_frames < 1:
        raise ValueError("num_rgb_frames must be positive")
    if (num_rgb_frames - 1) % VAE_TEMPORAL_COMPRESSION:
        raise ValueError(
            f"num_rgb_frames must satisfy (T-1) % {VAE_TEMPORAL_COMPRESSION} == 0, "
            f"got T={num_rgb_frames}"
        )
    return 1 + (num_rgb_frames - 1) // VAE_TEMPORAL_COMPRESSION


class FrozenWan22VideoEncoder(nn.Module):
    """Encode ordered video segments to Wan2.2 latents and back."""

    # WanVideoVAE.encode/decode iterate the batch in Python, one clip per
    # forward, which left the GPU badly underutilized (measured 66-79% of step
    # time). Clips are independent, so they go through as one real batch, split
    # into groups of this many clips to bound peak memory. The *time* axis is
    # never split -- that would break the temporal compression.
    DEFAULT_CLIPS_PER_CALL = 16

    def __init__(
        self,
        vae: WanVideoVAE38,
        *,
        device: torch.device,
        dtype: torch.dtype,
        clips_per_call: int = DEFAULT_CLIPS_PER_CALL,
    ) -> None:
        super().__init__()
        self.vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        if clips_per_call <= 0:
            raise ValueError("clips_per_call must be positive")
        self.clips_per_call = int(clips_per_call)

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        clips_per_call: int = DEFAULT_CLIPS_PER_CALL,
    ) -> "FrozenWan22VideoEncoder":
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Wan2.2 VAE weights not found: {path}")
        state_dict = wan_video_vae_state_dict_converter(
            load_state_dict(str(path), torch_dtype=dtype, device="cpu")
        )
        vae = WanVideoVAE38()
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        # The VAE is the one pretrained component here; a partial load would
        # silently poison every latent.
        if missing or unexpected:
            raise RuntimeError(
                f"Wan2.2 VAE state dict mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected (first missing: {list(missing)[:3]})"
            )
        return cls(vae, device=device, dtype=dtype, clips_per_call=clips_per_call)

    @property
    def latent_channels(self) -> int:
        return int(self.vae.z_dim)

    def latent_size(self, height: int, width: int) -> tuple[int, int]:
        factor = int(self.vae.upsampling_factor)
        if height % factor or width % factor:
            raise ValueError(
                f"image size ({height}, {width}) must be divisible by the VAE "
                f"spatial compression factor {factor}"
            )
        return height // factor, width // factor

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """[B, 3, T, H, W] RGB in [-1, 1] -> [B, 48, 1+(T-1)/4, h, w].

        The whole segment goes through in one pass, so the causal temporal
        convolutions fold each run of 4 frames into one latent step.
        """

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B, 3, T, H, W], got {tuple(video.shape)}")
        expected = latent_steps_for(video.shape[2])
        video = video.to(device=self.device, dtype=self.dtype)
        outputs = []
        for start in range(0, video.shape[0], self.clips_per_call):
            chunk = video[start : start + self.clips_per_call]
            # Direct call into the inner module: same maths as
            # WanVideoVAE.encode, without its per-clip Python loop.
            outputs.append(self.vae.model.encode(chunk, self.vae.scale))
        latents = torch.cat(outputs, dim=0)
        if latents.shape[2] != expected:
            raise RuntimeError(
                f"expected {expected} latent steps for T={video.shape[2]}, "
                f"got {latents.shape[2]}"
            )
        return latents

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, 48, Tz, h, w] -> [B, 3, 4*Tz-3, H, W] RGB in [-1, 1]."""

        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, Tz, h, w], got {tuple(latents.shape)}")
        latents = latents.to(device=self.device, dtype=self.dtype)
        outputs = []
        for start in range(0, latents.shape[0], self.clips_per_call):
            chunk = latents[start : start + self.clips_per_call]
            outputs.append(self.vae.model.decode(chunk, self.vae.scale).clamp_(-1, 1))
        return torch.cat(outputs, dim=0)
