"""H&R Mixture-of-Transformers flow model: ~0.6B, FastWAM-style, trained from scratch.

Architecture follows FastWAM and departs from it in exactly one place.

Same as FastWAM:
  * The whole ordered segment goes through the frozen Wan2.2 48-channel VAE in
    one pass, so its causal 3D convolutions apply both compressions: 16x spatial
    and 4:1 temporal. [B,3,T,H,W] -> [B,48,1+(T-1)/4,h,w], with z0<-f0,
    z1<-f1..4, z2<-f5..8. The DiT never sees RGB.
  * Latents are patchified (1, 2, 2) into spatial tokens that survive all the way
    through the transformer -- there is no per-frame bottleneck.
  * Video and action are two separate experts (Mixture of Transformers): each owns
    its own weights, and they are joined by one mixed self-attention per layer.
  * Per-block AdaLN-Zero style modulation from a per-token timestep, plus RoPE
    (3D over frame/height/width for video, 1D over horizon for action).
  * Rectified-flow / flow-matching objective with the shift-based continuous
    scheduler: x_t = (1-s) x_0 + s eps, target eps - x_0.
  * Generation is the same joint iterative flow integration in latent + action
    space. `HRMoTFlowModel.generate` is the latent-space core; the VAE decode and
    state un-normalization live in `hr_inference.generate_rollout`, because the
    VAE is deliberately kept outside this module.

Different from FastWAM:
  * FastWAM passes human context through cross-attention. Here the ground-truth
    human frames are VAE-encoded and placed *in the self-attention sequence* as
    clean condition tokens, so the robot-future video and action tokens attend to
    them directly. There is no text encoder and no cross-attention at all.

Token layout for T=9 RGB frames per stream (3 latent steps, 2 predicted):

    video expert                                  covers RGB
      [0]      robot   z0   clean, t=0  condition  f0
      [1..3]   human   z0.. clean, t=0  condition  f0 / f1-4 / f5-8
      [4..5]   robot   z1.. NOISY, t=s  prediction f1-4 / f5-8
    action expert
      [0]      robot state t0                 clean condition
      [1..32]  human hand motion t1..t32      clean condition
      [33..64] robot states t1..t32 NOISY     prediction

Only the robot's future latent steps and states are denoised; the robot z0 latent
step is restored as clean condition, exactly as FastWAM does.

Attention visibility: condition tokens see condition tokens only; prediction
tokens (robot-future video and robot-future action) see everything.

Human-condition availability is an extra mask layer on top of that structure,
not a change to it. Robot z0 is always present; by default all three human
latent steps, including human z0, are maskable, giving 2^Tz patterns (8 for
Tz=3). Masking is whole-step: all `tokens_per_frame` tokens of a human latent
step are dropped together,
never a scattered subset of patches. A missing frame is excluded as a key/value
so nothing reads it, its own query row is diagonal-only and discarded, and its
    RoPE position is *not* renumbered -- the step covering f5-8 stays at position 32
even when the f1-4 step is absent. Supplied frames additionally carry a 0/1 availability embedding.
Training and inference build the mask through the same function, so at inference
any number of human frames can be supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Optional, Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .wan22.mot import MoT
from .wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan22.wan_video_dit import (
    GateModule,
    SelfAttention,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

ROLE_ROBOT_CURRENT = 0
ROLE_HUMAN_CONTEXT = 1
ROLE_ROBOT_FUTURE = 2
NUM_ROLES = 3


@dataclass(frozen=True)
class HRMoTConfig:
    """Shapes for the ~0.6B H&R MoT. Only trainable modules are counted."""

    # Latent geometry, fixed by the frozen Wan2.2 VAE and the chosen image size.
    latent_channels: int = 48
    latent_height: int = 14
    latent_width: int = 24
    patch_height: int = 2
    patch_width: int = 2

    # Ordered RGB segment length handed to the VAE, per stream. The Wan2.2 VAE
    # folds each run of 4 frames into one latent step, so T=9 -> 3 latent steps:
    # z0 <- f0, z1 <- f1..4, z2 <- f5..8. The robot's z0 is the clean current
    # condition; z1.. are what gets denoised.
    num_rgb_frames: int = 9
    vae_temporal_factor: int = 4

    # Control steps between consecutive video frames. FastWAM keeps every 4th of
    # 33 control steps as a video frame, so 9 frames span 32 steps (1.09 s at the
    # data's 29.4 fps). Reading 9 contiguous frames instead spans 8 steps = 0.27 s,
    # in which almost nothing happens: the gripper changes in 6.4% of 8-step
    # windows versus 25.4% of 32-step ones.
    frame_stride: int = 4

    # Contiguous robot action steps predicted per sample, covering the video span.
    action_horizon: int = 32

    action_dim: int = 7

    # Clean anchors for the action expert. Without them it predicts an absolute
    # end-effector pose (x spans 103..346 mm) with no idea where the arm is now
    # and no aligned human counterpart, having to infer both from pixels through
    # the shared attention. Measured on the anchorless baseline, removing the
    # human cost the action branch +362% -- far more than it cost video, the
    # signature of an unanchored pathway.
    #   robot_state_f0     [7]                where the arm is at t0
    #   human_action t1..tH [action_horizon, 84] raw wrist frame + hand coords
    # False reproduces the anchorless baseline.
    use_state_context: bool = True
    # These two switches preserve the action-condition slots and every module,
    # but hard-mask the corresponding sources.  They are therefore suitable for
    # same-size carrier ablations, unlike `use_state_context=False`.
    use_robot_state_context: bool = True
    # Keep the action-side robot-current anchor while hard-masking every mapped
    # human-motion token. This defines the same-size human-video-only control:
    # all modules and parameters remain present, but no 84-D human signal is a
    # readable key/value for any prediction query.
    use_human_motion_context: bool = True
    # Hard-mask every human video latent while retaining the slots, projections,
    # role parameters and sequence length for a strict robot-video-only control.
    use_human_video_context: bool = True
    human_action_dim: int = 84
    # Isolated public-data adapter. Default preserves the original H&R schema.
    # Bridge uses measured robot EE state, not hand motion or control commands.
    source_motion_kind: str = "human_hand84"
    include_source_current: bool = False
    # Optional clean word-token prefix in the state expert. These modules/slots
    # are retained in every matched Bridge variant, including no-language runs.
    num_text_tokens: int = 0
    text_vocab_size: int = 0
    use_language_context: bool = True
    text_hidden_dim: int = 128

    # Whether the human z0 latent step may also be dropped. With it pinned on, the
    # "no human context at all" path is never trained, its availability embedding
    # row never receives gradient, and any measurement of that condition is out of
    # distribution. Pinning it also lets the model lean entirely on the
    # same-moment cross-embodiment correspondence instead of reading the
    # demonstration. Droppable by default, as part of the same 10% regularizer.
    droppable_human_current: bool = True

    # Shared across experts -- MoT mixes attention, so head geometry must match.
    num_layers: int = 22
    num_heads: int = 10
    attn_head_dim: int = 128

    video_hidden_dim: int = 1280
    video_ffn_dim: int = 5120
    action_hidden_dim: int = 640
    action_ffn_dim: int = 2560

    freq_dim: int = 256
    max_rope_position: int = 1024
    eps: float = 1e-6

    gradient_checkpointing: bool = True

    # Human-condition availability. Robot z0 is never masked. Human z0 is
    # droppable by default together with both human-future latent steps, yielding
    # 2^3 = 8 patterns; --pin-human-current restores the legacy 4-pattern contract.
    #
    # This is a *regularizer*, not a curriculum: the point is that the model can
    # still be asked for fewer human steps at inference, not that it should train
    # mostly on degraded context. Dropping context on half the samples actively
    # weakens the human pathway, which is the pathway under study. Default is 0.9
    # full context, i.e. a 10% dropout rate.
    full_human_context_prob: float = 0.9

    # Flow-matching schedule (FastWAM defaults).
    video_train_shift: float = 5.0
    video_infer_shift: float = 5.0
    action_train_shift: float = 5.0
    action_infer_shift: float = 5.0
    num_train_timesteps: int = 1000

    # The video loss averages over ~250k scalars per sample and the action loss
    # over ~21. Summing the two raw means gives the video term a far smaller and
    # noisier per-element gradient, which is one of the failures the previous run
    # hit. These weights are the knob for that balance.
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    def validate(self) -> None:
        expected_source_dim = {"human_hand84": 84, "bridge_measured_state7": 7}.get(
            self.source_motion_kind
        )
        if expected_source_dim is None:
            raise ValueError(f"Unknown source motion schema: {self.source_motion_kind}")
        if self.num_text_tokens:
            if not self.use_state_context or self.text_vocab_size < 3:
                raise ValueError("Text requires clean state context and a real word vocabulary")
            if self.text_hidden_dim % 4:
                raise ValueError("Text hidden dimension must divide four heads")
        if self.use_state_context and self.human_action_dim != expected_source_dim:
            raise ValueError(
                f"{self.source_motion_kind} requires {expected_source_dim} dimensions; "
                f"got {self.human_action_dim}"
            )
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.action_horizon != self.video_span:
            raise ValueError(
                "action_horizon must equal the video span "
                f"(num_rgb_frames-1)*frame_stride = {self.video_span}, "
                f"got {self.action_horizon}"
            )
        if self.latent_height % self.patch_height or self.latent_width % self.patch_width:
            raise ValueError(
                f"latent size ({self.latent_height}, {self.latent_width}) must be divisible "
                f"by DiT patch ({self.patch_height}, {self.patch_width})"
            )
        if self.num_rgb_frames < 5 or (self.num_rgb_frames - 1) % self.vae_temporal_factor:
            raise ValueError(
                "num_rgb_frames must be >=5 with (T-1) % "
                f"{self.vae_temporal_factor} == 0, got {self.num_rgb_frames}"
            )
        # Guard the indices actually used, not the frame count: the video stream
        # indexes `frame_rope_positions` and the action stream indexes up to
        # num_action_tokens, either of which can exceed num_rgb_frames.
        highest = max((*self.frame_rope_positions, self.num_action_tokens))
        if highest >= self.max_rope_position:
            raise ValueError(
                f"RoPE position {highest} exceeds the precomputed table "
                f"({self.max_rope_position})"
            )
        # rope_apply consumes head_dim/2 complex frequencies; the 3D split must
        # add back up to exactly that or the rotation silently misaligns.
        head_dim = self.attn_head_dim
        f_dim = head_dim - 2 * (head_dim // 3)
        complex_total = (f_dim // 2) + 2 * ((head_dim // 3) // 2)
        if complex_total != head_dim // 2:
            raise ValueError(
                f"attn_head_dim={head_dim} does not split evenly into 3D RoPE "
                f"({complex_total} complex frequencies, need {head_dim // 2}); "
                "use a value such as 96 or 128"
            )
        for name, dim in (
            ("video_hidden_dim", self.video_hidden_dim),
            ("action_hidden_dim", self.action_hidden_dim),
        ):
            if dim <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")

    @property
    def num_latent_steps(self) -> int:
        """1 + (T-1)/4 per stream."""
        return 1 + (self.num_rgb_frames - 1) // self.vae_temporal_factor

    @property
    def num_horizons(self) -> int:
        """Predicted robot latent steps: everything after the clean z0."""
        return self.num_latent_steps - 1

    @property
    def video_span(self) -> int:
        """Control steps covered by the video window."""
        return (self.num_rgb_frames - 1) * self.frame_stride

    @property
    def num_action_tokens(self) -> int:
        """One robot state per predicted control step."""
        return self.action_horizon

    @property
    def latent_step_frames(self) -> tuple[int, ...]:
        """Last CONTROL STEP each latent step covers.

        The VAE folds 4 video frames into one latent step and each video frame is
        `frame_stride` control steps apart, so a latent step spans
        `4 * frame_stride` control steps: with stride 4 that is 0 / 16 / 32.
        Expressing this in control steps rather than video-frame index is what
        lets a predicted robot latent step share a RoPE position with the human
        latent step covering the same real time.
        """
        span = self.vae_temporal_factor * self.frame_stride
        return (0, *((index + 1) * span for index in range(self.num_latent_steps - 1)))

    @property
    def tokens_per_frame(self) -> int:
        return (self.latent_height // self.patch_height) * (self.latent_width // self.patch_width)

    @property
    def num_action_condition_tokens(self) -> int:
        """Robot state at t0 plus one human-motion token per control step."""
        return (
            (1 + self.num_source_motion_tokens + self.num_text_tokens)
            if self.use_state_context
            else 0
        )

    @property
    def num_source_motion_tokens(self) -> int:
        return self.num_action_tokens + int(self.include_source_current)

    @property
    def num_action_sequence(self) -> int:
        return self.num_action_condition_tokens + self.num_action_tokens

    @property
    def action_rope_positions(self) -> tuple[int, ...]:
        """Control-step index per action token, condition block first.

        A predicted state at step i shares its RoPE position with the human motion
        at step i, mirroring how a predicted robot latent step shares a position
        with the human latent step covering the same real time.
        """
        future = tuple(range(1, self.num_action_tokens + 1))
        if not self.use_state_context:
            return future
        source = (0, *future) if self.include_source_current else future
        return (0, *source, *range(self.num_text_tokens), *future)

    def human_action_latent_step(self) -> tuple[int, ...]:
        """Which human latent step each human-motion token follows for availability.

        Control steps 1..16 fall inside latent step 1, 17..32 inside step 2, so a
        masked human latent step masks the action tokens covering the same real
        time. Clamped in case the horizon ever overruns the video span.
        """
        span = self.vae_temporal_factor * self.frame_stride
        last = self.num_latent_steps - 1
        future = tuple(min(1 + (index // span), last) for index in range(self.num_action_tokens))
        return (0, *future) if self.include_source_current else future

    @property
    def num_condition_frames(self) -> int:
        """robot z0 + every human latent step."""
        return 1 + self.num_latent_steps

    @property
    def num_video_frames(self) -> int:
        """condition slots + predicted robot latent steps."""
        return self.num_condition_frames + self.num_horizons

    @property
    def num_condition_tokens(self) -> int:
        return self.num_condition_frames * self.tokens_per_frame

    @property
    def num_video_tokens(self) -> int:
        return self.num_video_frames * self.tokens_per_frame

    @property
    def frame_rope_positions(self) -> tuple[int, ...]:
        """Temporal RoPE index per latent-step slot, in RGB-frame units.

        A predicted robot latent step shares its temporal position with the human
        latent step covering the same frames; the role embedding is what separates
        them. That alignment is the point.
        """
        steps = self.latent_step_frames
        return (steps[0], *steps, *steps[1:])

    @property
    def frame_roles(self) -> tuple[int, ...]:
        return (
            ROLE_ROBOT_CURRENT,
            *([ROLE_HUMAN_CONTEXT] * self.num_latent_steps),
            *([ROLE_ROBOT_FUTURE] * self.num_horizons),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_condition_prediction_mask(
    *,
    num_condition_tokens: int,
    num_video_tokens: int,
    num_action_tokens: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Visibility over the concatenated [video, action] MoT sequence.

    Condition queries (robot current + human ground truth) see condition keys
    only, so the clean context never leaks information from the noisy targets.
    Prediction queries (robot future video + robot future action) see everything.
    """

    if not 0 < num_condition_tokens <= num_video_tokens:
        raise ValueError(
            f"num_condition_tokens={num_condition_tokens} must be in (0, {num_video_tokens}]"
        )
    if num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be positive")
    total = num_video_tokens + num_action_tokens
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:num_condition_tokens, :num_condition_tokens] = True
    mask[num_condition_tokens:, :] = True
    return mask


def build_availability_attention_mask(
    config: "HRMoTConfig",
    human_mask: torch.Tensor,
    *,
    human_motion_mask: torch.Tensor | None = None,
    text_valid: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-sample [B, 1, S, S] visibility over the concatenated [video, action] sequence.

    `human_mask` is [B, 1+K] with True = that human video step was
    supplied. `human_motion_mask` has the same shape and controls the mapped
    84-D motion tokens. It defaults to `human_mask`, preserving the joint-mask
    training contract and every existing caller.

    Rows follow the group-level contract:

        valid condition   (R0, H0, available H*)  -> valid condition only
        missing condition (masked-out H*)         -> itself only, output discarded
        prediction        (robot future + state)  -> valid condition + prediction

    Missing frames are excluded as keys entirely, so no query ever reads them.
    Their own rows are diagonal-only purely to keep the softmax well defined; a
    fully-False row would produce NaN.
    """

    tokens_per_frame = config.tokens_per_frame
    human_slots = config.num_latent_steps
    batch = human_mask.shape[0]
    if human_mask.shape != (batch, human_slots):
        raise ValueError(f"human_mask must be [B, {human_slots}], got {tuple(human_mask.shape)}")
    device = device if device is not None else human_mask.device
    human_mask = human_mask.to(device=device, dtype=torch.bool)
    if human_motion_mask is None:
        human_motion_mask = human_mask
    if human_motion_mask.shape != (batch, human_slots):
        raise ValueError(
            f"human_motion_mask must be [B, {human_slots}], got "
            f"{tuple(human_motion_mask.shape)}"
        )
    human_motion_mask = human_motion_mask.to(device=device, dtype=torch.bool)
    if not config.use_human_video_context:
        human_mask = torch.zeros_like(human_mask)
    if not config.use_human_motion_context:
        human_motion_mask = torch.zeros_like(human_motion_mask)

    # --- video side: robot z0, human z0.., then the predicted robot steps ------
    frame_available = torch.ones((batch, config.num_video_frames), dtype=torch.bool, device=device)
    frame_available[:, 1 : 1 + human_slots] = human_mask
    is_condition_frame = torch.zeros(config.num_video_frames, dtype=torch.bool, device=device)
    is_condition_frame[: config.num_condition_frames] = True

    valid = (frame_available & is_condition_frame[None, :]).repeat_interleave(
        tokens_per_frame, dim=1
    )
    missing = ((~frame_available) & is_condition_frame[None, :]).repeat_interleave(
        tokens_per_frame, dim=1
    )
    prediction = (
        (~is_condition_frame)[None, :].expand(batch, -1).repeat_interleave(tokens_per_frame, dim=1)
    )

    # --- action side: optional clean anchors, then the predicted states --------
    if config.use_state_context:
        # The robot anchor slot is retained in same-size ablations, but can be
        # made a missing (diagonal-only) key/value.
        robot_now = torch.full(
            (batch, 1),
            bool(config.use_robot_state_context),
            dtype=torch.bool,
            device=device,
        )
        # A human-motion token inherits the availability of the latent step whose
        # control steps it covers, so masking a human step masks its motion too.
        steps = torch.tensor(config.human_action_latent_step(), dtype=torch.long, device=device)
        human_now = human_motion_mask[:, steps]
        action_valid = torch.cat((robot_now, human_now), dim=1)
        action_missing = torch.cat((~robot_now, ~human_now), dim=1)
        if config.num_text_tokens:
            if text_valid is None or not config.use_language_context:
                text_valid = torch.zeros(
                    batch, config.num_text_tokens, device=device, dtype=torch.bool
                )
            if text_valid.shape != (batch, config.num_text_tokens):
                raise ValueError("Incorrect clean text validity shape")
            text_valid = text_valid.to(device=device, dtype=torch.bool)
            action_valid = torch.cat((action_valid, text_valid), dim=1)
            action_missing = torch.cat((action_missing, ~text_valid), dim=1)
    else:
        action_valid = torch.zeros((batch, 0), dtype=torch.bool, device=device)
        action_missing = torch.zeros((batch, 0), dtype=torch.bool, device=device)
    action_prediction = torch.ones(
        (batch, config.num_action_tokens), dtype=torch.bool, device=device
    )
    zeros_pred = torch.zeros_like(action_prediction)

    valid = torch.cat((valid, action_valid, zeros_pred), dim=1)
    missing = torch.cat((missing, action_missing, zeros_pred), dim=1)
    prediction = torch.cat((prediction, torch.zeros_like(action_valid), action_prediction), dim=1)

    total = valid.shape[1]
    mask = valid[:, :, None] & valid[:, None, :]
    mask |= prediction[:, :, None] & (valid | prediction)[:, None, :]
    mask |= missing[:, :, None] & torch.eye(total, dtype=torch.bool, device=device)[None]
    return mask.unsqueeze(1)


class HRDiTBlock(nn.Module):
    """FastWAM's DiT block minus cross-attention: all conditioning is in-sequence."""

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()


class HRHead(nn.Module):
    """Final AdaLN-modulated projection, driven by the same per-token timestep."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x, t: [B, S, hidden]
        modulation = self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device)
        shift, scale = (modulation + t.unsqueeze(2)).chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))


class _ExpertBase(nn.Module):
    """Shared timestep-modulation plumbing and the MoT-required attributes."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_heads: int,
        attn_head_dim: int,
        freq_dim: int,
        eps: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.freq_dim = freq_dim
        self.use_gradient_checkpointing = bool(gradient_checkpointing)

        self.blocks = nn.ModuleList(
            HRDiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self._rope_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def embed_timesteps(self, token_timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, S] per-token flow time -> (t [B, S, D], t_mod [B, S, 6, D])."""

        batch, seq_len = token_timesteps.shape
        embedded = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1).float())
        t = self.time_embedding(embedded).reshape(batch, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        return t, t_mod


class HRVideoExpert(_ExpertBase):
    """Video expert over per-frame Wan2.2 latents with 3D RoPE and role embeddings."""

    def __init__(self, config: HRMoTConfig) -> None:
        super().__init__(
            hidden_dim=config.video_hidden_dim,
            ffn_dim=config.video_ffn_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            attn_head_dim=config.attn_head_dim,
            freq_dim=config.freq_dim,
            eps=config.eps,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.config = config
        patch = (1, config.patch_height, config.patch_width)
        self.patch_size = patch
        self.patch_embedding = nn.Conv3d(
            config.latent_channels, config.video_hidden_dim, kernel_size=patch, stride=patch
        )
        self.role_embedding = nn.Parameter(torch.zeros(NUM_ROLES, config.video_hidden_dim))
        # Availability code read by the *prediction* tokens. It has to live here,
        # not on the human slots: under the hard mask nothing can read a missing
        # frame, so a marker placed on it would be permanently unreachable and
        # carry zero gradient. One embedding per (human slot, available?) pair,
        # summed, tells the predictor exactly which human frames were supplied.
        self.availability_embedding = nn.Parameter(
            torch.zeros(config.num_latent_steps, 2, config.video_hidden_dim)
        )
        self.head = HRHead(
            config.video_hidden_dim,
            config.latent_channels * config.patch_height * config.patch_width,
            config.eps,
        )
        # Kept off the module buffer registry: complex tensors do not survive DDP
        # buffer broadcast cleanly, and these are deterministic constants anyway.
        self._freqs_3d = precompute_freqs_cis_3d(config.attn_head_dim, end=config.max_rope_position)

    def rope_freqs(self, device: torch.device) -> torch.Tensor:
        """[num_video_tokens, 1, head_dim/2] complex, in frame-major token order."""

        key = (device, 0)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        config = self.config
        grid_h = config.latent_height // config.patch_height
        grid_w = config.latent_width // config.patch_width
        f_freqs, h_freqs, w_freqs = (x.to(device) for x in self._freqs_3d)
        positions = torch.tensor(config.frame_rope_positions, dtype=torch.long, device=device)
        num_frames = positions.numel()
        freqs = torch.cat(
            (
                f_freqs[positions]
                .view(num_frames, 1, 1, -1)
                .expand(num_frames, grid_h, grid_w, -1),
                h_freqs[:grid_h].view(1, grid_h, 1, -1).expand(num_frames, grid_h, grid_w, -1),
                w_freqs[:grid_w].view(1, 1, grid_w, -1).expand(num_frames, grid_h, grid_w, -1),
            ),
            dim=-1,
        ).reshape(num_frames * grid_h * grid_w, 1, -1)
        self._rope_cache[key] = freqs
        return freqs

    def embed(self, latents: torch.Tensor, human_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B, N, C, h, w] latents -> [B, N*tokens_per_frame, D] tokens.

        `human_mask` is [B, 1+K] with True = this human frame is provided. Frames
        marked False are replaced by `absent_human_embedding`, so they keep their
        slot, role and RoPE position but carry no information from any episode.
        """

        config = self.config
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, N, C, h, w], got {tuple(latents.shape)}")
        if latents.shape[1] != config.num_video_frames:
            raise ValueError(
                f"expected {config.num_video_frames} video frames, got {latents.shape[1]}"
            )
        if latents.shape[2:] != (
            config.latent_channels,
            config.latent_height,
            config.latent_width,
        ):
            raise ValueError(
                f"latent shape {tuple(latents.shape[2:])} does not match config "
                f"({config.latent_channels}, {config.latent_height}, {config.latent_width})"
            )
        # patch_size[0] == 1, so the conv never mixes two frames together.
        x = self.patch_embedding(latents.permute(0, 2, 1, 3, 4))
        tokens = rearrange(x, "b d n h w -> b n (h w) d")
        if human_mask is not None:
            start = 1  # human latent steps occupy [1, 1 + num_latent_steps)
            end = 1 + config.num_latent_steps
            if human_mask.shape != (tokens.shape[0], end - start):
                raise ValueError(
                    f"human_mask must be [B, {end - start}], got {tuple(human_mask.shape)}"
                )
            keep = human_mask[:, :, None, None].to(torch.bool).to(tokens.device)
            # Whole-frame masking: every token of a human frame is dropped
            # together, never a scattered subset of patches. The hard attention
            # mask already makes these unreadable; zeroing also removes the data.
            human = tokens[:, start:end] * keep.to(tokens.dtype)
            # Availability code onto the robot-future (prediction) frames.
            slots = torch.arange(end - start, device=tokens.device)
            if config.use_human_video_context:
                code = self.availability_embedding.to(tokens.dtype)[
                    slots[None, :], human_mask.to(torch.long).to(tokens.device)
                ].sum(dim=1)[:, None, None, :]
            else:
                # A fixed "all absent" code would not carry episode data, but it
                # would create a side channel absent from the robot-z0-only claim.
                # Keep a zero-valued autograd edge so DDP still reduces this
                # retained same-size parameter instead of flagging it unused.
                code = self.availability_embedding.sum().to(tokens.dtype) * 0.0
            future = tokens[:, end:] + code
            tokens = torch.cat((tokens[:, :start], human, future), dim=1)
        roles = torch.tensor(config.frame_roles, dtype=torch.long, device=latents.device)
        tokens = tokens + self.role_embedding[roles][None, :, None, :].to(tokens.dtype)
        if not config.use_human_video_context:
            tokens[:, 1 : 1 + config.num_latent_steps] = 0
        return rearrange(tokens, "b n s d -> b (n s) d")

    def unembed(self, tokens: torch.Tensor, t: torch.Tensor, num_frames: int) -> torch.Tensor:
        """[B, N*tokens_per_frame, D] -> [B, N, C, h, w] velocity in latent space."""

        config = self.config
        grid_h = config.latent_height // config.patch_height
        grid_w = config.latent_width // config.patch_width
        x = self.head(tokens, t)
        return rearrange(
            x,
            "b (n h w) (c ph pw) -> b n c (h ph) (w pw)",
            n=num_frames,
            h=grid_h,
            w=grid_w,
            ph=config.patch_height,
            pw=config.patch_width,
        )


class HRActionExpert(_ExpertBase):
    """Action expert over one token per future horizon, with 1D RoPE on the horizon."""

    def __init__(self, config: HRMoTConfig) -> None:
        super().__init__(
            hidden_dim=config.action_hidden_dim,
            ffn_dim=config.action_ffn_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            attn_head_dim=config.attn_head_dim,
            freq_dim=config.freq_dim,
            eps=config.eps,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.config = config
        if config.num_text_tokens:
            self.text_embedding = nn.Embedding(
                config.text_vocab_size, config.text_hidden_dim, padding_idx=0
            )
            self.text_position = nn.Parameter(
                torch.randn(config.num_text_tokens, config.text_hidden_dim) * 0.02
            )
            self.text_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    config.text_hidden_dim,
                    4,
                    config.text_hidden_dim * 4,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                ),
                2,
                enable_nested_tensor=False,
            )
            self.text_projection = nn.Linear(config.text_hidden_dim, config.action_hidden_dim)
            self.text_role = nn.Parameter(torch.zeros(config.action_hidden_dim))
        self.action_embedding = nn.Linear(config.action_dim, config.action_hidden_dim)
        self.availability_embedding = nn.Parameter(
            torch.zeros(config.num_latent_steps, 2, config.action_hidden_dim)
        )
        if config.use_state_context:
            # The robot anchor reuses action_embedding: it is the same 7-D
            # quantity, so the same projection is the right one. Human hand
            # motion is a different 84-D modality and gets its own projection.
            self.human_action_embedding = nn.Linear(
                config.human_action_dim, config.action_hidden_dim
            )
            self.action_role_embedding = nn.Parameter(
                torch.zeros(NUM_ROLES, config.action_hidden_dim)
            )
        self.head = HRHead(config.action_hidden_dim, config.action_dim, config.eps)
        self._freqs_1d = precompute_freqs_cis(config.attn_head_dim, end=config.max_rope_position)

    def rope_freqs(self, device: torch.device) -> torch.Tensor:
        key = (device, 0)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.tensor(self.config.action_rope_positions, dtype=torch.long, device=device)
        freqs = self._freqs_1d.to(device)[positions].view(positions.numel(), 1, -1)
        self._rope_cache[key] = freqs
        return freqs

    def embed(
        self,
        actions: torch.Tensor,
        human_mask: torch.Tensor | None = None,
        robot_current_state: torch.Tensor | None = None,
        human_future_action: torch.Tensor | None = None,
        text_token_ids: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """-> [B, num_action_sequence, D], condition block first when enabled."""

        config = self.config
        if actions.ndim != 3 or actions.shape[1:] != (
            config.num_action_tokens,
            config.action_dim,
        ):
            raise ValueError(
                f"actions must be [B, {config.num_action_tokens}, {config.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        predicted = self.action_embedding(actions)
        if config.use_state_context:
            if robot_current_state is None or human_future_action is None:
                raise ValueError("use_state_context requires query current state and source motion")
            if robot_current_state.shape[1:] != (config.action_dim,):
                raise ValueError(
                    f"robot_current_state must be [B, {config.action_dim}], "
                    f"got {tuple(robot_current_state.shape)}"
                )
            if human_future_action.shape[1:] != (
                config.num_source_motion_tokens,
                config.human_action_dim,
            ):
                raise ValueError(
                    f"human_future_action must be [B, {config.num_source_motion_tokens}, "
                    f"{config.human_action_dim}], got {tuple(human_future_action.shape)}"
                )
            role = self.action_role_embedding.to(predicted.dtype)
            anchor = self.action_embedding(robot_current_state)[:, None, :]
            anchor = anchor + role[ROLE_ROBOT_CURRENT]
            if not config.use_robot_state_context:
                anchor = torch.zeros_like(anchor)
            human = self.human_action_embedding(human_future_action) + role[ROLE_HUMAN_CONTEXT]
            if human_mask is not None:
                # Zero a masked step's human-motion tokens as well; the hard attention
                # mask already makes them unreadable, this also drops the data.
                steps = torch.tensor(config.human_action_latent_step(), device=human.device)
                keep = human_mask.to(human.device)[:, steps][:, :, None]
                human = human * keep.to(human.dtype)
            predicted = predicted + role[ROLE_ROBOT_FUTURE]
            clean = [anchor, human]
            if config.num_text_tokens:
                batch = actions.shape[0]
                if text_token_ids is None:
                    text_token_ids = torch.zeros(
                        batch, config.num_text_tokens, dtype=torch.long, device=actions.device
                    )
                if text_valid is None or not config.use_language_context:
                    text_valid = torch.zeros(
                        batch, config.num_text_tokens, dtype=torch.bool, device=actions.device
                    )
                if (
                    text_token_ids.shape != (batch, config.num_text_tokens)
                    or text_valid.shape != text_token_ids.shape
                ):
                    raise ValueError("Invalid word-token IDs/validity")
                # Zero hidden input IDs before embedding. Even arbitrary IDs in
                # missing positions must not be looked up or affect attention.
                text_token_ids = text_token_ids.masked_fill(~text_valid, 0)
                safe_valid = text_valid.clone()
                safe_valid[:, 0] = True
                text = self.text_embedding(text_token_ids) + self.text_position[None]
                text = self.text_encoder(text, src_key_padding_mask=~safe_valid)
                text = (self.text_projection(text) + self.text_role).masked_fill(
                    ~text_valid[..., None], 0.0
                )
                clean.append(text)
            tokens = torch.cat((*clean, predicted), dim=1)
        else:
            tokens = predicted
        if human_mask is not None:
            if config.use_human_motion_context:
                slots = torch.arange(human_mask.shape[1], device=tokens.device)
                code = self.availability_embedding.to(tokens.dtype)[
                    slots[None, :], human_mask.to(torch.long).to(tokens.device)
                ].sum(dim=1)[:, None, :]
            else:
                # Same zero-edge contract as the video expert: no information or
                # numerical effect, while every retained parameter participates.
                code = self.availability_embedding.sum().to(tokens.dtype) * 0.0
            tokens = tokens + code
        return tokens

    def unembed(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.head(tokens, t)


class HRMoTFlowModel(nn.Module):
    """The trainable ~0.6B model. VAE encoding happens outside, in the trainer."""

    def __init__(self, config: HRMoTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # MoT owns the experts. Registering them a second time on this module
        # would duplicate every tensor in state_dict, so they are reached through
        # properties instead of attributes.
        self.mot = MoT(
            mixtures={"video": HRVideoExpert(config), "action": HRActionExpert(config)},
            mot_checkpoint_mixed_attn=config.gradient_checkpointing,
        )
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=config.video_train_shift
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=config.video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=config.action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=config.action_infer_shift
        )
        self.reset_parameters()

    @property
    def video_expert(self) -> HRVideoExpert:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> HRActionExpert:
        return self.mot.mixtures["action"]

    def reset_parameters(self) -> None:
        """Random init only. No pretrained weights enter the trainable model."""

        def init_linear(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_linear)
        nn.init.normal_(self.video_expert.role_embedding, mean=0.0, std=0.02)
        for expert in (self.video_expert, self.action_expert):
            nn.init.normal_(expert.availability_embedding, mean=0.0, std=0.02)
        if self.config.use_state_context:
            nn.init.normal_(self.action_expert.action_role_embedding, mean=0.0, std=0.02)
        # Wan/DiT init: `modulation` keeps its randn/sqrt(dim) scale (set in the
        # block) and residual output projections are down-scaled by depth so a
        # 22-layer stack does not blow up at step 0.
        residual_std = 0.02 / math.sqrt(2.0 * self.config.num_layers)
        # The final head is initialized *near* zero rather than at zero. Exactly
        # zero makes the model emit zero velocity, which is stable, but it also
        # makes d(loss)/d(head input) identically zero: on the first step only the
        # two head tensors per expert receive gradient and the other 679 do not.
        # A tiny nonzero scale keeps the near-zero start and lets gradient reach
        # every parameter from step 1.
        head_std = 1e-4
        for expert in (self.video_expert, self.action_expert):
            for block in expert.blocks:
                nn.init.normal_(block.self_attn.o.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.ffn[-1].weight, mean=0.0, std=residual_std)
            nn.init.normal_(expert.head.head.weight, mean=0.0, std=head_std)
            nn.init.zeros_(expert.head.head.bias)

    def attention_mask(
        self,
        human_mask: torch.Tensor,
        *,
        human_motion_mask: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
        device: torch.device,
    ) -> torch.Tensor:
        return build_availability_attention_mask(
            self.config,
            human_mask,
            human_motion_mask=human_motion_mask,
            text_valid=text_valid,
            device=device,
        )

    def human_availability_patterns(self, device: torch.device) -> torch.Tensor:
        """Every availability pattern over the maskable human latent steps.

        All 2^Tz patterns when the human current step is droppable, otherwise the
        2^(Tz-1) patterns that keep it.
        """

        slots = self.config.num_latent_steps
        if not self.config.droppable_human_current:
            slots -= 1
        codes = torch.arange(2**slots, device=device)
        bits = torch.arange(slots, device=device)
        return ((codes[:, None] >> bits[None, :]) & 1).bool()

    def sample_human_mask(
        self,
        batch: int,
        *,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """[B, Tz] bool. True = that human latent step is supplied.

        `full_human_context_prob` of the batch gets the complete context so
        full-context quality does not regress; the rest are drawn *uniformly* over
        every other pattern. This is a regularizer, not a curriculum.

        When `droppable_human_current` is set, the human z0 step is included in
        the draw, so "no human context at all" is a trained condition rather than
        an out-of-distribution query, and the model cannot lean entirely on the
        same-moment cross-embodiment correspondence.
        """

        config = self.config
        patterns = self.human_availability_patterns(device)
        # The last row is the all-ones pattern; draw the others uniformly.
        choice = torch.randint(
            0, patterns.shape[0] - 1, (batch,), device=device, generator=generator
        )
        drawn = patterns[choice]
        use_full = torch.rand(batch, device=device, generator=generator) < float(
            config.full_human_context_prob
        )
        drawn = torch.where(use_full[:, None], torch.ones_like(drawn), drawn)
        if config.droppable_human_current:
            return drawn
        current = torch.ones((batch, 1), dtype=torch.bool, device=device)
        return torch.cat((current, drawn), dim=1)

    def _token_timesteps(
        self, timestep: torch.Tensor, *, num_frames: int, noisy_frames: int
    ) -> torch.Tensor:
        """Clean condition frames get t=0; only the robot future frames carry t=s."""

        config = self.config
        batch = timestep.shape[0]
        per_frame = torch.zeros((batch, num_frames), dtype=torch.float32, device=timestep.device)
        per_frame[:, num_frames - noisy_frames :] = timestep.float()[:, None]
        return per_frame.repeat_interleave(config.tokens_per_frame, dim=1)

    def forward_tokens(
        self,
        *,
        robot_current_latent: torch.Tensor,
        human_latents: torch.Tensor,
        noisy_robot_future_latents: torch.Tensor,
        noisy_robot_future_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        human_mask: torch.Tensor | None = None,
        human_motion_mask: torch.Tensor | None = None,
        robot_current_state: torch.Tensor | None = None,
        human_future_action: torch.Tensor | None = None,
        text_token_ids: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the MoT and return (tokens_out, video_t, action_t).

        `tokens_out["video"]` still holds the condition positions, which
        `predict_velocity` drops. Tests use them to prove the attention mask
        really isolates condition tokens from the noisy prediction tokens.
        """

        config = self.config
        if robot_current_latent.ndim != 5 or robot_current_latent.shape[1] != 1:
            raise ValueError(
                f"robot_current_latent must be [B, 1, C, h, w], got "
                f"{tuple(robot_current_latent.shape)}"
            )
        if human_latents.shape[1] != config.num_latent_steps:
            raise ValueError(
                f"human_latents must carry {config.num_latent_steps} latent steps, "
                f"got {human_latents.shape[1]}"
            )
        if noisy_robot_future_latents.shape[1] != config.num_horizons:
            raise ValueError(
                f"noisy_robot_future_latents must carry {config.num_horizons} horizons, "
                f"got {noisy_robot_future_latents.shape[1]}"
            )

        latents = torch.cat(
            (robot_current_latent, human_latents, noisy_robot_future_latents), dim=1
        )
        device = latents.device

        if human_mask is None:
            supplied_human_mask = torch.ones(
                (latents.shape[0], config.num_latent_steps),
                dtype=torch.bool,
                device=device,
            )
        else:
            supplied_human_mask = human_mask.to(device=device, dtype=torch.bool)
        if human_motion_mask is None:
            human_motion_mask = supplied_human_mask
        else:
            human_motion_mask = human_motion_mask.to(device=device, dtype=torch.bool)
        # Configuration-level ablations fail closed: callers cannot accidentally
        # re-enable a source by passing an all-True runtime mask.
        human_mask = (
            supplied_human_mask
            if config.use_human_video_context
            else torch.zeros_like(supplied_human_mask)
        )
        if not config.use_human_motion_context:
            human_motion_mask = torch.zeros_like(human_motion_mask)
        video_tokens = self.video_expert.embed(latents, human_mask=human_mask)
        action_tokens = self.action_expert.embed(
            noisy_robot_future_action,
            human_mask=human_motion_mask,
            robot_current_state=robot_current_state,
            human_future_action=human_future_action,
            text_token_ids=text_token_ids,
            text_valid=text_valid,
        )

        video_token_t = self._token_timesteps(
            timestep_video,
            num_frames=config.num_video_frames,
            noisy_frames=config.num_horizons,
        )
        # Condition tokens are clean (t=0); only the predicted states carry t=s.
        action_token_t = torch.zeros(
            (latents.shape[0], config.num_action_sequence),
            dtype=torch.float32,
            device=device,
        )
        action_token_t[:, config.num_action_condition_tokens :] = timestep_action.float()[:, None]

        video_t, video_t_mod = self.video_expert.embed_timesteps(video_token_t)
        action_t, action_t_mod = self.action_expert.embed_timesteps(action_token_t)

        mask = self.attention_mask(
            human_mask, human_motion_mask=human_motion_mask, text_valid=text_valid, device=device
        )
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=mask,
            freqs_all={
                "video": self.video_expert.rope_freqs(device),
                "action": self.action_expert.rope_freqs(device),
            },
            # No cross-attention: human context lives in the self-attention sequence.
            context_all={"video": None, "action": None},
            t_mod_all={"video": video_t_mod, "action": action_t_mod},
        )
        return tokens_out, video_t, action_t

    def predict_velocity(
        self,
        *,
        robot_current_latent: torch.Tensor,
        human_latents: torch.Tensor,
        noisy_robot_future_latents: torch.Tensor,
        noisy_robot_future_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        human_mask: torch.Tensor | None = None,
        human_motion_mask: torch.Tensor | None = None,
        robot_current_state: torch.Tensor | None = None,
        human_future_action: torch.Tensor | None = None,
        text_token_ids: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flow velocity for the robot future latents and actions."""

        config = self.config
        tokens_out, video_t, action_t = self.forward_tokens(
            robot_current_latent=robot_current_latent,
            human_latents=human_latents,
            noisy_robot_future_latents=noisy_robot_future_latents,
            noisy_robot_future_action=noisy_robot_future_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            human_mask=human_mask,
            human_motion_mask=human_motion_mask,
            robot_current_state=robot_current_state,
            human_future_action=human_future_action,
            text_token_ids=text_token_ids,
            text_valid=text_valid,
        )
        # Only the robot-future slice is a prediction; drop the condition tokens.
        start = config.num_condition_tokens
        pred_video = self.video_expert.unembed(
            tokens_out["video"][:, start:, :],
            video_t[:, start:, :],
            num_frames=config.num_horizons,
        )
        start_action = config.num_action_condition_tokens
        pred_action = self.action_expert.unembed(
            tokens_out["action"][:, start_action:, :], action_t[:, start_action:, :]
        )
        return pred_video, pred_action

    def training_loss(
        self,
        *,
        robot_current_latent: torch.Tensor,
        human_latents: torch.Tensor,
        robot_future_latents: torch.Tensor,
        robot_future_action: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        human_mask: Optional[torch.Tensor] = None,
        human_motion_mask: Optional[torch.Tensor] = None,
        robot_current_state: Optional[torch.Tensor] = None,
        human_future_action: Optional[torch.Tensor] = None,
        text_token_ids: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        config = self.config
        batch = robot_future_latents.shape[0]
        device = robot_future_latents.device
        dtype = robot_future_latents.dtype

        timestep_video = self.train_video_scheduler.sample_training_t(batch, device, dtype)
        timestep_action = self.train_action_scheduler.sample_training_t(batch, device, dtype)

        noise_video = torch.randn(
            robot_future_latents.shape, device=device, dtype=dtype, generator=generator
        )
        noise_action = torch.randn(
            robot_future_action.shape, device=device, dtype=dtype, generator=generator
        )
        noisy_video = self.train_video_scheduler.add_noise(
            robot_future_latents, noise_video, timestep_video
        )
        noisy_action = self.train_action_scheduler.add_noise(
            robot_future_action, noise_action, timestep_action
        )
        target_video = self.train_video_scheduler.training_target(
            robot_future_latents, noise_video, timestep_video
        )
        target_action = self.train_action_scheduler.training_target(
            robot_future_action, noise_action, timestep_action
        )

        if human_mask is None:
            human_mask = self.sample_human_mask(batch, device=device, generator=generator)

        pred_video, pred_action = self.predict_velocity(
            robot_current_latent=robot_current_latent,
            human_latents=human_latents,
            noisy_robot_future_latents=noisy_video,
            noisy_robot_future_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            human_mask=human_mask,
            human_motion_mask=human_motion_mask,
            robot_current_state=robot_current_state,
            human_future_action=human_future_action,
            text_token_ids=text_token_ids,
            text_valid=text_valid,
        )

        # Per-sample means, then the scheduler's timestep weighting -- matching
        # FastWAM rather than averaging a single scalar over the whole batch.
        video_per_sample = (pred_video.float() - target_video.float()).pow(2).flatten(1).mean(dim=1)
        action_per_sample = (
            (pred_action.float() - target_action.float()).pow(2).flatten(1).mean(dim=1)
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_per_sample.device, video_per_sample.dtype
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_per_sample.device, action_per_sample.dtype
        )
        loss_video = (video_per_sample * video_weight).mean()
        loss_action = (action_per_sample * action_weight).mean()
        total = config.loss_lambda_video * loss_video + config.loss_lambda_action * loss_action
        return total, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "loss_video_unweighted": video_per_sample.mean().detach(),
            "loss_action_unweighted": action_per_sample.mean().detach(),
            "human_frames_kept": (
                human_mask.sum(dim=1).float().mean().detach()
                if config.use_human_video_context
                else torch.zeros((), device=device)
            ),
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        robot_current_latent: torch.Tensor,
        human_latents: torch.Tensor,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        human_mask: Optional[torch.Tensor] = None,
        human_motion_mask: Optional[torch.Tensor] = None,
        robot_current_state: Optional[torch.Tensor] = None,
        human_future_action: Optional[torch.Tensor] = None,
        text_token_ids: torch.Tensor | None = None,
        text_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FastWAM-style joint flow integration in latent + action space.

        This is the latent-space core and returns
        (robot_future_latents, normalized robot_future_state). It does not touch
        the VAE, which lives outside the model so DDP never syncs its 705M frozen
        parameters. For RGB frames and un-normalized state use
        `fasterwam.models.hr_inference.generate_rollout`.

        `human_mask` is [B, 1+K] and controls human video steps.
        `human_motion_mask` controls the 84-D motion tokens mapped to those
        steps. It defaults to `human_mask` for the standard model and to an
        all-False mask for a configured human-video-only model.
        """

        config = self.config
        batch = robot_current_latent.shape[0]
        device = robot_current_latent.device
        dtype = robot_current_latent.dtype

        latents_video = torch.randn(
            (
                batch,
                config.num_horizons,
                config.latent_channels,
                config.latent_height,
                config.latent_width,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents_action = torch.randn(
            (batch, config.num_action_tokens, config.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        timesteps_video, deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        timesteps_action, deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )

        for step in range(num_inference_steps):
            step_t_video = timesteps_video[step].expand(batch)
            step_t_action = timesteps_action[step].expand(batch)
            pred_video, pred_action = self.predict_velocity(
                robot_current_latent=robot_current_latent,
                human_latents=human_latents,
                noisy_robot_future_latents=latents_video,
                noisy_robot_future_action=latents_action,
                timestep_video=step_t_video,
                timestep_action=step_t_action,
                human_mask=human_mask,
                human_motion_mask=human_motion_mask,
                robot_current_state=robot_current_state,
                human_future_action=human_future_action,
                text_token_ids=text_token_ids,
                text_valid=text_valid,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, deltas_video[step], latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, deltas_action[step], latents_action
            )
        return latents_video, latents_action

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_breakdown(model: HRMoTFlowModel) -> dict[str, int]:
    return {
        "total": count_parameters(model),
        "video_expert": count_parameters(model.video_expert),
        "action_expert": count_parameters(model.action_expert),
        "video_blocks": sum(count_parameters(b) for b in model.video_expert.blocks),
        "action_blocks": sum(count_parameters(b) for b in model.action_expert.blocks),
    }
