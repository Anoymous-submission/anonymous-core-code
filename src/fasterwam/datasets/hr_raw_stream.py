"""Streaming H&R reader: raw HDF5 -> RGB frames at the fixed horizons, no cache.

One sample is:

  robot current frame            (offset 0, target episode)
  human frames                   (offsets 0 and every future horizon, *context* episode)
  human hand motion             (contiguous offsets 1..32, *context* episode)
  robot future frames            (every future horizon, target episode)  -- denoised
  robot future state             (contiguous offsets 1..32, target episode) -- denoised

Two things this module is deliberate about, both of which the audit of the
previous run flagged:

* The video segment is 9 frames sampled at stride 4, spanning 32 control steps
  (1.09 s at 29.4 fps), encoded as one clip so the VAE applies its temporal
  compression the way FastWAM does: z0<-frame 0, z1<-frames 1-4, z2<-frames 5-8,
  i.e. control steps 0 / 1-16 / 17-32.
* The human hand-motion condition and robot state target are 32 *contiguous*
  control steps covering that same
  span, so the action stream runs at 4x the video rate as in FastWAM.
* The robot target is `end_position` (6-D) + `gripper_state` (1-D). The HDF5
  `action` stream is a robot controller setpoint and is deliberately excluded:
  using it as human context leaks a near-copy of the future robot target.
  Human motion comes only from `transformed_hand_frames` and
  `transformed_hand_coords`.
* `human_camera` and `robot_camera` live in the SAME episode file, have the SAME
  length, and are temporally aligned frame by frame: index t is the same phase of
  the same task, performed by a human in one stream and by the robot in the other.
  So the human frames at t+1..t+8 show precisely what the robot should do next --
  that alignment is the entire conditioning signal, and `context="aligned"`
  preserves it by taking the human stream from the same episode at the same t0.
  `context="cross_episode"` is the ablation: a human clip from a different episode
  of the same instruction. Episode lengths inside one instruction group range from
  364 to 1218 frames, so there is no meaningful index correspondence across
  episodes, and that ablation is expected to destroy the signal.
* The window start `t0` is sampled uniformly per item, not pinned to frame 0.
  Reading only frame 0 meant the dataset held just one sample per episode -- 1,248
  in total, covering 2.3% of each trajectory -- which a 593M model memorizes
  almost immediately. Random starts give ~466,000 distinct windows over the same
  episodes. `t0` is capped at `len - max_offset` so no window is mostly padding.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

# Number of RGB frames handed to the VAE. Must satisfy (T-1) % 4 == 0: the Wan2.2
# VAE folds each run of 4 frames into one latent step, giving 1 + (T-1)/4 steps.
NUM_RGB_FRAMES: int = 9

# Control steps between consecutive video frames. FastWAM samples 33 control steps
# and keeps every 4th as a video frame (0, 4, ..., 32) -- `num_frames=33` with
# `action_video_freq_ratio=4` in robot_video_dataset.py. Reading 9 *contiguous*
# frames instead covers only 8 control steps, which at this data's 29.4 fps is
# 0.27 s: too short for anything to happen. Measured over 40 episodes, the binary
# gripper changes in 6.4% of 8-step windows but 25.4% of 32-step ones, so ~94% of
# samples carried no gripper event at all. With stride 4 the same 9 frames span
# 32 control steps = 1.09 s.
FRAME_STRIDE: int = 4

# Contiguous robot action steps predicted per sample, covering the full video
# span. FastWAM allows the action stream to run faster than the video
# (`action_horizon % (num_frames - 1) == 0`); here one action per control step.
ACTION_HORIZON: int = 32

# Native H&R frames are 240x426. Both image dims must be divisible by 32: the
# Wan2.2 VAE compresses 16x and the DiT patch takes another 2x.
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 384

STATE_DIM = 7

# Raw human motion: a 4x3 wrist frame followed by 24x3 hand coordinates.
# Keep all 84 dimensions. A derived 7-D wrist/grasp representation discards
# substantial finger-articulation variance and is not stored explicitly in HDF5.
HAND_POSE_DIM = 4 * 3
HAND_COORD_DIM = 24 * 3
HUMAN_ACTION_DIM = HAND_POSE_DIM + HAND_COORD_DIM  # 84
HUMAN_ACTION_SOURCE = "transformed_hand_frames(4x3)+transformed_hand_coords(24x3)"


@dataclass(frozen=True)
class RawHREpisode:
    path: Path
    annotation_path: str
    instruction: str | None
    group_key: str
    task: str
    needs_review: bool
    clean_behavior_eligible: bool | None
    trajectory_quality: str | None
    human_frames: int
    robot_frames: int


@dataclass(frozen=True)
class StateStats:
    """Per-dimension normalization for robot state and raw human hand motion."""

    robot_mean: tuple[float, ...]
    robot_std: tuple[float, ...]
    human_action_mean: tuple[float, ...] = (0.0,) * HUMAN_ACTION_DIM
    human_action_std: tuple[float, ...] = (1.0,) * HUMAN_ACTION_DIM

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "robot_mean": list(self.robot_mean),
            "robot_std": list(self.robot_std),
            "human_action_mean": list(self.human_action_mean),
            "human_action_std": list(self.human_action_std),
        }

    @classmethod
    def from_array(cls, values: np.ndarray) -> "StateStats":
        """Flat [2*STATE_DIM + 2*HUMAN_ACTION_DIM] vector -> stats."""
        expected = 2 * STATE_DIM + 2 * HUMAN_ACTION_DIM
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size != expected:
            raise ValueError(f"expected {expected} stat values, got {flat.size}")
        a, b = STATE_DIM, 2 * STATE_DIM
        c = b + HUMAN_ACTION_DIM
        return cls(
            robot_mean=tuple(float(x) for x in flat[:a]),
            robot_std=tuple(float(x) for x in flat[a:b]),
            human_action_mean=tuple(float(x) for x in flat[b:c]),
            human_action_std=tuple(float(x) for x in flat[c:]),
        )

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.robot_mean, dtype=np.float32),
                np.asarray(self.robot_std, dtype=np.float32),
                np.asarray(self.human_action_mean, dtype=np.float32),
                np.asarray(self.human_action_std, dtype=np.float32),
            ]
        )


def _resize_video(video: np.ndarray, height: int, width: int) -> torch.Tensor:
    """uint8 [N,H,W,3] -> float [N,3,height,width] in [-1, 1], the VAE's input range."""

    output = np.empty((len(video), height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(video):
        output[index] = cv2.resize(
            np.clip(frame, 0, 255).astype(np.uint8),
            (width, height),  # cv2 takes (w, h)
            interpolation=cv2.INTER_AREA,
        )
    return torch.from_numpy(output.copy()).permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0


def _take_with_last_padding(
    dataset: h5py.Dataset, indices: Sequence[int]
) -> tuple[np.ndarray, int]:
    """Gather `indices`, clamping past-the-end reads to the final frame.

    Also returns how many indices were clamped, so overflow stays measurable
    instead of silently reshaping the objective.
    """

    requested = np.asarray(indices, dtype=np.int64)
    clamped = np.minimum(requested, len(dataset) - 1)
    overflow = int((requested > clamped).sum())
    unique, inverse = np.unique(clamped, return_inverse=True)
    return np.asarray(dataset[unique])[inverse], overflow


def _task_from_path(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).parts[0]


def _load_annotation_rows(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_path = str(row.get("episode_path", ""))
            if "/v1/" not in episode_path:
                continue
            if episode_path in rows:
                raise ValueError(f"duplicate annotation path: {episode_path}")
            rows[episode_path] = row
    return rows


def _episode_lengths(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        human_frames = len(handle["cam_data/human_camera"])
        robot_frames = len(handle["cam_data/robot_camera"])
        human_hand_frames = len(handle["transformed_hand_frames"])
        human_hand_coords = len(handle["transformed_hand_coords"])
        robot_positions = len(handle["end_position"])
        robot_gripper = len(handle["gripper_state"])
    if len({human_frames, human_hand_frames, human_hand_coords}) != 1:
        raise ValueError(
            f"human modalities have different lengths in {path}: "
            f"video={human_frames}, frames={human_hand_frames}, coords={human_hand_coords}"
        )
    if len({robot_frames, robot_positions, robot_gripper}) != 1:
        raise ValueError(
            f"robot modalities have different lengths in {path}: "
            f"{robot_frames}, {robot_positions}, {robot_gripper}"
        )
    # context="aligned" reads the human stream at the robot's t0, which is only
    # meaningful if the two cameras are the same recording. Without this check a
    # mismatch is silently absorbed by last-frame clamping.
    if human_frames != robot_frames:
        raise ValueError(
            f"human/robot camera length mismatch in {path}: " f"{human_frames} != {robot_frames}"
        )
    if human_frames < 1 or robot_frames < 2:
        raise ValueError(
            f"episode is too short in {path}: human={human_frames}, robot={robot_frames}"
        )
    return human_frames, robot_frames


def discover_raw_v1_episodes(
    *,
    workspace_root: Path,
    data_root: Path,
    annotation_paths: Sequence[Path],
) -> list[RawHREpisode]:
    annotations = _load_annotation_rows(annotation_paths)
    raw_paths = sorted(path for path in data_root.rglob("*.hdf5") if path.is_file())
    records: list[RawHREpisode] = []
    raw_annotation_paths: set[str] = set()
    for path in raw_paths:
        annotation_path = str(path.relative_to(workspace_root))
        raw_annotation_paths.add(annotation_path)
        row = annotations.get(annotation_path)
        if row is None:
            raise ValueError(f"raw HDF5 lacks annotation: {annotation_path}")
        task = _task_from_path(path, data_root)
        instruction_value = row.get("instruction")
        instruction = (
            instruction_value.strip()
            if isinstance(instruction_value, str) and instruction_value.strip()
            else None
        )
        human_frames, robot_frames = _episode_lengths(path)
        records.append(
            RawHREpisode(
                path=path,
                annotation_path=annotation_path,
                instruction=instruction,
                group_key=instruction if instruction is not None else f"task:{task}",
                task=task,
                needs_review=bool(row.get("needs_review")),
                clean_behavior_eligible=row.get("clean_behavior_eligible"),
                trajectory_quality=(
                    str(row["trajectory_quality"])
                    if row.get("trajectory_quality") is not None
                    else None
                ),
                human_frames=human_frames,
                robot_frames=robot_frames,
            )
        )
    missing_raw = sorted(set(annotations) - raw_annotation_paths)
    if missing_raw:
        raise ValueError(f"annotations lack raw HDF5 files: {missing_raw[:5]}")
    if len(records) != len(annotations):
        raise ValueError(f"raw/annotation count mismatch: {len(records)} != {len(annotations)}")
    return records


def _robot_state(handle: h5py.File) -> np.ndarray:
    """[T, 7] robot end-effector pose + gripper."""

    robot = np.concatenate(
        (
            np.asarray(handle["end_position"], dtype=np.float32),
            np.asarray(handle["gripper_state"], dtype=np.float32)[:, None],
        ),
        axis=-1,
    )
    if robot.ndim != 2 or robot.shape[-1] != STATE_DIM:
        raise ValueError(f"robot state must be {STATE_DIM}-D, got {robot.shape}")
    return robot


def _human_action(handle: h5py.File) -> np.ndarray:
    """[T, 84] raw human wrist frame and articulated hand coordinates."""

    pose = np.asarray(handle["transformed_hand_frames"], dtype=np.float32)
    coords = np.asarray(handle["transformed_hand_coords"], dtype=np.float32)
    if len(pose) != len(coords):
        raise ValueError(f"human motion length mismatch: frames={len(pose)}, coords={len(coords)}")
    human_action = np.concatenate(
        (pose.reshape(len(pose), -1), coords.reshape(len(coords), -1)), axis=-1
    )
    if human_action.ndim != 2 or human_action.shape[-1] != HUMAN_ACTION_DIM:
        raise ValueError(f"human motion must be {HUMAN_ACTION_DIM}-D, got {human_action.shape}")
    return human_action


def compute_state_stats(records: Iterable[RawHREpisode]) -> StateStats:
    out = []
    for dim, reader in ((STATE_DIM, _robot_state), (HUMAN_ACTION_DIM, _human_action)):
        total = np.zeros(dim, dtype=np.float64)
        square = np.zeros(dim, dtype=np.float64)
        count = 0
        for record in records:
            with h5py.File(record.path, "r") as handle:
                values = reader(handle).astype(np.float64)
            total += values.sum(axis=0)
            square += np.square(values).sum(axis=0)
            count += len(values)
        if count == 0:
            raise ValueError("cannot normalize an empty H&R inventory")
        mean = total / count
        # Some wrist-local hand coordinates are exactly constant, so the floor
        # keeps normalization finite without deleting those dimensions.
        std = np.sqrt(np.maximum(square / count - np.square(mean), 1e-8))
        out.append((mean, std))
    return StateStats(
        robot_mean=tuple(out[0][0].tolist()),
        robot_std=tuple(out[0][1].tolist()),
        human_action_mean=tuple(out[1][0].tolist()),
        human_action_std=tuple(out[1][1].tolist()),
    )


class RawHRStreamingDataset(Dataset[dict[str, object]]):
    """Read the fixed future horizons straight from raw HDF5, with no cache."""

    def __init__(
        self,
        records: Sequence[RawHREpisode],
        *,
        stats: StateStats,
        num_rgb_frames: int = NUM_RGB_FRAMES,
        frame_stride: int = FRAME_STRIDE,
        action_horizon: int = ACTION_HORIZON,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        seed: int = 0,
        require_cross_episode_context: bool = True,
        deterministic: bool = False,
        windows_per_episode: int = 1,
        context: str = "aligned",
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        num_rgb_frames = int(num_rgb_frames)
        frame_stride = int(frame_stride)
        action_horizon = int(action_horizon)
        if num_rgb_frames < 5 or (num_rgb_frames - 1) % 4:
            raise ValueError(
                f"num_rgb_frames must be >=5 with (T-1) % 4 == 0, got {num_rgb_frames}"
            )
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        span = (num_rgb_frames - 1) * frame_stride
        if action_horizon != span:
            raise ValueError(
                f"action_horizon must equal the video span (num_rgb_frames-1)*stride "
                f"= {span}, got {action_horizon}"
            )
        for name, value in (("image_height", image_height), ("image_width", image_width)):
            if value % 32:
                raise ValueError(
                    f"{name}={value} must be divisible by 32 "
                    "(VAE 16x compression, then DiT 2x patch)"
                )
        self.num_rgb_frames = num_rgb_frames
        self.frame_stride = frame_stride
        self.action_horizon = action_horizon
        # The window reaches t0 + span, so t0 must stop that far from the end.
        self.max_offset = span
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.seed = int(seed)
        # deterministic=True makes t0 and the context pairing a pure function of
        # the sample index, so a held-out split scores the same batch at every
        # evaluation and the numbers are comparable across training steps.
        self.deterministic = bool(deterministic)
        self._stream: np.random.Generator | None = None
        # Each episode holds ~len-8 distinct windows, but one __getitem__ returns
        # one window, so a plain pass over the episode list is a very small epoch
        # (1,249 samples). This multiplier makes an epoch a meaningful pass over
        # the window space rather than over the episode list.
        if windows_per_episode < 1:
            raise ValueError("windows_per_episode must be >= 1")
        self.windows_per_episode = int(windows_per_episode)
        self.robot_mean = np.asarray(stats.robot_mean, dtype=np.float32)
        self.robot_std = np.asarray(stats.robot_std, dtype=np.float32)
        self.human_action_mean = np.asarray(stats.human_action_mean, dtype=np.float32)
        self.human_action_std = np.asarray(stats.human_action_std, dtype=np.float32)

        all_records = list(records)
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(all_records):
            groups.setdefault(record.group_key, []).append(index)

        # Only relevant to the cross_episode ablation: an episode whose instruction
        # group has only one member has no other episode to draw human context from. The previous pipeline silently fell
        # back to the target episode itself, so for those samples the "human
        # context" was the very episode being predicted. Drop them instead.
        singleton_indices = {
            index for members in groups.values() if len(members) == 1 for index in members
        }
        self.singleton_group_episodes = len(singleton_indices)
        if context == "cross_episode" and require_cross_episode_context and singleton_indices:
            keep = [index for index in range(len(all_records)) if index not in singleton_indices]
            if not keep:
                raise ValueError(
                    "every episode is in a singleton instruction group; there is no "
                    "cross-episode human context available"
                )
            all_records = [all_records[index] for index in keep]
            groups = {}
            for index, record in enumerate(all_records):
                groups.setdefault(record.group_key, []).append(index)

        self.records = all_records
        if context not in ("aligned", "cross_episode"):
            raise ValueError(f"context must be 'aligned' or 'cross_episode', got {context!r}")
        self.context = context
        self.require_cross_episode_context = bool(require_cross_episode_context)
        self.groups = {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_episode

    @property
    def num_latent_steps(self) -> int:
        return 1 + (self.num_rgb_frames - 1) // 4

    @property
    def video_span(self) -> int:
        """Control steps covered by the video window."""
        return (self.num_rgb_frames - 1) * self.frame_stride

    def video_window(self, start: int) -> tuple[int, ...]:
        """Strided video frame indices: t0, t0+stride, ..., t0+span."""
        return tuple(range(start, start + self.video_span + 1, self.frame_stride))

    def action_window(self, start: int) -> tuple[int, ...]:
        """Contiguous action steps t0+1 .. t0+horizon."""
        return tuple(range(start + 1, start + 1 + self.action_horizon))

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng((self.seed * 1_000_003 + int(index)) % (2**63 - 1))
        if self._stream is None:
            # torch reseeds each worker per epoch from the (seeded) main
            # generator, so this varies across epochs and stays reproducible.
            worker = torch.utils.data.get_worker_info()
            base = int(torch.initial_seed()) if worker is not None else self.seed
            self._stream = np.random.default_rng([base, self.seed])
        return self._stream

    def _start_frame(self, rng: np.random.Generator, num_frames: int) -> int:
        """Uniform window start that keeps every future horizon a real frame.

        Capping at `num_frames - max_offset` means `t0 + max_offset` lands on the
        last frame at worst, so a window is never mostly last-frame padding.
        """

        high = max(int(num_frames) - self.max_offset, 1)
        return int(rng.integers(0, high))

    def _context_index(self, target_index: int, rng: np.random.Generator) -> int:
        record = self.records[target_index]
        group = self.groups[record.group_key]
        if len(group) == 1:
            # Only reachable when require_cross_episode_context is False.
            return target_index
        candidates = group[group != target_index]
        return int(rng.choice(candidates))

    def _read_target(
        self, record: RawHREpisode, start: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Strided 9-frame robot segment, current state, and 32 future states."""

        video = self.video_window(start)
        actions = self.action_window(start)
        with h5py.File(record.path, "r") as handle:
            robot_video, video_overflow = _take_with_last_padding(
                handle["cam_data/robot_camera"], video
            )
            # (start, *actions): index 0 is the clean anchor at t0, the rest are targets.
            position, _ = _take_with_last_padding(handle["end_position"], (start, *actions))
            gripper, _ = _take_with_last_padding(handle["gripper_state"], (start, *actions))
        state = np.concatenate(
            (position.astype(np.float32), gripper.astype(np.float32)[:, None]), axis=-1
        )
        state = (state - self.robot_mean) / self.robot_std
        frames = _resize_video(robot_video, self.image_height, self.image_width)
        return (
            frames,
            torch.from_numpy(state[0].copy()),
            torch.from_numpy(state[1:].copy()),
            video_overflow,
        )

    def _read_context(
        self, record: RawHREpisode, start: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        video = self.video_window(start)
        actions = self.action_window(start)
        with h5py.File(record.path, "r") as handle:
            human_video, overflow = _take_with_last_padding(handle["cam_data/human_camera"], video)
            pose, _ = _take_with_last_padding(handle["transformed_hand_frames"], actions)
            coords, _ = _take_with_last_padding(handle["transformed_hand_coords"], actions)
        human_action = np.concatenate(
            (
                pose.astype(np.float32).reshape(len(pose), -1),
                coords.astype(np.float32).reshape(len(coords), -1),
            ),
            axis=-1,
        )
        if human_action.ndim != 2 or human_action.shape[-1] != HUMAN_ACTION_DIM:
            raise ValueError(f"human motion must be {HUMAN_ACTION_DIM}-D, got {human_action.shape}")
        human_action = (human_action - self.human_action_mean) / self.human_action_std
        return (
            _resize_video(human_video, self.image_height, self.image_width),
            torch.from_numpy(human_action.copy()),
            overflow,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        episode = index % len(self.records)
        target = self.records[episode]
        rng = self._rng(index)
        robot_start = self._start_frame(rng, target.robot_frames)
        if self.context == "aligned":
            # human_camera[t] and robot_camera[t] are the same phase of the same
            # task in the same episode, so the human stream is read from the same
            # record at the same t0. This is what makes the human future frames
            # informative about the robot future.
            context_index, context, human_start = episode, target, robot_start
        else:
            # Ablation: a different episode of the same instruction. Lengths vary
            # by 3x inside a group, so there is no index correspondence to keep.
            context_index = self._context_index(episode, rng)
            context = self.records[context_index]
            human_start = self._start_frame(rng, context.human_frames)
        robot_video, robot_current_state, robot_future_state, robot_overflow = self._read_target(
            target, robot_start
        )
        human_video, human_future_action, human_overflow = self._read_context(context, human_start)
        return {
            # Stable dataset-item identity. Unlike batch index/row, this does not
            # change when an evaluator changes batch size and is therefore safe
            # for deterministic counterfactual sampling.
            "sample_index": int(index),
            # [3, T, H, W] ordered segment -> VAE -> z0 clean cond, z1.. predicted
            "robot_video": robot_video.permute(1, 0, 2, 3).contiguous(),
            # [3, T, H, W] ordered segment from a different episode, all clean cond
            "human_video": human_video.permute(1, 0, 2, 3).contiguous(),
            # [7] clean anchor: where the arm is right now
            "robot_current_state": robot_current_state,
            # [action_horizon, 84] clean condition: raw human wrist + hand motion
            "human_future_action": human_future_action,
            # [action_horizon, 7] target: end_position (6) + gripper_state (1)
            "robot_future_state": robot_future_state,
            "num_rgb_frames": self.num_rgb_frames,
            "frame_stride": self.frame_stride,
            "action_horizon": self.action_horizon,
            "aligned_context": self.context == "aligned",
            "robot_start_frame": robot_start,
            "human_start_frame": human_start,
            "overflow_padded_frames": torch.tensor(
                robot_overflow + human_overflow, dtype=torch.long
            ),
            "target_index": episode,
            "context_index": context_index,
            "target_group": target.group_key,
            "context_group": context.group_key,
            "target_path": target.annotation_path,
            "context_path": context.annotation_path,
        }
