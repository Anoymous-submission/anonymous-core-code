"""Synchronized human RGB/motion interventions. Robot inputs never modified."""

import h5py
import numpy as np
import torch
from fasterwam.datasets.hr_raw_stream import _resize_video

SPECS = {"ID": {"kind": "identity"}}
for d in [-16, -8, -4, 4, 8, 16]:
    SPECS[f"joint_shift_{d:+d}"] = {"kind": "shift", "delta": d, "modalities": "both"}
SPECS.update(
    {
        "video_shift_+8": {"kind": "shift", "delta": 8, "modalities": "video"},
        "motion_shift_+8": {"kind": "shift", "delta": 8, "modalities": "motion"},
        "same_task_phase": {"kind": "same", "phase": "relative"},
        "same_task_start": {"kind": "same", "phase": "start"},
        "same_task_end": {"kind": "same", "phase": "end"},
        "other_task_phase": {"kind": "wrong", "phase": "relative"},
        "joint_reverse": {"kind": "reverse"},
        "joint_static": {"kind": "static"},
        "joint_hold16": {"kind": "hold"},
    }
)


def replace_context(dataset, items, metas, name, pairing):
    spec = SPECS[name]
    videos = []
    motions = []
    info = []
    for item, meta in zip(items, metas):
        idx = meta["sample_index"]
        original = pairing["geometry"][str(idx)]
        source = original
        start = int(item["human_start_frame"])
        requested = start
        original_start = start
        assert original_start <= original["max_start"], "Original human motion/video would pad"
        if spec["kind"] in ["same", "wrong"]:
            donor = pairing["pairs"][str(idx)][spec["kind"]]
            source = pairing["geometry"][str(donor)]
            assert (
                source["path"] != original["path"]
                and source["human_rgb_content_sha256"] != original["human_rgb_content_sha256"]
            )
            assert (source["group"] == original["group"]) == (spec["kind"] == "same")
            phase = spec["phase"]
            start = (
                0
                if phase == "start"
                else (
                    source["max_start"]
                    if phase == "end"
                    else round(start / max(original["max_start"], 1) * source["max_start"])
                )
            )
            requested = start
        elif spec["kind"] == "shift":
            requested = start + spec["delta"]
            start = requested
        start = max(0, min(start, source["max_start"]))
        timeline = np.arange(33, dtype=np.int64)
        if spec["kind"] == "reverse":
            timeline = timeline[::-1]
        if spec["kind"] == "static":
            timeline = np.zeros(33, dtype=np.int64)
        if spec["kind"] == "hold":
            timeline = (timeline // 16) * 16
        vi = (start + timeline[::4]).tolist()
        mi = (start + timeline[1:]).tolist()
        if spec["kind"] == "identity":
            video = item["human_video"]
            motion = item["human_future_action"]
        else:
            with h5py.File(source["path"], "r") as f:
                # Gather from the small contiguous segment; repeated/reverse indices are legal.
                low = min(vi + mi)
                high = max(vi + mi) + 1
                rgb = f["cam_data/human_camera"][low:high][np.array(vi) - low]
                pose = f["transformed_hand_frames"][low:high][np.array(mi) - low]
                coords = f["transformed_hand_coords"][low:high][np.array(mi) - low]
            video = (
                _resize_video(rgb, dataset.image_height, dataset.image_width)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            raw = np.concatenate(
                [
                    pose.astype(np.float32).reshape(32, -1),
                    coords.astype(np.float32).reshape(32, -1),
                ],
                axis=-1,
            )
            assert raw.shape == (32, 84)
            motion = torch.from_numpy(
                ((raw - dataset.human_action_mean) / dataset.human_action_std).copy()
            )
            if spec.get("modalities") == "video":
                motion = item["human_future_action"]
                mi = (original_start + np.arange(1, 33)).tolist()
            if spec.get("modalities") == "motion":
                video = item["human_video"]
                vi = (original_start + np.arange(0, 33, 4)).tolist()
        videos.append(video)
        motions.append(motion)
        info.append(
            dict(
                spec,
                source_path=source["path"],
                source_group=source["group"],
                source_content_sha256=source["human_rgb_content_sha256"],
                requested_start=requested,
                actual_start=start,
                start_clamped=start != requested,
                video_frame_indices=vi,
                motion_frame_indices=mi,
                modalities_replaced=spec.get("modalities", "both") if name != "ID" else "neither",
            )
        )
    return torch.stack(videos), torch.stack(motions), info
