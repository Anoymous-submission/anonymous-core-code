"""Predict actions from allowed inputs only; execute and score in another process.

This module does not import a simulator or read expert/physics/source-response
arrays. Standalone None always enables its learned target-only residual.
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from policy_moments import Policy
from action_limits import decode
from data_contract import dataset_contract, source_contract


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--context", choices=["matching", "absent", "wrong"], required=True)
    p.add_argument(
        "--source-override", help="Directory with aligned variant raw source RGB and imposed motion"
    )
    p.add_argument("--allow-development", action="store_true")
    p.add_argument("--condition")
    p.add_argument("--split", choices=["validation", "test"])
    p.add_argument(
        "--evaluation-plan", default=str(Path(__file__).with_name("EVALUATION_PLAN.json"))
    )
    args = p.parse_args()
    checkpoint = Path(args.checkpoint)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run = ckpt["args"]
    task = run["task"]
    mode = run["mode"]
    development = bool(run.get("development_steps", 0))
    assert args.allow_development or (not development and ckpt["step"] == 4096)
    contract = dataset_contract(
        data,
        task,
        formal=not args.allow_development,
        plan_path=args.evaluation_plan,
        condition=args.condition,
        split=args.split,
    )
    if not args.allow_development:
        context_by_condition = {
            "id_matching": "matching",
            "id_absent": "absent",
            "id_wrong_physics": "wrong",
            "id_changed_source_command": "matching",
            "source_appearance": "matching",
            "geometry_matching": "matching",
            "geometry_absent": "absent",
        }
        assert args.context == context_by_condition[args.condition]
        assert bool(args.source_override) == (
            args.condition in ("id_changed_source_command", "source_appearance")
        )
    model = Policy().cuda().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    with np.load(data / "records.npz", allow_pickle=False) as r:
        queries = r["query"].astype(np.float32)
        source_ids = r["source_id"].astype(np.int64)
        family = r["family"].copy()
        siblings = r["sibling"].copy()
        query_index = r["query_index"].copy()
    assert np.array_equal(source_ids, 2 * family + siblings)
    assert len(queries) == contract["records"]
    inputs = Path(args.source_override) if args.source_override else data
    variant = None
    if args.source_override:
        variant = source_contract(
            inputs,
            contract,
            task,
            condition=args.condition,
            context=args.context,
            plan_path=args.evaluation_plan,
            formal=not args.allow_development,
        )
    rgb = np.load(inputs / "source_rgb.npy", mmap_mode="r", allow_pickle=False)
    motion = np.load(inputs / "source_motion.npy", allow_pickle=False)
    assert len(rgb) == len(motion) == 2 * len(np.unique(family))
    ids = source_ids ^ 1 if args.context == "wrong" else source_ids
    model_mode = "target_only" if mode == "none" else mode
    raw = []
    with torch.inference_mode():
        for start in range(0, len(queries), 64):
            end = min(start + 64, len(queries))
            selected = ids[start:end]
            n = end - start
            q = torch.as_tensor(queries[start:end], device="cuda")
            visible = args.context != "absent" and mode != "none"
            if visible and mode in ("video", "full"):
                v = (
                    torch.as_tensor(np.array(rgb[selected]), device="cuda")
                    .permute(0, 1, 4, 2, 3)
                    .float()
                    / 255
                )
            else:
                v = torch.zeros((n, 16, 3, 96, 96), device="cuda")
            if visible and mode in ("motion", "full"):
                m = torch.as_tensor(motion[selected] / 8, dtype=torch.float32, device="cuda")
            else:
                m = torch.zeros((n, 12), device="cuda")
            # Standalone None is a learned population controller, not the base-only ablation.
            present = torch.full((n,), mode == "none" or visible, device="cuda")
            base, residual = model(q, m, v, present, model_mode)
            raw.append(((base + residual) * 8).cpu().numpy())
    raw = np.concatenate(raw)
    executed = decode(raw, task)
    clipped = np.any(raw[:, : executed.shape[1]] != executed, axis=1)
    np.savez_compressed(
        out / "ACTIONS.npz",
        raw=raw,
        executed=executed,
        clipped=clipped,
        family=family,
        sibling=siblings,
        query_index=query_index,
        source_id=ids,
    )
    proof = dict(
        task=task,
        mode=mode,
        context=args.context,
        records=len(queries),
        checkpoint_step=ckpt["step"],
        training_seed=run["seed"],
        development=development or args.allow_development,
        formal_updates=0,
        policy_weights_updated=False,
        dataset_contract=contract,
        source_contract=variant,
        condition=args.condition,
        split=contract["split"],
        standalone_none_residual_enabled=mode == "none",
        source_override=args.source_override,
        checkpoint_path=str(checkpoint.resolve()),
        data_path=str(data.resolve()),
        source_path=str(inputs.resolve()),
        checkpoint_sha=sha(checkpoint),
        record_file_sha=sha(data / "records.npz"),
        source_files={name: sha(inputs / name) for name in ("source_rgb.npy", "source_motion.npy")},
        action_file_sha=sha(out / "ACTIONS.npz"),
        clipped_records=int(clipped.sum()),
        allowed_arrays_read=[
            "query",
            "source_id",
            "family",
            "sibling",
            "query_index",
            "source_rgb",
            "source_motion",
        ],
    )
    (out / "PREDICTION.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
