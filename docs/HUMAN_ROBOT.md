# Human–robot data and training

## Data contract

No recordings or annotation files are distributed in this release. To run the
recorded-data trainer, supply temporally aligned HDF5 episodes and JSONL
annotations with the following schema:

| HDF5 key | Shape / role |
| --- | --- |
| `cam_data/human_camera` | `[T,H,W,3]`, uint8 RGB |
| `cam_data/robot_camera` | `[T,H,W,3]`, uint8 RGB |
| `transformed_hand_frames` | `[T,4,3]`, human wrist-frame vectors |
| `transformed_hand_coords` | `[T,24,3]`, human hand coordinates |
| `end_position` | `[T,6]`, robot end-effector pose |
| `gripper_state` | `[T]` or `[T,1]`, robot gripper |

All streams in one episode share the same length and time axis. The reader checks
this. Raw robot controller setpoints are not used as human action inputs.
Human motion has 84 dimensions; the robot state has seven.

Annotations contain one JSON object per line with `episode_path` and
`instruction`. Paths must be relative to `--workspace-root`, include `/v1/`, and
match the HDF5 inventory exactly. For example, use
`data/v1/task_a/episode_000.hdf5` with `--data-root data/v1`.

Images are resized to 224 by 384. Frame offsets 0,4,...,32 form a nine-frame
clip; state and hand trajectories use contiguous offsets 1,...,32. Short clips
are padded with the final frame and padding is recorded. Normalization statistics
come only from the training episodes.

## Training

Obtain the Wan2.2 VAE checkpoint separately and put it at
`checkpoints/Wan2.2_VAE_bf16.safetensors`, or pass `--vae-path` explicitly.

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_hr_mot_600m.py \
  --workspace-root . --data-root data/v1 --annotations data/annotations.jsonl \
  --vae-path checkpoints/Wan2.2_VAE_bf16.safetensors \
  --output-dir outputs/full --batch-size 32 --steps 10000 \
  --holdout-episodes 64 --windows-per-episode 256 --seed 0
```

`--batch-size` is per process. Add `--no-human-motion-context` for independently
trained Video. The archived core trainer provides Full and Video; other
independently trained baseline orchestration is outside this release. The model
API supports separate video/motion masks for fixed-weight interventions. Do not
report such interventions as independently trained Action or None baselines.

By default, complete context has probability 0.9; the remaining mass covers seven
joint temporal-block visibility patterns. This is not independent all-video or
all-action dropout. The query remains visible. Checkpoint, optimizer and run
metadata are written locally; do not upload them without separate sanitization.

## Inference

`fasterwam.models.hr_inference.generate_rollout` accepts a model, a frozen encoder,
normalization statistics, normalized human RGB, normalized current robot state,
normalized human motion and the current robot RGB frame. It encodes only the
current robot frame, generates the future latents/states jointly with 20 Euler
steps, decodes RGB and denormalizes robot states. It does not need future target
frames as inputs. The returned RGB is neural output; simulator-rendered RGB in
controlled tasks is a different artifact.

For lower-level modality interventions use `HRMoTFlowModel.generate` with
`human_mask` and `human_motion_mask` of shape `[B,3]`. Position indices remain
fixed when context tokens are hidden. Refer to regression tests for examples.

The CPU smoke command uses a tiny randomly initialized model with synthetic
latents. It tests the software interface, not pretrained VAE behavior or paper
accuracy. Full RGB inference needs the VAE weights and a trained model.
