# Protocol entrypoints

Run commands from the repository root. Install `.[simulation,metrics,test]` for
all optional dependencies, with mutually compatible PyTorch and torchvision
builds. Each `--help` command is read-only. Training requires user-provided data
and CUDA; no launcher in this release rents hardware or starts a remote queue.

```bash
python scripts/run_protocol.py carrier_baselines scripts/train_hr_mot_600m.py --help
python scripts/run_protocol.py factorial scripts/train_hr_mot_600m.py --help
python scripts/run_protocol.py factorial eval_extended.py --help
python scripts/run_protocol.py arm_video scripts/train_hr_mot_600m.py --help
python scripts/run_protocol.py language train_pretrain.py --help
python scripts/run_protocol.py context_arrival rollout.py --help
```

## Recording-based variants

Use the data and optimization arguments in `HUMAN_ROBOT.md` for the portable
carrier trainer. Action uses `--no-human-video-context`; None additionally uses
`--no-human-motion-context`. Video uses only `--no-human-motion-context`.
Train each baseline independently with the same seed, split and update budget.

For the arm variant, set `HR_ARM_MASK_ROOT` and `HR_ARM_DATA_ROOT` to local paths.
The reader requires the sibling `MASKS_QA_APPROVED.json` receipt and per-episode
mask files with validated provenance. The trainer without motion uses
`--no-human-motion-context`; retaining motion is a separate comparison. The
release does not supply the original segmentation masks or claim they can be
recreated from the reader alone.

The factorial trainer accepts `--init-checkpoint` and
`--init-checkpoint-sha256`. Use the same None initialization in all four cells.
Demo exposure is either full=0, none=1 or full=0.5, none=0.5 via
`--full-human-context-prob` and `--none-human-context-prob`. Add
`--disable-future-coupling` only to the decoupled cells. Use 10,000 additional
updates and the prescribed learning-rate endpoints 0.00003 and 0.000003.
Run `factorial/test_factorial.py` through the launcher for the graph/gradient
check. This initialization protocol differs from the from-scratch main models.

Frozen evaluators accept local run directories, query manifests, output locations
and expected checkpoint hashes. The historical recording evaluators retain a
single-visible-GPU assertion; set `CUDA_VISIBLE_DEVICES` to one device from 0–3.
Supply cached, separately licensed LPIPS weights as appropriate. Evaluation
output can contain local paths and recording identities and is not public-source
content.

```bash
python scripts/run_protocol.py carrier_baselines full_evaluation/evaluate.py --help
python scripts/run_protocol.py carrier_baselines lighting/evaluate.py --help
python scripts/run_protocol.py carrier_baselines prefix/evaluate.py --help
python scripts/run_protocol.py video_evaluation evaluate.py --help
```

Late context uses `--protocol 8plus24` or `--protocol 32plus96` and the `none`,
`arrival`, or `full` domain. The input query file contains batches of at most four
eligible windows under its historical `scores_all64` key. Later blocks feed back
predicted RGB and state; they do not reset to ground truth.

## Language

Set `VAE_PATH` to the local VAE. `export_rgb.py` takes `--manifest`, `--captions`,
`--stats` and `--out protocols/language/rgb_data`. The supplied statistics must
come from the original training split. The raw RGB/state export is not a latent
cache; VAE encoding remains online. Train `language` and `none` independently
with `train_pretrain.py --mode ...`, then use `evaluate_pretrain.py --mode ...`.

The strict original data contract is 14,634 training and 701 initial test records.
Evaluation writes `evaluation/`; preserve it as `evaluation_final/` before running
the final eligibility recomputation. Provide
`native_token_efficiency/test_record_snapshot.json` with the original `test_records`
identity rows, and the main query manifest with its `known_overlap` flags:

```bash
python scripts/run_protocol.py language recompute_eligible.py --base protocols/language --queries LOCAL_QUERIES.json --out LOCAL_LANGUAGE_RESULTS.json
```

## Synthetic execution

Existing fixed 2D commands are in `SIMULATION.md`. Generate their fixed data
before invoking `experiments/adaptive_rebound/generate.py` or
`experiments/adaptive_pushing/generate.py`; those programs replace only source
controls and RGB and assert unchanged learner records. Train the adaptive models
in their own directories. The evaluation scripts use `--base experiments`,
`--task`, `--out` and `--workers`; each expects the final checkpoints and matching
stored test predictions from its trainer. Use only `--task rebound` with
`adaptive_rebound.py`, `--task pushing` with `adaptive_pushing.py`, and
`--task throw` with `aimed_throwing.py`.

For original fixed 3D, use `spatial_fixed/generate_data.py`, 32 ordered training
shards per task, 64 families per shard, and seed bases gate=2309241000,
bank=2309242000, ramp=2309243000 with shard index added. Freeze each task with
`freeze_data.py`, then merge its three freezes into
`experiments/spatial_fixed/FORMAL_FREEZE.json` using `merge_freezes.py`.
Train four modes and three seeds with `train_policy.py`; output runs must be
`spatial_fixed/runs/TASK_MODE_seedSEED/final.pt` for the paired evaluator.

`spatial_paired/pipeline.py --smoke` precedes its formal invocation without that
flag. Both require trained policies. The paired pipeline performs generation,
prediction, native/finer execution and paired uncertainty; it is not a quick
CPU-only smoke test.

For shared-source 3D, run `adaptive_spatial/generate_adaptive.py TASK`, then
`spatial/prepare.py TASK`, after the original fixed shards and paired evaluation
data exist. These use hard links and require a shared filesystem. Train with
`spatial/train_policy.py` against each policy's new `FORMAL_FREEZE.json`, writing
`spatial/TASK/runs/{fixed,adaptive}_MODE_seed0/final.pt`. Train the None control
once on the fixed view. `spatial/evaluate.py TASK` scores both source policies
and shares that same None checkpoint. The historical execution pool uses up to
128 CPU workers; inspect and adjust resource settings before a full run.

The raw-data contracts and freeze hashes must be regenerated for this portable
source release. Original machine-specific freeze files must not be copied into
the public repository.

### Independent context arrival and arm-only input

`context_arrival/rollout.py` accepts independent `--video-domain` and
`--motion-domain` values (`none`, `arrival`, `full`). Arrival means hidden in
block 0 and available from block 1. For example, `--video-domain full
--motion-domain arrival` keeps video available while motion arrives later.
`--domain` remains the shared default, and `--baseline-type None|Action|Video|Full`
provides input presets; explicit stream switches override those presets.
These are input interventions, not training a new baseline. Use each independently
trained checkpoint for baseline comparisons. The script records checkpoint identity,
implementation package, effective schedules and per-block visibility.

Use `--model-package carrier_baselines` for checkpoints trained with that isolated
package; the default `main` uses the main model. Strict configuration and state-dict
loading remain enabled. Enabling a stream disabled in training raises an error.
Output folders include the preset and both stream schedules.

The `arm_video` trainer supports `--no-human-motion-context` for arm-video-only
training; omit it for arm video plus motion. Both `HR_ARM_MASK_ROOT` and
`HR_ARM_DATA_ROOT` are mandatory, together with the existing mask QA approval and
coverage checks. Missing masks cannot silently become complete-scene video.
These implementation repairs do not regenerate or replace previously reported scores.
