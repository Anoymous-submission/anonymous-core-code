# Controlled simulations

The simulators use native MuJoCo dynamics and synthetic RGB. Policy inputs exclude
hidden physical coefficients. Data generators and experts can access these
coefficients to construct labels. Model image features are computed online.

Run scripts from the repository root after installation. Choose the appropriate
MuJoCo rendering backend before data generation. Outputs stay in ignored data/run
directories. Commands below start new local runs; existing experiment artifacts
are not included.

## Throwing, rebound and pushing

Generate complete splits before training:

```bash
for split in train val test; do
  python experiments/throwing/generate.py --split "$split"
done
python experiments/throwing/train.py --policy fixed --mode full --seed 0
python experiments/throwing/train.py --policy aimed --mode full --seed 0
```

The same split commands work in `experiments/rebound` and `experiments/pushing`.
Throwing and rebound use `none`, `motion`, `video`, `full`. Pushing uses `none`,
`m`, `v`, `l`, and their combinations (see `MODES` in its model). Commands named
`motion` or `m` correspond to the numerical Action condition. Response-language
is a deterministic template derived from observed source responses, not free-form
human annotation. Each trainer refuses to overwrite an already started run.

Throwing generates matched fixed and aimed source policies. Rebound and pushing
here provide the fixed-source core pipelines. The separate adaptive-source
follow-ups for rebound/pushing are not bundled. The throwing wrong-demo path
swaps numerical commands together with video, so aimed demonstrations remain
coherent; this corrects the original fixed-probe-only diagnostic path.

Throwing/rebound use 16 epochs; pushing uses 32, each yielding 4,096 optimizer
updates on its full generated training split. Validation predictions are retained;
final test predictions are evaluated at the fixed endpoint. `--smoke` in these
archived trainers is a multi-update overfit diagnostic, not a quick CPU check.

Training reports action MSE and future trajectory ADE. Those numbers do not
measure physical execution success. The simulators provide native trajectory
execution: `flight` for throwing and `simulate` for rebound/pushing. Preserve task
contact requirements and terminal position/velocity tolerances when scoring.

## Gate, bank and ramp

`experiments/spatial` contains all three simulator/expert interfaces and the
policy with a common 16D source specification concatenated to the 16D target
query. This is the shared-specification protocol; its `none` mode means Spec-only,
not absence of source specification. The base network receives only the target
query, while the residual also receives the shared source specification. RGB
change moments are computed online and are part of this policy's disclosed inputs.

Generate a small development shard and run a bounded training check:

```bash
python experiments/spatial/generate_data.py --task gate --families 1 \
  --seed 0 --development --out data/gate_example
python experiments/spatial/train_policy.py --task gate --mode full --seed 0 \
  --data experiments/spatial/data/gate_example --development-steps 2 \
  --out outputs/gate_example
```

The data generator creates the padded source specification only from known
source query fields; physics and observed response arrays remain audit-only.
Generation retains failed cases and stops training-data completion when an expert
fails its required checks. Failure is not silently resampled.

For formal fixed-source regeneration use 32 shards of 64 families, with consecutive
seeds starting at 2309241000 (gate), 2309242000 (bank), or 2309243000 (ramp).
Freeze the ordered shard list with `freeze_data.py --task ... --data ... --out ...`
and pass the same absolute paths and `--freeze` to `train_policy.py`. The trainer
checks shard seeds/counts, file digests and code digests before 4,096 updates.
It saves optimizer, scheduler, random states and per-record coverage. Half the
training exposures mask Action/Video, while the common source specification stays
visible. Both nominal and actual-action supervision are present.

Validation/test data generation requires an explicit `--evaluation-plan` file;
no private historical test manifest or model selection is bundled. The adaptive
source-policy orchestration and fixed/adaptive evaluation sources are included
separately; see `PROTOCOLS.md` for their prerequisites and execution order. The
shared-specification policy must not be conflated with the original fixed-probe
model in `experiments/spatial_fixed`.
