# Source coverage and reproduction boundaries

The release scope is the core method and experiment families in the current
manuscript. It is not an export of the entire research workspace. Superseded toy
experiments, private communications, manuscripts, remote job owners, monitoring
scripts, logs and original Git histories are intentionally outside this scope.

| Experiment family | Released source | Inputs still required |
| --- | --- | --- |
| Main Full and independent Video | `src/fasterwam`, `scripts/train_hr_mot_600m.py`, `protocols/video_evaluation` | Aligned recordings, annotations, training split, VAE and final weights |
| Independent Action and None | `protocols/carrier_baselines` | Same recordings and split; independent final weights |
| Fixed Full masks, wrong demonstrations, lighting, temporal prefix | `protocols/carrier_baselines/{full_evaluation,lighting,prefix}` | Frozen Full weights and eligible query manifest |
| Hand/arm-only video, with or without motion | `protocols/arm_video` | Reviewed segmentation masks and their approval manifest; exact mask generation assets are not redistributed |
| Future-video/action coupling × demonstration exposure | `protocols/factorial` | Common initialization checkpoint, matched data and final weights |
| Additional context-dropout and initialization controls | `protocols/context_ablation` | Data and initialization specified for that control; not a substitute for the factorial protocol |
| Late-arriving demonstrations | `protocols/context_arrival` | 83 eligible query windows, frozen Full weights, VAE and metric weights |
| Caption-only vs None | `protocols/language` | Captions, segment manifest, raw recordings, training-only normalization statistics and eligibility manifest |
| Main 50 / late-arrival 83 / language 559 cohort checks | `protocols/cohorts/recompute.py`, `protocols/language/recompute_eligible.py` | Saved predictions and corresponding query identities; these scripts do not regenerate predictions |
| Fixed/aimed throwing; fixed/adaptive rebound and pushing | `experiments/{throwing,rebound,pushing,adaptive_rebound,adaptive_pushing}` | Synthetic data regenerated locally; CUDA for training |
| Their paired ID/OOD execution scores | `experiments/evaluation` | Fixed final synthetic-task checkpoints and generated test records |
| Original fixed-probe 3D gate/bank/ramp | `experiments/spatial_fixed` | Regenerated shards, frozen manifests, three-seed final checkpoints |
| Paired geometry 3D execution | `experiments/spatial_paired` | Original 16-dimensional-query policies and generated paired test records |
| Adaptive 3D demonstrations and shared-source-context controls | `experiments/adaptive_spatial`, `experiments/spatial` | Fixed training records and regenerated adaptive demonstrations; separate 32-dimensional-query policies |
| Theory width/risk experiment | `experiments/theory` | No private assets; full matrix must be run separately from its diagnostic |

## Important distinctions

The main flow model, historical carrier model, text model and coupling model are
isolated packages. Launch them with `scripts/run_protocol.py`. Installing or
importing one model over another can silently change masks or checkpoint loading;
matching parameter shapes alone is insufficient.

The carrier trainer is a portable extension of the main trainer that exposes the
historical model's video-visibility switch. It preserves the main sampling and
optimization code. It is not a byte-for-byte archive of the original launch
environment. The factorial package contains its own archived training/model
implementation and must use its common-initialization protocol.

The original spatial model takes a 16-dimensional learner query. The shared-source
comparison takes that query plus a 16-dimensional padded source specification.
Its labels and exposure schedule differ from the original fixed-probe trainer;
do not merge these results or compare them as a one-change ablation.

The language evaluator first scores 701 windows. The eligibility recomputation
retains 559 paired windows from 49 episodes and verifies identity, caption end,
query anchor and generation seed. It does not choose windows by prediction error.
The independent None baseline and masked Language weights remain separate rows.

## What inclusion does not establish

Source coverage is not end-to-end reproduction. No full CUDA retraining, private
recording evaluation, original segmentation recreation, pretrained-VAE numerical
parity or full rendered 3D suite was performed during packaging. CPU tests and
entrypoint checks establish narrower properties, recorded in `VALIDATION.md`.

Original recordings, captions, masks, weights, query IDs and raw predictions are
not silently replaced by generated examples. Without those assets, the exact
reported recording-based numbers cannot be independently reproduced from this
archive alone. Synthetic tasks can be regenerated, but exact original hardware
and simulator-version equivalence still requires separate validation.
