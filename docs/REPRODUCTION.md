# Reproduction scope and modifications

This release provides inspectable core software and runnable local pipelines.
It is not a claim that every reported paper number can be reproduced solely from
this archive. Private aligned recordings, trained checkpoints, original test IDs,
segmentation assets, and historical result artifacts are not redistributed.

## Included

- The archived main flow-model computation, online VAE interface, reader and
  inference modules. The trainable default parameter count is checked by tests.
- Full/Video and portable independent Action/None training, with isolated model
  versions for historical carriers, coupling, arm-video and caption protocols.
- Synthetic-data generation and training for fixed/aimed throwing, fixed rebound,
  and fixed/adaptive rebound and friction pushing, plus native execution evaluators.
- All three spatial simulators, expert solvers, fixed-source data generation,
  original fixed-probe and separate shared-source-specification policies, adaptive
  source generation, paired geometry evaluation and matched-budget trainers.
- Fixed-model context interventions, late-arrival inference, language eligibility
  recomputation and common-cohort consistency checks.
- The exact population-risk objective and prescribed width experiment matrix.
- Regression tests and an independent release-content audit.

## Packaging changes

- Machine-specific checkpoint defaults are replaced by a relative path.
- Hostname recording is disabled in the human–robot training entry point.
- Spatial source-query padding is integrated into data generation and bound by
  file hashes; the portable freeze helper replaces machine-specific owners.
- Validation/test spatial generation requires a supplied evaluation plan.
- The throwing wrong-context diagnostic swaps action and video coherently.
- The theory script uses a bundled protocol instead of a private planning file.
- Unused model download helpers are excluded; VAE loading uses local tensor files.
- Added protocol isolation prevents historical model packages from overwriting
  each other. Each variant keeps its matching attention implementation.
- Historical absolute experiment folders become relative release directories.
  Evaluation hashes are supplied explicitly instead of embedding private weights.
- Duplicate `formal_code` directories are replaced by the release manifest and
  per-run source hashes. Original sources remain unchanged outside this export.
- Adaptive spatial immutable data links are hard links so manifest validation
  does not follow symlinks outside the data directory.
- Language raw export takes explicit training statistics instead of reading a
  private checkpoint; its workers use a spawn-safe entrypoint. Eligibility rules,
  prediction metrics and bootstrap sampling are preserved.
- Raw-export checksums use a streaming SHA-256 helper compatible with the stated
  Python 3.10 minimum; file handles are closed after hashing.
- Formatting, documentation, package metadata and small test/example entry points
  are cleaned. Third-party licenses and dependency attribution are preserved.

The mathematical model definitions and simulator behavior are preserved. Public
source hashes refer to this sanitized release; private source mappings are not
part of the repository. CPU smoke checks do not establish GPU training accuracy,
full pretrained-VAE reconstruction, or numerical equivalence across MuJoCo versions.

See `COVERAGE.md` for the experiment-by-experiment source and external-asset map.
See `PROTOCOLS.md` before using the historical pipelines; some are expensive
full-suite programs even when their option is named `--smoke`.

Independently trained modality comparisons are distinct from masking a fixed
Full model. Terminal execution success is distinct from future prediction error.
The controlled tasks are bounded simulations; no claim of real-robot closed-loop
transfer or universal visual generalization follows from these implementations.
