# Release validation

The expanded source passed 60 CPU regression tests covering the model, tensor
geometry, default parameter count, forward/backward behavior, visibility masks,
online inference's current-frame boundary, data alignment, checkpoint contracts,
controlled-model modality isolation and candidate-action separation. Three of
these tests start separate interpreters for the carrier, arm and language models
and check finite gradients and hidden-context poisoning invariance.
Two additional hash tests cover empty and multi-block files with the Python 3.11
`file_digest` API disabled, verifying the Python 3.10-compatible export path.
This is not a claim that the complete suite was run under Python 3.10.

The isolated context-ablation package passed its separate 86-test suite. The
archived factorial graph check passed four categories: the precise two-edge mask
change, direct-graph equivalence after absent-carrier removal, nonvacuous
directional prediction independence, and finite gradients in all four cells.

Additional checks passed:

- The tiny flow-generation example produced finite video latents and 32 robot states.
- Training, data-generation, evaluation and aggregation entrypoints accepted
  `--help` in their matching package environments.
- A fresh spawned raw-export worker preserved synthetic RGB frame selection and
  contiguous state alignment exactly. Importing the exporter does not start work.
- Gate, bank and ramp executed finite native trajectories without rendering.
  These are simulator interface checks, not trained-policy success evaluations.
- The theory preflight passed gradient checks, explicit population-objective
  equivalence and exact save/reload next-update checks. It made 218 diagnostic
  CPU optimizer updates; no formal training matrix was launched.
- The primary Python package built successfully as a wheel. Isolated protocol
  packages and experiment scripts are distributed in the source archive, not
  installed as overlapping `fasterwam` wheels.

The local checks used Python 3.13.1, PyTorch 2.8.0, NumPy 2.4.4 and MuJoCo 3.14.0.
No CUDA training, full VAE-weight execution, simulator rendering or complete
paper-result reproduction was performed for this release. Simulator-version
changes can affect numerical trajectories; these checks do not certify equality
with the historical experiment environment.

Release-file integrity and anonymity are checked separately by
`scripts/audit_release.py` against `RELEASE_MANIFEST.json`. A private denylist was
also applied outside the release. The audit checks known identifier/credential
patterns and Git metadata; it does not promise anonymity against every possible
semantic inference or future account activity.
