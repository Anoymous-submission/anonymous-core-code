# Population-risk width experiment

Optimize the exact population objective
`0.5 * ||G - F B.T W B||_F^2`, with `B B.T = I` and bounded spectral norm of W.
A positive-diagonal QR parameterization fixes the basis convention. Float64 CPU
Adam uses a cosine learning-rate schedule and projects W after every update.

The prespecified matrix contains four source settings, widths 1–4, three seeds,
and additional norm-cap comparisons, for 60 runs. Each run has 4,000 updates.
The preflight checks automatic differentiation, the explicit attention formula,
QR invariants, and checkpoint reload. It is diagnostic and supplies no training
initialization to formal runs. Failed runs and checkpoints are retained.

```bash
python experiments/theory/run.py --preflight
python experiments/theory/run.py --stage sanity
python experiments/theory/run.py --stage all
```

`sanity` is a three-run subset. A subsequent `all` reuses completed matching runs.
Risk is a population objective, not performance on private recordings or a
trained robotics policy. Results are written under ignored `runs` and `evidence`.
