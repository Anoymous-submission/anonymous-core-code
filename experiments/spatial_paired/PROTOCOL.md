# Paired geometry evaluation

Freeze policies at update 4096. For each task, generate 128 families, two
physical siblings and four learner queries. Reuse each source demonstration
across ID, factor 1, factor 2, and both-factor geometry conditions. This yields
4096 records per task. The first smoke run uses two families and a separate seed.

Use the original 16-dimensional learner-query policy in `../spatial_fixed/`.
Run all four independently trained modalities and three training seeds. Also
mask the fixed Full policy, and evaluate nominal and numerical-reference actions.
Execute commands at both native and finer timesteps. Retain contact failures,
misses and numerical-reference failures. Report paired uncertainty with training
seeds and correlated within-family queries explicitly represented.

The geometry ranges and RNG seeds are fixed in `pipeline.py`. Do not replace this
pipeline with the source-aware 32-dimensional policy in `../spatial/`; that is a
separate comparison with a different conditioning schema and exposure protocol.
