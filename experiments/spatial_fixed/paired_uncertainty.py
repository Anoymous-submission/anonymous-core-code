"""Paired success uncertainty with explicit seed/family sampling units.

Input axes: training seed, independent test family, correlated records within
family. Crossed sampling uses one family resample shared across selected seeds.
The nested option is exposed for design diagnostics, not silently substituted.
"""

import numpy as np


def paired_interval(left, right, *, replicates=10000, seed=2309290000, method="crossed"):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Expected aligned seed/family/record arrays")
    if not np.isin(left, [0.0, 1.0]).all() or not np.isin(right, [0.0, 1.0]).all():
        raise ValueError("Success must be binary")
    if min(left.shape) < 1:
        raise ValueError("No empty sampling units")
    difference = (left - right).mean(axis=2)
    ns, nf = difference.shape
    rng = np.random.default_rng(seed)
    seed_ids = rng.integers(ns, size=(replicates, ns))
    if method == "crossed":
        family_ids = rng.integers(nf, size=(replicates, nf))
        samples = difference[seed_ids[:, :, None], family_ids[:, None, :]].mean(axis=(1, 2))
    elif method == "nested":
        family_ids = rng.integers(nf, size=(replicates, ns, nf))
        samples = difference[seed_ids[:, :, None], family_ids].mean(axis=(1, 2))
    else:
        raise ValueError(f"Unknown resampling design: {method}")
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(
        estimate=float(difference.mean()),
        low=float(low),
        high=float(high),
        per_seed=difference.mean(axis=1).tolist(),
        training_seeds=ns,
        families=nf,
        records_per_family=left.shape[2],
        replicates=replicates,
        bootstrap_seed=seed,
        method=method,
    )


def pack_success(archives):
    """Preserve every family/sibling/query cell and check alignment across seeds."""
    packed = []
    identity = None
    for archive in archives:
        family = np.asarray(archive["family"])
        sibling = np.asarray(archive["sibling"])
        query = np.asarray(archive["query_index"])
        order = np.lexsort((query, sibling, family))
        key = np.stack([family[order], sibling[order], query[order]], axis=1)
        if identity is None:
            identity = key
        elif not np.array_equal(identity, key):
            raise ValueError("Seed evaluations have different task identities")
        if len(np.unique(key, axis=0)) != len(key):
            raise ValueError("Duplicate task records")
        fs = np.unique(family)
        expected = np.array([(f, s, q) for f in fs for s in range(2) for q in range(4)])
        if not np.array_equal(key, expected):
            raise ValueError("Missing family/sibling/query records")
        packed.append(np.asarray(archive["success"])[order].reshape(len(fs), 8))
    return np.stack(packed), identity
