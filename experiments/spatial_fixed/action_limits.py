"""Public actuator limits; inference decoding, no simulator or hidden state."""

import numpy as np

BOUNDS = {
    "gate": (np.full(6, -12.0), np.full(6, 12.0)),
    "bank": (np.array([1.5, -2.0, 2.0]), np.array([5.5, 2.0, 8.0])),
    "ramp": (np.array([1.0, -1.2]), np.array([5.5, 1.2])),
}


def decode(raw, task):
    lo, hi = BOUNDS[task]
    raw = np.asarray(raw)[..., : len(lo)]
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite policy action")
    return np.clip(raw, lo, hi)
