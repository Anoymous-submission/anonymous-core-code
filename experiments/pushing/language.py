"""Idealized observable narration; no material parameter or target action access."""

import re
import numpy as np

WORDS = "pad the blue pusher moves slowly steadily quickly after half a second red block has traveled between and centimeters it stops from its starting point".split()
VOCAB = {w: i for i, w in enumerate(WORDS + [str(i) for i in range(0, 201, 5)])}
MAX_TOKENS = 48


def caption(motion, states):
    # Quantized observations from simulator-tracked visible bodies, not hidden mu.
    speed = np.linalg.norm(motion[20, :2] - motion[10, :2]) / 0.04
    adverb = "slowly" if speed < 0.8 else "steadily" if speed < 1.0 else "quickly"
    distances = [float(states[10, 0] - states[0, 0]), float(states[-1, 0] - states[0, 0])]
    bins = [int(np.floor(max(0, x) / 0.05)) * 5 for x in distances]
    assert all(0 <= b < 200 for b in bins)
    text = f"The blue pusher moves {adverb}. After half a second the red block has traveled between {bins[0]} and {bins[0]+5} centimeters. It stops between {bins[1]} and {bins[1]+5} centimeters from its starting point."
    tokens = [VOCAB[w] for w in re.findall(r"[a-z]+|[0-9]+", text.lower())]
    assert len(tokens) <= MAX_TOKENS
    return text, np.array(tokens + [0] * (MAX_TOKENS - len(tokens)), np.int64)
