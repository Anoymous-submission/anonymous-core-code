"""Contiguous latent-prefix availability; does not alter RGB targets."""

SPECS = {
    "video_none": {"mask": [False, False, False], "human_last_step": None},
    "z0": {"mask": [True, False, False], "human_last_step": 0},
    "z01": {"mask": [True, True, False], "human_last_step": 16},
    "z012": {"mask": [True, True, True], "human_last_step": 32},
}


def transform(x, name, ids):
    return x.clone(), [dict(SPECS[name], sample_index=int(i)) for i in ids]
