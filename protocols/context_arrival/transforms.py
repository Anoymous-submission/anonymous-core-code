SPECS = {
    name: dict(video_on=k, motion_on=k, correction_at=0, video_off=20)
    for name, k in [("none", 20), ("arrive10", 10), ("full", 0)]
}


def transform(x, name, ids):
    assert name in SPECS
    return x.clone(), [
        dict(SPECS[name], kind=name, sample_index=int(i), motion_indices=list(range(32)))
        for i in ids
    ]
