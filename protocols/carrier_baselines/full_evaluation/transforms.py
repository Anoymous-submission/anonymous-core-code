"""Fixed input-only human-video interventions; never modify robot RGB/targets."""

import math
import numpy as np
import torch
import torch.nn.functional as F

SPECS = {
    "ID": {"kind": "identity"},
    "exposure075": {"kind": "exposure", "gain": 0.75},
    "exposure050": {"kind": "exposure", "gain": 0.50},
    "exposure025": {"kind": "exposure", "gain": 0.25},
    "gamma_dark": {"kind": "gamma", "gain": 0.65, "gamma": 1.5},
    "crop090": {"kind": "center_crop", "side_fraction": 0.90},
    "crop080": {"kind": "center_crop", "side_fraction": 0.80},
    "crop065": {"kind": "center_crop", "side_fraction": 0.65},
    "blur1": {"kind": "gaussian_blur", "sigma_pixels": 1.0},
    "blur2": {"kind": "gaussian_blur", "sigma_pixels": 2.0},
    "blur4": {"kind": "gaussian_blur", "sigma_pixels": 4.0},
    "occlusion010": {"kind": "occlusion", "area_fraction": 0.10},
    "occlusion025": {"kind": "occlusion", "area_fraction": 0.25},
    "resolution2": {"kind": "resolution", "factor": 2},
    "resolution4": {"kind": "resolution", "factor": 4},
    "static_first": {"kind": "time", "indices": [0] * 9},
    "static_last": {"kind": "time", "indices": [8] * 9},
    "reverse_future": {"kind": "time", "indices": [0, 8, 7, 6, 5, 4, 3, 2, 1]},
    "shuffle_future": {"kind": "shuffle_future"},
    "hold_stride2": {"kind": "time", "indices": [0, 0, 2, 2, 4, 4, 6, 6, 8]},
    "hold_stride4": {"kind": "time", "indices": [0, 0, 0, 0, 4, 4, 4, 4, 8]},
}


def transform(video, name, sample_ids):
    """B,C,T,H,W in [-1,1]; spatial transform is identical across clip time."""
    s = SPECS[name]
    b, c, t, h, w = video.shape
    assert c == 3 and t == 9 and len(sample_ids) == b
    x = video.clone()
    info = []
    if s["kind"] == "identity":
        return x, [dict(s) for _ in sample_ids]
    if s["kind"] in ["exposure", "gamma"]:
        x = ((x + 1) * 0.5).clamp(0, 1).pow(s.get("gamma", 1.0)) * s["gain"] * 2 - 1
    elif s["kind"] in ["center_crop", "resolution", "gaussian_blur"]:
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if s["kind"] == "center_crop":
            nh, nw = round(h * s["side_fraction"]), round(w * s["side_fraction"])
            top, left = (h - nh) // 2, (w - nw) // 2
            y = F.interpolate(
                y[:, :, top : top + nh, left : left + nw],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        elif s["kind"] == "resolution":
            y = F.interpolate(y, size=(h // s["factor"], w // s["factor"]), mode="area")
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        else:
            sigma = s["sigma_pixels"]
            radius = math.ceil(3 * sigma)
            g = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
            g = torch.exp(-g.square() / (2 * sigma * sigma))
            g /= g.sum()
            kernel = (g[:, None] * g[None, :])[None, None].expand(c, 1, -1, -1)
            y = F.conv2d(F.pad(y, (radius,) * 4, mode="reflect"), kernel, groups=c)
        x = y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    elif s["kind"] == "occlusion":
        nh, nw = round(h * math.sqrt(s["area_fraction"])), round(w * math.sqrt(s["area_fraction"]))
        for i, idx in enumerate(sample_ids):
            rng = np.random.default_rng(151500 + int(idx))
            top, left = int(rng.integers(h - nh + 1)), int(rng.integers(w - nw + 1))
            x[i, :, :, top : top + nh, left : left + nw] = 0.0  # fixed mid-gray, not sample-derived
            info.append(
                dict(
                    s,
                    top=top,
                    left=left,
                    height=nh,
                    width=nw,
                    actual_area_fraction=nh * nw / (h * w),
                )
            )
    elif s["kind"] == "time":
        x = x[:, :, s["indices"]]
    elif s["kind"] == "shuffle_future":
        for i, idx in enumerate(sample_ids):
            order = [0] + np.random.default_rng(152500 + int(idx)).permutation(
                np.arange(1, 9)
            ).tolist()
            x[i] = video[i].index_select(1, torch.tensor(order, device=video.device))
            info.append(dict(s, indices=order))
    else:
        raise ValueError(s)
    assert x.shape == video.shape and torch.isfinite(x).all()
    assert x.min() >= -1.00001 and x.max() <= 1.00001
    return x, info or [dict(s) for _ in sample_ids]
