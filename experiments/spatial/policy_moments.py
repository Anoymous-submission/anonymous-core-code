"""Learned nominal skill plus optional response-conditioned action correction."""

import torch
from torch import nn

QDIM = 32
MDIM = 12
ADIM = 6


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(16, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, ADIM),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.LayerNorm(128),
        )
        self.position = nn.Parameter(torch.zeros(16, 128))
        layer = nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.0, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ordered_readout = nn.Sequential(
            nn.Linear(16 * 128, 256), nn.GELU(), nn.Linear(256, 128)
        )
        self.change_readout = nn.Sequential(nn.Linear(16 * 6, 128), nn.GELU(), nn.Linear(128, 128))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 96), torch.linspace(-1, 1, 96), indexing="ij")
        self.register_buffer(
            "image_moments", torch.stack([xx, yy, xx * xx, yy * yy, xx * yy]).flatten(1)
        )
        self.motion = nn.Sequential(nn.Linear(MDIM, 64), nn.GELU(), nn.Linear(64, 64))
        self.residual = nn.Sequential(
            nn.Linear(QDIM + 128 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, ADIM),
        )
        nn.init.normal_(self.residual[-1].weight, std=0.001)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, query, motion, video, present, mode):
        base = self.base(query[:, :16])
        p = present.float().reshape(-1, 1)
        v = query.new_zeros(len(query), 128)
        m = query.new_zeros(len(query), 64)
        if mode in ("video", "full"):
            # Frame differences are computed online from the raw RGB batch.
            # Absolute first frame is retained in slot 0 for source-state context.
            delta = video - video[:, :1]
            # Fixed generic moments of RGB changes, computed online. No masks,
            # coordinates, physics labels, thresholds or object detector inputs.
            energy = delta.square().sum(2).flatten(2)
            total = energy.sum(-1, keepdim=True)
            moments = (energy @ self.image_moments.T) / total.clamp_min(1e-8)
            statistics = torch.cat([moments, torch.log1p(total)], dim=-1)
            delta = delta.clone()
            delta[:, 0] = video[:, 0]
            frames = self.cnn(delta.flatten(0, 1)).reshape(len(query), 16, 128)
            v = (
                self.ordered_readout(self.temporal(frames + self.position).flatten(1))
                + self.change_readout(statistics.flatten(1))
            ) * p
        if mode in ("motion", "full"):
            m = self.motion(motion) * p
        correction = self.residual(torch.cat([query, v, m], dim=-1))
        # Shared source specification remains active when A/V are absent.
        return base, correction


def action(base, correction):
    return base + correction
