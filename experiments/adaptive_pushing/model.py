import torch
from torch import nn
from language import VOCAB, MAX_TOKENS

MODES = ["none", "v", "m", "l", "vm", "vl", "ml", "vml"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(512, 48),
            nn.LayerNorm(48),
        )
        self.video = nn.Sequential(nn.Linear(32 * 48, 128), nn.GELU(), nn.Linear(128, 96))
        self.motion = nn.Sequential(nn.Linear(81 * 3, 96), nn.GELU(), nn.Linear(96, 64))
        self.words = nn.Embedding(len(VOCAB), 16, padding_idx=0)
        self.language = nn.Sequential(nn.Linear(MAX_TOKENS * 16, 96), nn.GELU(), nn.Linear(96, 64))
        self.fuse = nn.Sequential(
            nn.Linear(96 + 64 + 64 + 9, 192), nn.GELU(), nn.Linear(192, 128), nn.GELU()
        )
        self.action = nn.Linear(128, 1)
        self.future = nn.Sequential(nn.Linear(129, 128), nn.GELU(), nn.Linear(128, 32))

    def forward(self, video, motion, tokens, query, probe, mode):
        b = len(query)
        v = (
            self.video(self.cnn(video.flatten(0, 1)).reshape(b, -1))
            if "v" in mode
            else query.new_zeros(b, 96)
        )
        m = self.motion(motion.flatten(1)) if "m" in mode else query.new_zeros(b, 64)
        l = self.language(self.words(tokens).flatten(1)) if "l" in mode else query.new_zeros(b, 64)
        z = self.fuse(torch.cat([v, m, l, query], -1))
        action = 0.35 + 1.85 * torch.sigmoid(self.action(z))
        future = query[:, :2, None].transpose(1, 2) + self.future(
            torch.cat([z, probe], -1)
        ).reshape(b, 16, 2)
        return action, future
