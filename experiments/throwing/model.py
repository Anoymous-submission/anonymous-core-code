import torch
from torch import nn


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
        self.video = nn.Sequential(nn.Linear(32 * 48, 128), nn.GELU(), nn.Linear(128, 128))
        self.motion = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 32))
        self.fuse = nn.Sequential(
            nn.Linear(128 + 32 + 17, 192), nn.GELU(), nn.Linear(192, 128), nn.GELU()
        )
        self.action = nn.Linear(128, 2)
        self.future = nn.Sequential(nn.Linear(130, 128), nn.GELU(), nn.Linear(128, 48))

    def forward(self, video, motion, query, probe, mode):
        b = len(query)
        v = (
            self.video(self.cnn(video.flatten(0, 1)).reshape(b, -1))
            if mode in ["video", "full"]
            else query.new_zeros(b, 128)
        )
        m = self.motion(motion) if mode in ["motion", "full"] else query.new_zeros(b, 32)
        z = self.fuse(torch.cat([v, m, query], -1))
        action = self.action(z)
        future = query[:, :3, None].transpose(1, 2) + self.future(
            torch.cat([z, probe], -1)
        ).reshape(b, 16, 3)
        return action, future
