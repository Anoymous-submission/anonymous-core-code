"""Run a tiny CPU flow-generation example without private data or VAE weights."""

import json
import torch
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel, count_parameters


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = HRMoTConfig(
        latent_height=4,
        latent_width=6,
        num_layers=2,
        num_heads=2,
        video_hidden_dim=64,
        video_ffn_dim=128,
        action_hidden_dim=32,
        action_ffn_dim=64,
        gradient_checkpointing=False,
    )
    model = HRMoTFlowModel(cfg).eval()
    shape = (cfg.latent_channels, cfg.latent_height, cfg.latent_width)
    video, states = model.generate(
        robot_current_latent=torch.randn(1, 1, *shape),
        human_latents=torch.randn(1, cfg.num_latent_steps, *shape),
        robot_current_state=torch.randn(1, cfg.action_dim),
        human_future_action=torch.randn(1, cfg.num_action_tokens, cfg.human_action_dim),
        num_inference_steps=2,
    )
    assert torch.isfinite(video).all() and torch.isfinite(states).all()
    print(
        json.dumps(
            dict(
                parameters=count_parameters(model),
                future_latent_shape=list(video.shape),
                future_state_shape=list(states.shape),
                synthetic=True,
                trained=False,
            )
        )
    )


if __name__ == "__main__":
    main()
