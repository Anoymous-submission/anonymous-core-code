"""Exercise isolated historical packages; a mismatched attention dependency fails here."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("variant", ["carrier_baselines", "arm_video", "language"])
def test_isolated_model_masks_and_gradients(variant):
    root = Path(__file__).resolve().parents[1]
    probe = r"""
import torch
from fasterwam.models.hr_mot import HRMoTConfig, HRMoTFlowModel
torch.set_num_threads(2)
torch.manual_seed(19)
kwargs=dict(latent_height=4,latent_width=4,num_layers=2,num_heads=2,
    attn_head_dim=96,video_hidden_dim=64,video_ffn_dim=128,
    action_hidden_dim=32,action_ffn_dim=64,gradient_checkpointing=False)
is_text="num_text_tokens" in HRMoTConfig.__dataclass_fields__
if is_text: kwargs.update(num_text_tokens=4,text_vocab_size=12)
c=HRMoTConfig(**kwargs);m=HRMoTFlowModel(c).eval()
with torch.no_grad():
    for p in m.parameters():
        if torch.count_nonzero(p)==0: p.normal_(0,.02)
x=dict(robot_current_latent=torch.randn(2,1,48,4,4),
    human_latents=torch.randn(2,3,48,4,4),
    noisy_robot_future_latents=torch.randn(2,2,48,4,4),
    noisy_robot_future_action=torch.randn(2,32,7),
    timestep_video=torch.ones(2)*.4,timestep_action=torch.ones(2)*.6,
    human_mask=torch.zeros(2,3,dtype=torch.bool),
    human_motion_mask=torch.zeros(2,3,dtype=torch.bool),
    robot_current_state=torch.randn(2,7),human_future_action=torch.randn(2,32,84))
if is_text:
    x.update(text_token_ids=torch.ones(2,4,dtype=torch.long),text_valid=torch.zeros(2,4,dtype=torch.bool))
a=m.predict_velocity(**x)
poison=dict(x,human_latents=x['human_latents']*3+5,human_future_action=x['human_future_action']*3+5)
if is_text: poison['text_token_ids']=x['text_token_ids']+2
b=m.predict_velocity(**poison)
assert all(torch.isfinite(t).all() for t in a)
assert all(torch.equal(u,v) for u,v in zip(a,b))
# Each stream remains independently maskable while the other is available.
for hidden in ('video', 'motion'):
    trial=dict(x, human_mask=torch.full((2,3),hidden!='video',dtype=torch.bool),
               human_motion_mask=torch.full((2,3),hidden!='motion',dtype=torch.bool))
    key='human_latents' if hidden=='video' else 'human_future_action'
    with torch.no_grad():
        before=m.predict_velocity(**trial)
        after=m.predict_velocity(**dict(trial, **{key: trial[key]*3+5}))
    assert all(torch.equal(u,v) for u,v in zip(before,after)), hidden
sum(t.square().mean() for t in a).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "protocols" / variant / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe], env=env, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stderr
