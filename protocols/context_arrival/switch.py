"""Only modify context; use original flow integrator, no latent resets."""

import hashlib
import torch


def sha(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def generate_switched(
    model,
    *,
    video_on,
    motion_on,
    poison_hidden=False,
    correction_at=None,
    correct_human=None,
    video_off=20,
    **inputs,
):
    assert (
        inputs["num_inference_steps"] == 20 and video_on in [0, 10, 20] and motion_on in [0, 10, 20]
    )
    assert video_off in [10, 20] and correction_at in [None, 0, 10, 15, 20]
    original = model.predict_velocity
    history = []
    batch = inputs["robot_current_latent"].shape[0]

    def intercept(**kw):
        s = len(history)
        v = video_on <= s < video_off
        m = s >= motion_on
        corrected = correction_at is not None and s >= correction_at
        if corrected:
            assert correct_human is not None
            kw["human_latents"] = correct_human
        kw["human_mask"] = torch.full_like(inputs["human_mask"], v)
        kw["human_motion_mask"] = torch.full_like(inputs["human_motion_mask"], m)
        if poison_hidden:
            if not v:
                kw["human_latents"] = kw["human_latents"] * 3 + 1
            if not m:
                kw["human_future_action"] = kw["human_future_action"] * 3 + 1
        history.append(
            [
                dict(
                    update_index=s,
                    video_visible=v,
                    motion_visible=m,
                    correct_video_content=corrected,
                    pre_video_sha256=sha(kw["noisy_robot_future_latents"][i]),
                    pre_state_sha256=sha(kw["noisy_robot_future_action"][i]),
                    t_video=float(kw["timestep_video"][i]),
                    t_action=float(kw["timestep_action"][i]),
                )
                for i in range(batch)
            ]
        )
        return original(**kw)

    model.predict_velocity = intercept
    try:
        result = model.generate(**inputs)
    finally:
        model.predict_velocity = original
    assert len(history) == 20
    return result, [[history[s][i] for s in range(20)] for i in range(batch)]
