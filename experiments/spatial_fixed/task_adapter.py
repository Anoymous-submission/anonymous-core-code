"""Explicit interfaces for three native 3D tasks; no use in policy forward."""

import numpy as np
import gate_physics as gate
import bank_physics as bank
import ramp_physics as ramp
import ramp_expert_v2
import bank_expert_v2

ACTION_DIMS = dict(gate=6, bank=3, ramp=2)
QUERY_DIMS = dict(gate=10, bank=12, ramp=12)


def planned_gate_action(q, wind, drag):
    # The prescribed gate tolerance permits a small waypoint displacement.
    # Plan geometric rim clearance analytically, then execute the original task.
    # The finer native simulator is not used to select these actions.
    offsets = [
        (0, 0, 0),
        (0.10, 0, -0.10),
        (0.14, 0, -0.14),
        (0.14, 0, 0),
        (0, 0, -0.18),
        (0.18, 0, -0.08),
        (-0.18, 0, 0),
        (-0.18, 0, -0.08),
        (-0.10, 0, -0.18),
        (-0.05, 0, -0.20),
    ]
    for attempt, offset in enumerate(offsets):
        aim = np.asarray(q, dtype=float).copy()
        aim[3:6] += offset
        action = gate.oracle(aim, wind, drag)
        analytic, _ = gate.recurrence(q[:3], action, q[9], wind, drag, dt=0.001)
        relative = analytic - q[3:6]
        near = np.abs(relative[:, 0]) <= 0.08
        clearance = bool(
            np.all(np.abs(relative[near, 1]) < 0.22) and np.all(np.abs(relative[near, 2]) < 0.48)
        )
        result = gate.simulate(q, action, wind, drag)
        if clearance and result["success"]:
            return (
                action,
                result,
                dict(nfev=attempt + 1, waypoint_offset=list(offset), analytic_clearance=True),
            )
    return (
        action,
        result,
        dict(nfev=len(offsets), waypoint_offset=list(offset), analytic_clearance=False),
    )


def target(task, rng, ood=False):
    if task == "gate":
        q = gate.sample_query(rng)
        if ood:
            q[4] = rng.choice([-1, 1]) * rng.uniform(0.5, 0.8)
            q[7] = rng.choice([-1, 1]) * rng.uniform(0.8, 1.0)
        action, result, _ = planned_gate_action(q, [0, 0, 0], 0.7)
        if not result["success"]:
            raise RuntimeError("Nominal gate query is not feasible")
        return q, action, 0
    module = bank if task == "bank" else ramp
    q, reference, result = module.sample_query(rng, ood=ood)
    if task == "ramp" and (result["crossing"] is None or not result["success"]):
        # sample_query sets the goal after simulation; independently score it.
        result = module.simulate(q, reference, 0.28)
        if not result["success"]:
            raise RuntimeError("Nominal ramp query is not feasible")
    return q, reference, result.get("query_sampling_attempts", 1) - 1


def parameters(task, rng):
    if task == "gate":
        pair = [np.r_[*gate.sample_physics(rng, r)] for r in ("mild", "strong")]
        order = rng.permutation(2)
        return np.array([pair[i] for i in order]), np.array(order)
    if task == "bank":
        return np.array([bank.sample_physics(rng) for _ in range(2)]), np.full(2, -1)
    return np.array([[ramp.sample_physics(rng)] for _ in range(2)]), np.full(2, -1)


def execute(task, q, action, physics, **kwargs):
    if task == "gate":
        return gate.simulate(q, action, physics[:3], physics[3], **kwargs)
    if task == "bank":
        return bank.simulate(q, action, physics[0], physics[1], **kwargs)
    return ramp.simulate(q, action, physics[0], **kwargs)


def expert(task, q, physics, reference):
    if task == "gate":
        return planned_gate_action(q, physics[:3], physics[3])
    if task == "bank":
        return bank_expert_v2.oracle(q, *physics, initial=reference)
    return ramp_expert_v2.oracle(q, physics[0], initial=reference)


def source(task, rng):
    if task == "gate":
        q = np.array([-0.6, 0, 1.5, 0.55, 0.2, 2.7, 1.6, -0.1, 2.2, 1.0])
        q[:3] += rng.uniform(-0.08, 0.08, 3)
        q[4:6] += rng.uniform(-0.1, 0.1, 2)
        a = gate.oracle(q, [0, 0, 0], 0.7)
        motion = np.pad(np.r_[q[:3], a], (0, 3))
    elif task == "bank":
        q = np.r_[
            [0.0, 0.0, 1.5], [0.0, 0.0, 0.8], [1.1, 0.0, 1.5], rng.uniform(-0.08, 0.08, 2), 1.0
        ]
        a = np.array([2.8, 0.0, 5.1]) + rng.uniform(-0.08, 0.08, 3)
        q[3:6] = bank.simulate(q, a, 0.4, 0.1, endpoint_only=True)["positions"][-1]
        motion = np.r_[q[:3], a, q[6:12]]
    else:
        q = np.r_[
            [0.0, 0.0, 2.1], 0.27, rng.uniform(-0.08, 0.08), 0.9, [1.6, 0.0, 1.2], 1.0, -0.675, 0.0
        ]
        a = np.array([3.5, rng.uniform(-0.1, 0.1)])
        initial = ramp.simulate(q, a, 0.28)
        if initial["crossing"] is None:
            raise RuntimeError("Source ramp calibration did not cross")
        q[6:8] = initial["crossing"][:2]
        motion = np.pad(np.r_[q[:6], q[9:12], a], (0, 1))
    assert motion.shape == (12,)
    return q, a, motion


def source_video(task, q, a, physics, style=0):
    opts = dict(render=True, size=96, style=style)
    if task == "gate":
        opts["gated"] = False
    if task == "ramp":
        opts["horizon"] = 0.9
    result = execute(task, q, a, physics, **opts)
    assert result["frames"].shape == (16, 96, 96, 3)
    # These responses are audit outputs, not model inputs or training labels.
    ids = np.round(np.linspace(0, len(result["positions"]) - 1, 16)).astype(int)
    return result["frames"], result["positions"][ids]
