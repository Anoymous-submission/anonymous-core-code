"""Native 6-DOF block contact, inclined-ramp edge departure, then free flight."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
from functools import lru_cache
import mujoco as mj
import mujoco.rollout
import numpy as np
from scipy.optimize import minimize
from bank_physics import rotation
from render_cache import renderer_for

DT = 0.001
MAX_TIME = 1.7
HALF = 0.045
TOL = 0.13
VIEWS = {}


@lru_cache(maxsize=4)
def model(dt=DT, length=0.5, mass=1.0):
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="80"/>
    <visual><global offwidth="512" offheight="512"/><quality shadowsize="2048"/><headlight ambient=".5 .5 .5"/></visual>
    <asset><texture type="skybox" builtin="gradient" rgb1=".35 .55 .7" rgb2=".9 .97 .94" width="512" height="2048"/>
    <texture name="tiles" type="2d" builtin="checker" rgb1=".76 .85 .78" rgb2=".65 .76 .67" width="512" height="512"/>
    <material name="ground" texture="tiles" texrepeat="8 8"/></asset>
    <worldbody><light pos="0 -3 6"/>
      <geom name="floor" type="plane" pos="0 0 -2.5" size="8 8 .1" material="ground"/>
      <body name="block"><freejoint/><geom name="block" type="box" size="{HALF} {HALF} {HALF}" mass="{mass}" rgba=".93 .23 .13 1"/></body>
      <body name="ramp"><geom name="ramp" type="box" size="{length} .6 .04" rgba=".67 .44 .24 1"/>
      <geom name="left_rail" type="box" pos="0 -.64 .07" size="{length} .03 .09" rgba=".24 .35 .4 1"/>
      <geom name="right_rail" type="box" pos="0 .64 .07" size="{length} .03 .09" rgba=".24 .35 .4 1"/></body>
      <body name="goal"><geom type="cylinder" size=".16 .012" rgba="1 .8 .1 .8" contype="0" conaffinity="0"/>
       <geom type="box" size=".012 .012 .22" rgba="1 .8 .1 1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>"""
    return mj.MjModel.from_xml_string(xml)


def simulate(
    query,
    action,
    friction,
    render=False,
    size=96,
    dt=DT,
    horizon=MAX_TIME,
    style=0,
    crossing_only=False,
):
    # query: ramp center XYZ, tilt, yaw, half-length, target XYZ, mass, start local X/Y.
    length = float(query[5])
    mass = float(query[9])
    m = model(float(dt), length, mass)
    d = mj.MjData(m)
    rot = rotation(float(query[4]), float(query[3]))
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, rot.reshape(-1))
    m.body_pos[m.body("ramp").id] = query[:3]
    m.body_quat[m.body("ramp").id] = quat
    m.body_pos[m.body("goal").id] = query[6:9]
    for name in ("block", "ramp"):
        g = m.geom(name).id
        m.geom_friction[g] = [friction, 0.003, 0.0001]
        m.geom_solref[g] = [0.01, 1.0]
        m.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2]
    d.qpos[:3] = query[:3] + rot @ np.array([query[10], query[11], 0.04 + HALF + 0.0005])
    d.qpos[3:7] = quat
    d.qvel[:3] = rot @ np.array([action[0], action[1], 0.0])
    mj.mj_forward(m, d)
    if crossing_only:
        initial = np.r_[0.0, d.qpos.copy(), d.qvel.copy()]
        states, _ = mj.rollout.rollout(m, d, initial, nstep=int(round(horizon / dt)))
        path = np.vstack([initial[1:4], states[0, :, 1:4]])
        local = (path - query[:3]) @ rot
        candidates = np.flatnonzero(
            (path[:-1, 2] >= query[8]) & (path[1:, 2] < query[8]) & (local[1:, 0] > length)
        )
        crossing = None
        if len(candidates):
            k = int(candidates[0])
            alpha = (path[k, 2] - query[8]) / (path[k, 2] - path[k + 1, 2])
            crossing = path[k] * (1 - alpha) + path[k + 1] * alpha
        return dict(positions=path, crossing=crossing)
    n = int(round(horizon / dt))
    sample = set(np.round(np.linspace(0, n, 16)).astype(int))
    path = []
    vel = []
    frames = []
    quats = []
    contact = []
    rail = []
    cross = None
    cross_time = None
    departed = False
    block = m.geom("block").id
    ramp = m.geom("ramp").id
    renderer = None
    if render:
        renderer = renderer_for(VIEWS, m, size)
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.8, 0, 0.8]
        cam.distance = 4.8
        cam.azimuth = 130 + style * 8
        cam.elevation = -25
    for k in range(n + 1):
        pos = d.qpos[:3].copy()
        path.append(pos)
        vel.append(d.qvel.copy())
        quats.append(d.qpos[3:7].copy())
        touching = False
        for c in d.contact:
            ids = {int(c.geom1), int(c.geom2)}
            if ids == {block, ramp}:
                contact.append(k)
                touching = True
            elif block in ids and ramp not in ids and cross is None:
                rail.append(k)
        local = rot.T @ (pos - query[:3])
        if contact and not touching and local[0] > length:
            departed = True
        if (
            k > 0
            and cross is None
            and departed
            and path[-2][2] >= query[8] > pos[2]
            and d.qvel[2] < 0
        ):
            alpha = (path[-2][2] - query[8]) / (path[-2][2] - pos[2])
            cross = path[-2] * (1 - alpha) + pos * alpha
            cross_time = (k - 1 + alpha) * dt
        if renderer is not None and k in sample:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        if k < n:
            mj.mj_step(m, d)
    error = float(np.linalg.norm(cross[:2] - query[6:8])) if cross is not None else float("inf")
    return dict(
        positions=np.array(path),
        velocities=np.array(vel),
        quaternions=np.array(quats),
        frames=np.array(frames, dtype=np.uint8),
        crossing=cross,
        crossing_time=cross_time,
        endpoint_error=error,
        ramp_contact_steps=len(contact),
        other_contact_steps=len(rail),
        departed_downhill_edge=departed,
        success=bool(error <= TOL and contact and departed and not rail),
    )


def oracle(query, friction):
    def residual(u):
        r = simulate(query, u, friction, crossing_only=True)
        if r["crossing"] is None:
            return (r["positions"][-1] - query[6:9])[:2]
        return r["crossing"][:2] - query[6:8]

    fit = minimize(
        lambda u: float(np.sum(residual(u) ** 2)),
        [3.5, 0],
        method="Powell",
        bounds=[(1.0, 5.5), (-1.2, 1.2)],
        options=dict(xtol=1e-5, ftol=1e-8, maxiter=25, maxfev=1000),
    )
    actual = simulate(query, fit.x, friction)
    return fit.x, actual, dict(nfev=fit.nfev, converged=bool(fit.success))


def sample_query(rng, ood=False):
    center = np.array([0, 0, rng.uniform(1.8, 2.2)])
    tilt = float(rng.uniform(0.2, 0.36))
    yaw = float(rng.uniform(-0.3, 0.3))
    if ood:
        yaw = float(rng.choice([-1, 1]) * rng.uniform(0.4, 0.6))
        tilt = float(rng.uniform(0.4, 0.5))
    length = float(rng.uniform(0.8, 1.2))
    mass = float(rng.choice([0.7, 1.0, 1.4]))
    plane = center[2] - length * np.sin(tilt) - rng.uniform(0.35, 0.65)
    q = np.r_[
        center, tilt, yaw, length, [1.2, 0, plane], mass, -0.75 * length, rng.uniform(-0.08, 0.08)
    ]
    ref = np.array([rng.uniform(3.0, 4.0), rng.uniform(-0.25, 0.25)])
    nominal = simulate(q, ref, 0.28)
    if nominal["crossing"] is not None:
        q[6:8] = nominal["crossing"][:2]
    return q, ref, nominal


def sample_physics(rng):
    return float(rng.uniform(0.06, 0.55))
