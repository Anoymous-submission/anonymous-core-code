"""MuJoCo ballistic flight under explicit linear aerodynamic drag; no pixel predictor."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco as mj
from pathlib import Path

CACHE = {}
VIEW = {}
DT = 0.004


def scene(mass=1.0, radius=0.05):
    key = (float(mass), float(radius))
    if key not in CACHE:
        xml = f"""<mujoco><option timestep="{DT}" gravity="0 0 -9.81" integrator="Euler"/><visual><global offwidth="96" offheight="96"/><quality shadowsize="0"/></visual><worldbody><light pos="0 0 4" castshadow="false"/><geom name="floor" type="plane" size="5 5 .1" rgba=".4 .45 .45 1"/><body name="ball" pos="0 0 .6"><freejoint/><geom type="sphere" size="{radius}" mass="{mass}" rgba=".9 .2 .08 1"/></body><body name="basket" pos="1 0 .6"><geom type="cylinder" size=".22 .006" rgba=".9 .8 .1 .8" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
        CACHE[key] = mj.MjModel.from_xml_string(xml)
    m = CACHE[key]
    d = mj.MjData(m)
    return m, d


def initialize(m, d, start, velocity):
    mj.mj_resetData(m, d)
    d.qpos[:3] = start
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:3] = velocity
    mj.mj_forward(m, d)


def step(m, d, wind, beta):
    d.xfrc_applied[1, :3] = beta * (np.r_[wind, 0.0] - d.qvel[:3])
    mj.mj_step(m, d)


def flight(start, velocity, mass, wind, beta, seconds=1.5, render=False, target=None):
    m, d = scene(mass)
    initialize(m, d, start, velocity)
    m.body_pos[m.body("basket").id] = [*(target if target is not None else [2.0, 2.0]), 0.6]
    mj.mj_forward(m, d)
    renderer = None
    if render:
        renderer = VIEW.get(id(m))
        if renderer is None:
            renderer = VIEW[id(m)] = mj.Renderer(m, 64, 64)
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.5, 0, 0.8]
        cam.distance = 3.4
        cam.azimuth = 110
        cam.elevation = -28
    frames = []
    positions = []
    landing = None
    prev = d.qpos[:3].copy()
    for k in range(int(seconds / DT) + 1):
        if k % 10 == 0:
            positions.append(d.qpos[:3].copy())
            if renderer:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())
        if k == int(seconds / DT):
            break
        prev = d.qpos[:3].copy()
        step(m, d, wind, beta)
        now = d.qpos[:3].copy()
        if landing is None and d.qvel[2] < 0 and prev[2] >= 0.6 and now[2] < 0.6:
            a = (prev[2] - 0.6) / (prev[2] - now[2])
            landing = prev[:2] * (1 - a) + now[:2] * a
    # Renderer retained for this process; no hidden state enters dynamics.
    return dict(
        positions=np.array(positions, dtype=np.float32),
        frames=np.array(frames, dtype=np.uint8),
        landing=landing,
    )


def oracle(start, target, mass, wind, beta):
    # Derive affine landing response by three MuJoCo executions, not label input to network.
    ends = [flight(start, [*v, 4.5], mass, wind, beta)["landing"] for v in [[0, 0], [1, 0], [0, 1]]]
    base = ends[0]
    A = np.stack([ends[1] - base, ends[2] - base], axis=1)
    return np.linalg.solve(A, np.array(target) - base)


def fast_trajectory(start, velocity, mass, wind, beta, steps=375):
    # Independent explicit recurrence matching MuJoCo Euler unconstrained free flight before contact.
    x = np.array(start, dtype=float)
    v = np.array(velocity, dtype=float)
    states = [x.copy()]
    land = None
    for k in range(steps):
        prev = x.copy()
        v += DT * (beta / mass * (np.r_[wind, 0.0] - v) + [0, 0, -9.81])
        x += DT * v
        if land is None and v[2] < 0 and prev[2] >= 0.6 > x[2]:
            a = (prev[2] - 0.6) / (prev[2] - x[2])
            land = prev[:2] * (1 - a) + x[:2] * a
        if (k + 1) % 10 == 0:
            states.append(x.copy())
    return np.array(states), land
