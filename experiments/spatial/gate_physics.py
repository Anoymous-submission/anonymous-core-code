"""True XYZ free-body gate interception with two scheduled velocity impulses.

MuJoCo executes every reported trajectory. The independent linear recurrence is
only an oracle/data-validation tool; it will not be called by the neural policy.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
from functools import lru_cache

import mujoco as mj
import numpy as np
from render_cache import renderer_for

DT = 0.004
SWITCH = 0.8
HORIZON = 1.6
RADIUS = 0.055
GATE_TOL = 0.22
GOAL_TOL = 0.20
# Inner opening half-width/height minus the payload sphere radius.
APERTURE_Y = 0.34 - 0.055 - RADIUS
APERTURE_Z = 0.60 - 0.055 - RADIUS
VIEWS = {}


def trajectory_metrics(path, query, dt, contact_steps):
    """Score actual interpolated left-to-right aperture crossings in the horizon.

    Timing retains the original two separate constraints: proximity at SWITCH,
    and a valid crossing during (0, HORIZON]. It does not require crossing at
    exactly SWITCH; prescribed expert waypoints can precede the gate plane.
    """
    path = np.asarray(path, dtype=float)
    switch = int(round(SWITCH / dt))
    assert len(path) == int(round(HORIZON / dt)) + 1
    ids = np.flatnonzero((path[:-1, 0] < query[3]) & (path[1:, 0] >= query[3]))
    crossings = []
    times = []
    for k in ids:
        fraction = (query[3] - path[k, 0]) / (path[k + 1, 0] - path[k, 0])
        crossings.append(path[k] + fraction * (path[k + 1] - path[k]))
        times.append((k + fraction) * dt)
    crossings = np.asarray(crossings, dtype=float).reshape(-1, 3)
    times = np.asarray(times, dtype=float)
    offsets = crossings - query[3:6]
    valid = (
        (np.abs(offsets[:, 1]) < APERTURE_Y)
        & (np.abs(offsets[:, 2]) < APERTURE_Z)
        & (times > 0)
        & (times <= HORIZON)
    )
    selected = np.flatnonzero(valid)
    crossing_time = float(times[selected[0]]) if len(selected) else float("nan")
    ge = float(np.linalg.norm(path[switch] - query[3:6]))
    ee = float(np.linalg.norm(path[-1] - query[6:9]))
    crossed = bool(len(ids))
    inside = bool(valid.any())
    return dict(
        gate_error=ge,
        endpoint_error=ee,
        gate_contact_steps=int(contact_steps),
        crossed_gate_plane=crossed,
        crossed_gate_aperture=inside,
        gate_crossing_time=crossing_time,
        gate_crossing_count=int(len(ids)),
        success=bool(ge <= GATE_TOL and ee <= GOAL_TOL and inside and not contact_steps),
    )


@lru_cache(maxsize=8)
def model(gated=True, dt=DT, mass=1.0):
    ring = ""
    if gated:
        for name, pos, size in (
            ("left", "0 -.34 0", ".015 .055 .655"),
            ("right", "0 .34 0", ".015 .055 .655"),
            ("top", "0 0 .60", ".015 .34 .055"),
            ("bottom", "0 0 -.60", ".015 .34 .055"),
        ):
            ring += f'<geom name="gate_{name}" type="box" pos="{pos}" size="{size}" rgba=".2 .7 .75 1"/>'
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 -9.81" integrator="Euler"/>
      <visual><global offwidth="512" offheight="512"/><quality shadowsize="2048"/>
      <headlight ambient=".5 .5 .5" diffuse=".5 .5 .5"/></visual>
      <asset><texture type="skybox" builtin="gradient" rgb1=".35 .55 .75" rgb2=".9 .95 1" width="512" height="2048"/>
      <texture name="tiles" type="2d" builtin="checker" rgb1=".79 .84 .88" rgb2=".68 .75 .81" width="512" height="512"/>
      <material name="ground" texture="tiles" texrepeat="8 8" reflectance=".05"/></asset>
      <worldbody><light pos="0 -3 6" diffuse=".9 .85 .8"/>
      <geom name="floor" type="plane" pos="0 0 -.8" size="7 7 .1" material="ground"/>
      <geom type="box" pos="-.75 .8 -.3" size=".3 .3 .5" rgba=".32 .42 .52 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="2.5 .8 -.05" size=".25 .25 .75" rgba=".4 .5 .6 1" contype="0" conaffinity="0"/>
      <body name="payload"><freejoint/><geom name="payload" type="sphere" size="{RADIUS}" mass="{mass}" rgba=".93 .22 .16 1"/></body>
      <body name="gate" pos=".7 0 1.6">{ring}</body>
      <body name="goal"><geom name="goal" type="sphere" size=".20" rgba="1 .78 .1 .18" contype="0" conaffinity="0"/>
      <geom type="cylinder" size=".22 .008" rgba="1 .75 .1 .7" contype="0" conaffinity="0"/>
      <geom type="box" size=".012 .012 .27" rgba="1 .78 .1 1" contype="0" conaffinity="0"/></body>
      </worldbody></mujoco>"""
    return mj.MjModel.from_xml_string(xml)


def recurrence(start, action, mass, wind, drag, dt=DT):
    n = int(round(HORIZON / dt))
    switch = int(round(SWITCH / dt))
    x = np.asarray(start, dtype=float).copy()
    v = np.asarray(action[:3], dtype=float).copy()
    states = [x.copy()]
    velocities = [v.copy()]
    for k in range(n):
        if k == switch:
            v += action[3:6]
        v += dt * (drag / mass * (np.asarray(wind) - v) + [0, 0, -9.81])
        x += dt * v
        states.append(x.copy())
        velocities.append(v.copy())
    return np.array(states), np.array(velocities)


def oracle(query, wind, drag):
    start = query[:3]
    gate = query[3:6]
    goal = query[6:9]
    mass = float(query[9])
    decay = 1 - DT * drag / mass

    def coefficients(n):
        series = decay * (1 - decay**n) / (1 - decay)
        return DT * series, DT**2 * (n - series) / (1 - decay)

    k = int(round(SWITCH / DT))
    n = int(round(HORIZON / DT))
    b1, c1 = coefficients(k)
    b2, c2 = coefficients(n)
    bp, _ = coefficients(n - k)
    force = drag / mass * np.asarray(wind) + [0, 0, -9.81]
    initial = (gate - start - c1 * force) / b1
    impulse = (goal - start - b2 * initial - c2 * force) / bp
    return np.r_[initial, impulse]


def simulate(query, action, wind, drag, gated=True, render=False, size=96, style=0, dt=DT):
    mass = float(query[9])
    m = model(bool(gated), float(dt), mass)
    d = mj.MjData(m)
    body = m.body("payload").id
    m.body_pos[m.body("gate").id] = query[3:6]
    m.body_pos[m.body("goal").id] = query[6:9]
    colors = ((0.93, 0.22, 0.16, 1), (0.9, 0.3, 0.12, 1), (0.78, 0.12, 0.25, 1))
    m.geom_rgba[m.geom("payload").id] = colors[style % 3]
    tints = (
        (1, 1, 1, 1),
        (0.82, 1, 0.84, 1),
        (1, 0.87, 0.75, 1),
        (0.7, 0.92, 1, 1),
        (1, 0.73, 0.87, 1),
    )
    m.mat_rgba[m.mat("ground").id] = tints[style % 5]
    d.qpos[:3] = query[:3]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:3] = action[:3]
    mj.mj_forward(m, d)
    n = int(round(HORIZON / dt))
    switch = int(round(SWITCH / dt))
    frames_at = set(np.round(np.linspace(0, n, 16)).astype(int).tolist())
    path = []
    velocity = []
    frames = []
    contacts = []
    renderer = None
    if render:
        renderer = renderer_for(VIEWS, m, size)
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.75, 0, 1.25]
        cam.distance = 5.0
        cam.azimuth = 125 + style * 8
        cam.elevation = -24
    gate_ids = (
        {m.geom(f"gate_{s}").id for s in ("left", "right", "top", "bottom")} if gated else set()
    )
    for k in range(n + 1):
        path.append(d.qpos[:3].copy())
        velocity.append(d.qvel[:3].copy())
        if renderer is not None and k in frames_at:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        for c in d.contact:
            if c.geom1 in gate_ids or c.geom2 in gate_ids:
                contacts.append(k)
        if k == n:
            break
        if k == switch:
            d.qvel[:3] += action[3:6]
        d.xfrc_applied[body, :3] = drag * (np.asarray(wind) - d.qvel[:3])
        mj.mj_step(m, d)
    path = np.array(path)
    velocity = np.array(velocity)
    metrics = trajectory_metrics(path, query, dt, len(contacts))
    return dict(
        positions=path, velocities=velocity, frames=np.array(frames, dtype=np.uint8), **metrics
    )


def sample_query(rng):
    return np.r_[
        rng.uniform([-0.65, -0.35, 0.9], [-0.35, 0.35, 1.4]),
        rng.uniform([0.45, -0.35, 1.25], [0.95, 0.35, 2.0]),
        rng.uniform([1.5, -0.6, 0.75], [2.2, 0.6, 1.65]),
        rng.choice([0.7, 1.0, 1.4]),
    ]


def sample_physics(rng, regime=None):
    if regime is None:
        regime = "mild" if rng.random() < 0.5 else "strong"
    if regime == "mild":
        return rng.uniform([-0.15, -0.15, -0.08], [0.15, 0.15, 0.08]), float(
            rng.uniform(0.65, 0.75)
        )
    assert regime == "strong"
    return rng.uniform([-1.6, -1.6, -0.5], [1.6, 1.6, 0.5]), float(rng.uniform(0.2, 1.2))
