"""Gravity-driven 3D bank shot against a tilted native-contact panel."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
from functools import lru_cache
import mujoco as mj
import numpy as np
from scipy.optimize import minimize, differential_evolution
from render_cache import renderer_for

DT = 0.001
HORIZON = 1.2
RADIUS = 0.055
TOL = 0.14
VIEWS = {}


def rotation(yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]])


@lru_cache(maxsize=4)
def model(dt=DT, mass=1.0):
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="80"/>
    <visual><global offwidth="512" offheight="512"/><quality shadowsize="2048"/><headlight ambient=".5 .5 .5"/></visual>
    <asset><texture type="skybox" builtin="gradient" rgb1=".38 .55 .73" rgb2=".93 .96 1" width="512" height="2048"/>
    <texture name="tiles" type="2d" builtin="checker" rgb1=".8 .77 .69" rgb2=".69 .66 .58" width="512" height="512"/>
    <material name="ground" texture="tiles" texrepeat="8 8"/></asset>
    <worldbody><light pos="-1 -3 6"/>
      <geom name="floor" type="plane" pos="0 0 -2.5" size="6 6 .1" material="ground"/>
      <body name="ball"><freejoint/><geom name="ball" type="sphere" size="{RADIUS}" mass="{mass}" rgba=".94 .22 .15 1"/></body>
      <body name="panel"><geom name="panel" type="box" size=".045 1.25 1.8" rgba=".24 .52 .73 .78"/>
        <geom type="box" pos="0 1.3 0" size=".065 .06 1.9" rgba=".16 .24 .32 1"/>
        <geom type="box" pos="0 -1.3 0" size=".065 .06 1.9" rgba=".16 .24 .32 1"/>
      </body>
      <body name="goal"><geom type="sphere" size=".14" rgba="1 .78 .1 .22" contype="0" conaffinity="0"/>
       <geom type="cylinder" size=".16 .008" rgba="1 .78 .1 .9" contype="0" conaffinity="0"/>
       <geom type="box" size=".008 .008 .2" rgba="1 .8 .1 1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>"""
    return mj.MjModel.from_xml_string(xml)


def simulate(
    query, action, damping, friction, render=False, size=96, dt=DT, style=0, endpoint_only=False
):
    # q = start3, target3, panel-center3, yaw, pitch, mass.
    mass = float(query[11])
    m = model(float(dt), mass)
    d = mj.MjData(m)
    m.body_pos[m.body("panel").id] = query[6:9]
    rot = rotation(float(query[9]), float(query[10]))
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, rot.reshape(-1))
    m.body_quat[m.body("panel").id] = quat
    m.body_pos[m.body("goal").id] = query[3:6]
    for name in ("ball", "panel"):
        g = m.geom(name).id
        m.geom_solref[g] = [0.025, damping]
        m.geom_friction[g] = [friction, 0.001, 0.0001]
        m.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2]
    d.qpos[:3] = query[:3]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:3] = action
    mj.mj_forward(m, d)
    if endpoint_only:
        mj.mj_step(m, d, nstep=int(round(HORIZON / dt)))
        return dict(positions=np.stack([np.asarray(query[:3]), d.qpos[:3].copy()]))
    n = int(round(HORIZON / dt))
    sample = set(np.round(np.linspace(0, n, 16)).astype(int))
    path = []
    vel = []
    frames = []
    contact_steps = []
    other_contacts = []
    panel = m.geom("panel").id
    ball = m.geom("ball").id
    renderer = None
    if render:
        renderer = renderer_for(VIEWS, m, size)
        # View the launch side; the former rear view occluded the response.
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.45, 0, 1.5]
        cam.distance = 6.0
        cam.azimuth = 25 + style * 8
        cam.elevation = -18
    for k in range(n + 1):
        path.append(d.qpos[:3].copy())
        vel.append(d.qvel.copy())
        for c in d.contact:
            ids = {int(c.geom1), int(c.geom2)}
            if ids == {panel, ball}:
                contact_steps.append(k)
            elif ball in ids:
                other_contacts.append(k)
        if renderer is not None and k in sample:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        if k < n:
            mj.mj_step(m, d)
    path = np.array(path)
    err = float(np.linalg.norm(path[-1] - query[3:6]))
    return dict(
        positions=path,
        velocities=np.array(vel),
        frames=np.array(frames, dtype=np.uint8),
        endpoint_error=err,
        hit_panel=bool(contact_steps),
        panel_contact_steps=len(contact_steps),
        other_contact_steps=len(other_contacts),
        success=bool(err <= TOL and contact_steps and not other_contacts),
    )


def oracle(query, damping, friction, initial=None):
    start = query[:3]
    goal = query[3:6]
    guess = np.array(
        [2.7, (goal[1] - start[1]) / HORIZON, (goal[2] - start[2] + 4.905 * HORIZON**2) / HORIZON]
    )
    if initial is not None:
        guess = np.asarray(initial, dtype=float)

    def residual(u):
        result = simulate(query, u, damping, friction, endpoint_only=True)
        return result["positions"][-1] - goal

    fit = minimize(
        lambda u: float(np.sum(residual(u) ** 2)),
        guess,
        method="Powell",
        bounds=[(1.5, 5.5), (-2, 2), (2, 8)],
        options=dict(xtol=1e-5, ftol=1e-8, maxiter=30, maxfev=1600),
    )
    chosen = fit.x
    result = simulate(query, chosen, damping, friction)
    nfev = int(fit.nfev)
    refinements = 0
    # Contact-time discretization can leave Powell at a shallow local minimum.
    # Refine native endpoint error, without changing targets or success limits.
    if not result["success"] or result["endpoint_error"] > 0.03:
        for scale in (0.025, 0.08):
            simplex = np.vstack([chosen, chosen + np.eye(3) * scale])
            refine = minimize(
                lambda u: float(np.sum(residual(u) ** 2)),
                chosen,
                method="Nelder-Mead",
                bounds=[(1.5, 5.5), (-2, 2), (2, 8)],
                options=dict(initial_simplex=simplex, xatol=1e-6, fatol=1e-10, maxfev=1200),
            )
            candidate = simulate(query, refine.x, damping, friction)
            nfev += int(refine.nfev)
            refinements += 1
            if (
                candidate["hit_panel"]
                and not candidate["other_contact_steps"]
                and candidate["endpoint_error"] < result["endpoint_error"]
            ):
                chosen = refine.x
                result = candidate
            if result["endpoint_error"] < 0.015:
                break
    global_fallback = False
    if not result["success"] or result["endpoint_error"] > 0.03:
        # Retain difficult targets. Search another basin instead of rejecting
        # their physics/geometry or accepting a borderline expert label.
        bounds = [(1.5, 5.5), (-2, 2), (2, 8)]
        search = differential_evolution(
            lambda u: float(np.sum(residual(u) ** 2)),
            bounds,
            seed=230924500,
            popsize=8,
            maxiter=80,
            polish=False,
            tol=1e-8,
        )
        polish = minimize(
            lambda u: float(np.sum(residual(u) ** 2)),
            search.x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(maxfev=1500, xatol=1e-7, fatol=1e-12),
        )
        candidate = simulate(query, polish.x, damping, friction)
        nfev += int(search.nfev + polish.nfev)
        global_fallback = True
        if (
            candidate["hit_panel"]
            and not candidate["other_contact_steps"]
            and candidate["endpoint_error"] < result["endpoint_error"]
        ):
            chosen = polish.x
            result = candidate
    return (
        chosen,
        result,
        dict(
            nfev=nfev,
            converged=bool(fit.success),
            refinements=refinements,
            global_fallback=global_fallback,
        ),
    )


def sample_query(rng, ood=False):
    if ood:
        # Keep the extrapolated geometry fixed. Search commands within the
        # existing public limits; do not reject a difficult panel orientation.
        import itertools

        start = rng.uniform([-0.25, -0.2, 1.4], [0.1, 0.2, 1.8])
        center = np.array([rng.uniform(0.95, 1.25), 0, 1.5])
        rng.uniform(-0.3, 0.3)
        rng.uniform(-0.16, 0.16)
        yaw = float(rng.choice([-1, 1]) * rng.uniform(0.36, 0.5))
        pitch = float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.28))
        mass = float(rng.choice([0.7, 1.0, 1.4]))
        q = np.r_[start, [0, 0, 1], center, yaw, pitch, mass]
        first = np.array([rng.uniform(2.5, 3.1), rng.uniform(-0.35, 0.35), rng.uniform(4.9, 5.4)])
        candidates = [
            np.array(a)
            for a in itertools.product(
                [1.6, 2.5, 3.5, 4.5, 5.4], [-1.0, -0.5, 0, 0.5, 1.0], [2.2, 3.5, 5.0, 6.5, 7.8]
            )
        ]
        candidates.sort(key=lambda a: float(np.sum((a - first) ** 2)))
        for attempt, reference in enumerate([first] + candidates, 1):
            final = simulate(q, reference, 0.4, 0.1, endpoint_only=True)["positions"][-1]
            if not (0.3 < final[2] < 2.0 and final[0] < q[6] - 0.2):
                continue
            candidate = q.copy()
            candidate[3:6] = final
            nominal = simulate(candidate, reference, 0.4, 0.1)
            if nominal["hit_panel"] and not nominal["other_contact_steps"]:
                nominal["query_sampling_attempts"] = attempt
                nominal["query_geometry_resampling"] = 0
                return candidate, reference, nominal
        raise RuntimeError(f"Fixed OOD bank geometry has no feasible nominal command: {q.tolist()}")
    for attempt in range(128):
        start = rng.uniform([-0.25, -0.2, 1.4], [0.1, 0.2, 1.8])
        center = np.array([rng.uniform(0.95, 1.25), 0, 1.5])
        yaw = float(rng.uniform(-0.3, 0.3))
        pitch = float(rng.uniform(-0.16, 0.16))
        if ood:
            yaw = float(rng.choice([-1, 1]) * rng.uniform(0.36, 0.5))
            pitch = float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.28))
        mass = float(rng.choice([0.7, 1.0, 1.4]))
        q = np.r_[start, [0, 0, 1], center, yaw, pitch, mass]
        reference = np.array(
            [rng.uniform(2.5, 3.1), rng.uniform(-0.35, 0.35), rng.uniform(4.9, 5.4)]
        )
        nominal = simulate(q, reference, 0.4, 0.10)
        q[3:6] = nominal["positions"][-1]
        if (
            nominal["hit_panel"]
            and not nominal["other_contact_steps"]
            and 0.3 < q[5] < 2.0
            and q[3] < q[6] - 0.2
        ):
            nominal["query_sampling_attempts"] = attempt + 1
            return q, reference, nominal
    raise RuntimeError("Nominal bank-query feasibility sampler exhausted")


def sample_physics(rng):
    return float(rng.uniform(0.25, 0.55)), float(rng.uniform(0.035, 0.165))
