import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco as mj

CACHE = {}
VIEWS = {}
DT = 0.001
T = 1.2


def scene(mass):
    if mass not in CACHE:
        xml = f"""<mujoco><option timestep="{DT}" gravity="0 0 0" integrator="implicitfast" cone="elliptic" iterations="80"/><visual><global offwidth="96" offheight="96"/><quality shadowsize="0"/></visual><worldbody><light pos="0 -1 4" diffuse="1 1 1" castshadow="false"/><geom type="plane" size="4 4 .1" rgba=".65 .67 .7 1" contype="0" conaffinity="0"/><body name="ball" pos="0 0 .07"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><geom name="ball" type="sphere" size=".05" mass="{mass}" rgba=".9 .1 .05 1"/></body><body name="wall" pos="1 0 .1"><geom name="wall" type="box" size=".03 2 .1" rgba=".2 .35 .7 1"/></body><body name="goal" pos=".3 .3 .005"><geom type="cylinder" size=".1 .004" rgba=".95 .85 .1 1" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
        CACHE[mass] = mj.MjModel.from_xml_string(xml)
    return CACHE[mass]


def simulate(action, mass, L, angle, damping, friction, render=False, goal=None):
    m = scene(float(mass))
    d = mj.MjData(m)
    n = np.array([np.cos(angle), np.sin(angle)])
    t = np.array([-n[1], n[0]])
    wall = m.body("wall").id
    m.body_pos[wall] = [*((L + 0.08) * n), 0.1]
    m.body_quat[wall] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    m.body_pos[m.body("goal").id] = [*(goal if goal is not None else [0.3, 0.3]), 0.005]
    for name in ["ball", "wall"]:
        g = m.geom(name).id
        m.geom_solref[g] = [0.04, damping]
        m.geom_friction[g] = [friction, 0.001, 0.001]
        m.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2]
    d.qvel[:] = action
    mj.mj_forward(m, d)
    view = None
    if render:
        if float(mass) not in VIEWS:
            VIEWS[float(mass)] = mj.Renderer(m, 64, 64)
        view = VIEWS[float(mass)]
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.5, 0, 0]
        cam.distance = 2.9
        cam.azimuth = 90
        cam.elevation = -85
    positions = []
    velocities = []
    frames = []
    hit = False
    incoming = None
    outgoing = None
    last = False
    for k in range(1201):
        if k % 80 == 0:
            positions.append(d.qpos.copy())
            velocities.append(d.qvel.copy())
            if view:
                view.update_scene(d, camera=cam)
                frames.append(view.render().copy())
        if k == 1200:
            break
        pre = d.qvel.copy()
        mj.mj_step(m, d)
        contact = bool(d.ncon)
        if contact and not hit:
            incoming = pre.copy()
            hit = True
        if last and not contact:
            outgoing = d.qvel.copy()
        last = contact
    return dict(
        final=d.qpos.copy(),
        positions=np.array(positions, dtype=np.float32),
        velocities=np.array(velocities, dtype=np.float32),
        video=np.array(frames, dtype=np.uint8),
        hit=hit,
        incoming=incoming,
        outgoing=outgoing,
    )


def oracle(goal, mass, L, angle, damping, friction):
    # Bracket each world-normal coordinate with actual contact trajectories.
    n = np.array([np.cos(angle), np.sin(angle)])
    t = np.array([-n[1], n[0]])
    basis = np.stack([n, t], axis=1)
    g = np.asarray(goal)
    gt = basis.T @ g
    u = np.array([2.0, gt[1] / T])
    best = None
    error = float("inf")
    for cycle in range(3):
        for k in [0, 1]:
            lo, hi = (L / T + 0.05, 5.0) if k == 0 else (-2.0, 2.0)
            ulo = u.copy()
            uhi = u.copy()
            ulo[k] = lo
            uhi[k] = hi
            flo = (basis.T @ simulate(basis @ ulo, mass, L, angle, damping, friction)["final"])[
                k
            ] - gt[k]
            fhi = (basis.T @ simulate(basis @ uhi, mass, L, angle, damping, friction)["final"])[
                k
            ] - gt[k]
            if flo * fhi > 0:
                raise RuntimeError(("unbracketed", k, gt.tolist(), flo, fhi))
            for j in range(18):
                u[k] = (lo + hi) / 2
                r = simulate(basis @ u, mass, L, angle, damping, friction)
                err = np.linalg.norm(r["final"] - g)
                if r["hit"] and err < error:
                    best = (basis @ u.copy(), r)
                    error = err
                if error < 0.003:
                    return best
                fm = (basis.T @ r["final"])[k] - gt[k]
                if fm * flo > 0:
                    lo = u[k]
                    flo = fm
                else:
                    hi = u[k]
                    fhi = fm
    return best
