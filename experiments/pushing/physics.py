import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco as mj

CACHE = {}
VIEWS = {}
DT = 0.001
T = 1.6


def scene(mass, slope=0.0):
    key = (mass, slope)
    quat = f"{np.cos(slope/2)} 0 {np.sin(slope/2)} 0"
    if key not in CACHE:
        xml = f"""<mujoco><option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="80"/><visual><global offwidth="128" offheight="128"/><quality shadowsize="1024"/></visual><worldbody><light name="key" pos=".1 -.6 1.4" dir="0 .2 -1" directional="false" diffuse=".9 .9 .9" ambient=".2 .2 .2" castshadow="true"/><geom name="table" quat="{quat}" type="plane" size="1 .7 .03" rgba=".45 .5 .55 1" solref=".01 1"/><body name="block" pos="0 0 .031"><freejoint/><geom name="block" type="box" size=".03 .025 .025" mass="{mass}" rgba=".85 .08 .04 1" solref=".01 1"/></body><body name="pusher" mocap="true" pos="-.06 0 .03"><geom name="pusher" type="sphere" size=".022" rgba=".1 .25 .9 1" solref=".01 1"/></body><body name="goal" mocap="true" pos=".3 0 .001"><geom type="cylinder" size=".035 .001" rgba=".95 .82 .1 1" contype="0" conaffinity="0"/></body><geom name="rail1" type="box" pos=".15 -.14 .002" size=".003 .03 .002" rgba=".08 .08 .08 1" contype="0" conaffinity="0"/><geom name="rail2" type="box" pos=".3 -.14 .002" size=".003 .03 .002" rgba=".08 .08 .08 1" contype="0" conaffinity="0"/></worldbody></mujoco>"""
        CACHE[key] = mj.MjModel.from_xml_string(xml)
    return CACHE[key]


def simulate(
    action,
    mu,
    mass=0.2,
    slope=0.0,
    start=(0.0, 0.0),
    goal=(0.3, 0.0),
    light="neutral",
    render=False,
    dt=DT,
):
    speed, angle = map(float, action)
    m = scene(float(mass), float(slope))
    m.opt.timestep = dt
    d = mj.MjData(m)
    ct, st = np.cos(slope), np.sin(slope)
    R = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    quat = np.array([np.cos(slope / 2), 0, np.sin(slope / 2), 0])
    direction = np.array([np.cos(angle), np.sin(angle)])
    start = np.asarray(start)
    m.geom_friction[m.geom("table").id] = [mu, 0.002, 0.0001]
    m.geom_friction[m.geom("block").id] = [mu, 0.002, 0.0001]
    m.geom_friction[m.geom("pusher").id] = [0.1, 0.002, 0.0001]
    d.mocap_pos[1] = R @ np.r_[goal, 0.001]
    d.mocap_quat[1] = quat
    for name, x in [("rail1", 0.15), ("rail2", 0.3)]:
        m.geom_pos[m.geom(name).id] = R @ np.array([x, -0.14, 0.002])
        m.geom_quat[m.geom(name).id] = quat
    styles = {
        "neutral": ([0.1, -0.6, 1.4], [0.9, 0.9, 0.9], [0.2, 0.2, 0.2]),
        "side": ([-0.4, 0.5, 0.8], [0.7, 0.7, 0.7], [0.16, 0.16, 0.16]),
        "warm": ([0.6, -0.2, 1.0], [1.0, 0.8, 0.6], [0.15, 0.15, 0.15]),
        "dim": ([0.1, -0.6, 1.4], [0.2, 0.2, 0.2], [0.04, 0.04, 0.04]),
        "cool": ([-0.5, -0.2, 0.6], [0.45, 0.65, 1.0], [0.08, 0.08, 0.12]),
    }
    pos, diff, amb = styles[light]
    m.light_pos[0] = pos
    m.light_diffuse[0] = diff
    m.light_ambient[0] = amb
    d.qpos[:3] = R @ np.r_[start, 0.027]
    d.qpos[3:7] = quat
    d.qvel[:] = 0
    # Fixed-duration push with a 20ms speed ramp, then vertical withdrawal.
    offset = 0.03 * abs(direction[0]) + 0.025 * abs(direction[1]) + 0.022 + 0.003

    def hand(t):
        u = min(max(t, 0), 0.12)
        progress = speed * (u * u / 0.04 if u < 0.02 else u - 0.01)
        p = start + direction * (-offset + progress)
        height = 0.027 + min(max(t - 0.12, 0) * 2, 0.18)
        return R @ np.r_[p, height]

    d.mocap_pos[0] = hand(0)
    d.mocap_quat[0] = quat
    mj.mj_forward(m, d)
    indices = set(np.rint(np.linspace(0, int(T / dt), 32)).astype(int))
    states = []
    frames = []
    motion = []
    rotations = []
    contact = False
    if render:
        key = (float(mass), float(slope))
        if key not in VIEWS:
            VIEWS[key] = mj.Renderer(m, 64, 64)
        view = VIEWS[key]
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.24, 0, 0.03]
        cam.distance = 1.15
        cam.azimuth = 100
        cam.elevation = -58
    bid = m.geom("block").id
    pid = m.geom("pusher").id
    k = 0
    while k <= int(T / dt):
        if k in indices:
            p = R.T @ d.qpos[:3]
            v = R.T @ d.qvel[:3]
            states.append(np.r_[p, v])
            rotations.append(np.r_[d.qpos[3:7], d.qvel[3:6]].copy())
            if render:
                view.update_scene(d, camera=cam)
                frames.append(view.render().copy())

        if k == int(T / dt):
            break
        if k * dt < 0.22:
            d.mocap_pos[0] = hand((k + 1) * dt)
            mj.mj_step(m, d)
            for j in range(d.ncon):
                con = d.contact[j]
                if {int(con.geom1), int(con.geom2)} == {bid, pid}:
                    contact = True
            k += 1
        else:
            nxt = min(i for i in indices if i > k)
            mj.mj_step(m, d, nstep=nxt - k)
            k = nxt
    motion = [R.T @ hand(t) for t in np.arange(81) * 0.004]
    final = R.T @ d.qpos[:3]
    velocity = R.T @ d.qvel[:3]
    return dict(
        final=final,
        velocity=velocity,
        states=np.array(states),
        video=np.array(frames, dtype=np.uint8),
        motion=np.array(motion),
        contact=contact,
        rotations=np.array(rotations),
        plane_normal=d.geom_xmat[m.geom("table").id].reshape(3, 3)[:, 2].copy(),
    )


def expert(goal_x, mu, mass, slope, start=(0.0, 0.0)):
    lo, hi = 0.35, 2.2
    low = simulate([lo, 0], mu, mass, slope, start)
    high = simulate([hi, 0], mu, mass, slope, start)
    assert low["final"][0] < goal_x < high["final"][0], (goal_x, low["final"], high["final"])
    best = None
    for _ in range(16):
        speed = (lo + hi) / 2
        r = simulate([speed, 0], mu, mass, slope, start)
        err = abs(r["final"][0] - goal_x)
        if best is None or err < best[0]:
            best = (err, speed, r)
        if r["final"][0] < goal_x:
            lo = speed
        else:
            hi = speed
    if best[0] > 0.003:
        center = best[1]
        for speed in np.linspace(max(0.35, center - 0.06), min(2.2, center + 0.06), 41):
            r = simulate([speed, 0], mu, mass, slope, start)
            err = abs(r["final"][0] - goal_x)
            if err < best[0]:
                best = (err, float(speed), r)
    return best[1], best[2]
