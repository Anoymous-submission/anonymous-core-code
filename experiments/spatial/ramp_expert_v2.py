"""Robust native shooting for a retained6-DOF ramp task.

Plans jointly at1ms and0.5ms. The0.25ms simulator is reserved for validation.
No query, friction or success threshold is changed; this code never runs in a policy.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from ramp_physics import simulate


def oracle(query, friction, initial):
    bounds = [(1.0, 5.5), (-1.2, 1.2)]
    reference = np.asarray(initial, dtype=float)

    def error(a, dt):
        r = simulate(query, a, friction, dt=dt, crossing_only=True)
        if r["crossing"] is None:
            return 100.0
        return float(np.sum((r["crossing"][:2] - query[6:8]) ** 2))

    def robust_objective(a):
        errors = [error(a, dt) for dt in (0.001, 0.0005)]
        return max(errors) + 0.01 * sum(errors)

    def assess(a):
        native = simulate(query, a, friction, dt=0.001)
        middle = simulate(query, a, friction, dt=0.0005)
        rank = max(native["endpoint_error"], middle["endpoint_error"])
        if not (
            native["ramp_contact_steps"]
            and native["departed_downhill_edge"]
            and not native["other_contact_steps"]
        ):
            rank += 10
        if not (
            middle["ramp_contact_steps"]
            and middle["departed_downhill_edge"]
            and not middle["other_contact_steps"]
        ):
            rank += 10
        return native, middle, float(rank)

    chosen = reference.copy()
    native, middle, rank = assess(chosen)
    reference_rank = rank
    nfev = 0
    global_used = False
    if rank > 0.025:
        local = minimize(
            lambda a: error(a, 0.001),
            reference,
            method="Powell",
            bounds=bounds,
            options=dict(xtol=1e-5, ftol=1e-8, maxiter=25, maxfev=1000),
        )
        nfev += int(local.nfev)
        nr, mr, newrank = assess(local.x)
        if newrank < rank:
            chosen = local.x
            native = nr
            middle = mr
            rank = newrank
    if rank > 0.06:
        fit = differential_evolution(
            robust_objective,
            bounds,
            seed=230924800,
            popsize=8,
            maxiter=70,
            polish=False,
            tol=1e-7,
            x0=chosen,
        )
        polish = minimize(
            robust_objective,
            fit.x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(maxfev=800, xatol=1e-6, fatol=1e-10),
        )
        nfev += int(fit.nfev + polish.nfev)
        global_used = True
        nr, mr, newrank = assess(polish.x)
        if newrank < rank:
            chosen = polish.x
            native = nr
            middle = mr
            rank = newrank
    return (
        chosen,
        native,
        dict(
            nfev=nfev,
            global_fallback=global_used,
            planning_timesteps=[0.001, 0.0005],
            planning_max_error=rank,
            middle_success=middle["success"],
            reference_planning_max_error=reference_rank,
        ),
    )
