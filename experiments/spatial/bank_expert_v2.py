"""Uniform training-label refinement, preserving every sampled task.

Native and half-step success gate the same bounded extra search on every record.
Half-step is part of planning/acceptance, not an independent validation claim.
This is label generation only, never a neural-policy forward operation.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import bank_physics as bank


def refine(query, damping, friction, initial):
    bounds = [(1.5, 5.5), (-2, 2), (2, 8)]

    def assess(a):
        n = bank.simulate(query, a, damping, friction)
        f = bank.simulate(query, a, damping, friction, dt=0.0005)
        return n, f

    def objective(a):
        end = bank.simulate(query, a, damping, friction, endpoint_only=True)["positions"][-1]
        return float(np.sum((end - query[3:6]) ** 2))

    chosen = np.asarray(initial).copy()
    native, fine = assess(chosen)
    nfev = 0
    used = False
    if not (native["success"] and fine["success"]):
        used = True
        search = differential_evolution(
            objective,
            bounds,
            seed=2309249900,
            popsize=16,
            maxiter=180,
            x0=chosen,
            polish=False,
            tol=1e-10,
        )
        polish = minimize(
            objective,
            search.x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(maxfev=2200, xatol=1e-8, fatol=1e-13),
        )
        nfev = int(search.nfev + polish.nfev)
        candidates = [(chosen, native, fine)]
        for a in (search.x, polish.x):
            candidates.append((a, *assess(a)))
        chosen, native, fine = min(
            candidates,
            key=lambda c: (
                not (c[1]["success"] and c[2]["success"]),
                max(c[1]["endpoint_error"], c[2]["endpoint_error"]),
            ),
        )
    return (
        chosen,
        native,
        dict(
            bank_label_version=2,
            refinement_used=used,
            refinement_nfev=nfev,
            planning_timesteps=[0.001, 0.0005],
            fine_success=bool(fine["success"]),
            planning_fine_error=float(fine["endpoint_error"]),
            additional_search_seed=2309249900,
            additional_search_maxfev=10888,
        ),
    )


def oracle(query, damping, friction, initial):
    old, _, details = bank.oracle(query, damping, friction, initial=initial)
    action, result, extra = refine(query, damping, friction, old)
    return action, result, {**details, **extra}
