"""Versioned finite known-physics reference; no optimality or reachability guarantee.

This candidate is separate from the current training-label generators. Bank
runs the same bounded sequence on EVERY query, never escalates after looking at
policy outcomes, and never uses finer validation to select a command.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from action_limits import BOUNDS
import bank_physics as bank
from task_adapter import planned_gate_action
import ramp_expert_v2

BUDGET = {
    "bank": dict(
        algorithm="Powell then independent differential evolution then Nelder-Mead, all stages unconditional",
        powell_maxfev=1600,
        powell_maxiter=30,
        de_seed=230924500,
        de_popsize=8,
        de_maxiter=80,
        polish_maxfev=1500,
        max_objective_evaluations=5044,
        candidate_full_executions=4,
        selection_timestep=0.001,
        validation_timestep=0.0005,
    ),
    "gate": dict(
        algorithm="fixed at most10 analytic waypoint candidates with native clearance scoring",
        max_candidates=10,
        selection_timestep=0.004,
        validation_timestep=0.002,
    ),
    "ramp": dict(
        algorithm="fixed ramp_expert_v2 branch rules for every query; no policy-dependent budget changes",
        powell_maxfev=1000,
        de_seed=230924800,
        de_popsize=8,
        de_maxiter=70,
        polish_maxfev=800,
        max_objective_evaluations=2936,
        planning_timesteps=[0.001, 0.0005],
        validation_timestep=0.00025,
    ),
}


def solve(task, query, physics, nominal):
    if task == "gate":
        action, result, details = planned_gate_action(query, physics[:3], physics[3])
    elif task == "ramp":
        action, result, details = ramp_expert_v2.oracle(query, physics[0], initial=nominal)
    else:
        assert task == "bank"
        bounds = list(zip(*BOUNDS["bank"]))
        damping, friction = physics[:2]

        def objective(a):
            endpoint = bank.simulate(query, a, damping, friction, endpoint_only=True)["positions"][
                -1
            ]
            return float(np.sum((endpoint - query[3:6]) ** 2))

        local = minimize(
            objective,
            nominal,
            method="Powell",
            bounds=bounds,
            options=dict(xtol=1e-5, ftol=1e-8, maxiter=30, maxfev=1600),
        )
        search = differential_evolution(
            objective, bounds, seed=230924500, popsize=8, maxiter=80, polish=False, tol=1e-8
        )
        polish = minimize(
            objective,
            search.x,
            method="Nelder-Mead",
            bounds=bounds,
            options=dict(maxfev=1500, xatol=1e-7, fatol=1e-12),
        )
        candidates = [np.asarray(nominal), local.x, search.x, polish.x]
        executions = [bank.simulate(query, a, damping, friction) for a in candidates]
        feasible = [
            i for i, r in enumerate(executions) if r["hit_panel"] and not r["other_contact_steps"]
        ]
        pool = feasible if feasible else list(range(4))
        index = min(pool, key=lambda i: executions[i]["endpoint_error"])
        action = candidates[index]
        result = executions[index]
        details = dict(
            nfev=int(local.nfev + search.nfev + polish.nfev),
            all_three_search_stages_called=True,
            chosen_candidate=index,
            full_candidate_executions=4,
        )
        assert details["nfev"] <= BUDGET["bank"]["max_objective_evaluations"]
    lo, hi = BOUNDS[task]
    assert np.isfinite(action).all() and np.all(action >= lo) and np.all(action <= hi)
    return (
        action,
        result,
        dict(
            **details,
            reference_version=1,
            task=task,
            budget=BUDGET[task],
            output_is_ground_truth_action=False,
            certified_optimal=False,
        ),
    )
