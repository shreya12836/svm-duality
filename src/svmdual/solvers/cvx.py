"""Exact interior-point solutions of the L2-SVM primal and dual with CVXPY.

Primal (kernel case through the empirical feature map Phi, K = Phi Phi^T):

    min_{w, b, xi} 1/2 ||w||^2 + C sum xi   s.t. y_i (Phi_i w + b) >= 1 - xi_i, xi >= 0

The multipliers of the margin constraints are the dual variables alpha, and
those of xi >= 0 are mu; stationarity requires w = Phi^T (alpha * y) and
alpha + mu = C, which we record as residuals.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from svmdual.kernels import Kernel, primal_features
from svmdual.model import (
    STATUS_INACCURATE,
    STATUS_OPTIMAL,
    SVMSolution,
    best_offset,
    l2_dual_objective,
    l2_primal_objective,
    offset_from_kkt,
    validate_inputs,
)

DEFAULT_SOLVER = "CLARABEL"
SV_TOL_REL = 1e-6  # alpha_i > SV_TOL_REL * C counts as nonzero for interior-point output
OPTIMAL_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})


def check_status(problem: cp.Problem, name: str) -> str:
    """Raise unless CVXPY reports an optimal (possibly inaccurate) solution."""
    if problem.status not in OPTIMAL_STATUSES:
        raise RuntimeError(f"{name}: solver returned status {problem.status!r}")
    return STATUS_OPTIMAL if problem.status == cp.OPTIMAL else STATUS_INACCURATE


def cvx_primal(
    X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, solver: str = DEFAULT_SOLVER
) -> SVMSolution:
    """Solve the primal exactly; recover alpha and mu from constraint multipliers."""
    validate_inputs(X, y, C)
    start = time.perf_counter()
    Phi = primal_features(X, kernel)
    w, b, xi = cp.Variable(Phi.shape[1]), cp.Variable(), cp.Variable(len(y))
    margin = cp.multiply(y, Phi @ w + b) >= 1 - xi
    nonneg = xi >= 0
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi)), [margin, nonneg])
    problem.solve(solver=solver)
    status = check_status(problem, "cvx_primal")
    alpha = np.clip(np.asarray(margin.dual_value, dtype=float), 0.0, C)
    mu = np.asarray(nonneg.dual_value, dtype=float)
    K = kernel(X, X)
    extras = {
        "stationarity_w": float(np.abs(w.value - Phi.T @ (alpha * y)).max()),
        "stationarity_xi": float(np.abs(C - alpha - mu).max()),
        "feature_rank": float(Phi.shape[1]),
    }
    return SVMSolution(
        solver="cvx_primal",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=float(b.value),
        X_train=X,
        y_train=y,
        primal_objective=float(problem.value),
        dual_objective=l2_dual_objective(alpha, K, y),
        runtime_s=time.perf_counter() - start,
        n_iter=problem.solver_stats.num_iters,
        status=status,
        w=np.asarray(w.value, dtype=float) if kernel.name == "linear" else None,
        extras=extras,
    )


def cvx_dual(
    X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, solver: str = DEFAULT_SOLVER
) -> SVMSolution:
    """Solve the dual exactly.

    b is the exact primal minimiser for the recovered f (best_offset). The
    textbook average over free SVs is kept in extras["b_kkt"]: with
    interior-point output it is biased by multipliers at noise level.
    """
    validate_inputs(X, y, C)
    start = time.perf_counter()
    Phi = primal_features(X, kernel)
    alpha_var = cp.Variable(len(y))
    objective = cp.sum(alpha_var) - 0.5 * cp.sum_squares(Phi.T @ cp.multiply(alpha_var, y))
    equality = y @ alpha_var == 0
    constraints = [alpha_var >= 0, alpha_var <= C, equality]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver)
    status = check_status(problem, "cvx_dual")
    alpha = np.clip(np.asarray(alpha_var.value, dtype=float), 0.0, C)
    K = kernel(X, X)
    g = K @ (alpha * y)
    b, _ = best_offset(g, y)
    return SVMSolution(
        solver="cvx_dual",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=b,
        X_train=X,
        y_train=y,
        primal_objective=l2_primal_objective(alpha, b, K, y, C),
        dual_objective=float(problem.value),
        runtime_s=time.perf_counter() - start,
        n_iter=problem.solver_stats.num_iters,
        status=status,
        # The multiplier of y^T a = 0 equals b up to CVXPY's sign convention; kept signed.
        extras={
            "b_equality_multiplier": float(equality.dual_value),
            "b_kkt": offset_from_kkt(alpha, g, y, C, tol=SV_TOL_REL * C),
        },
    )
