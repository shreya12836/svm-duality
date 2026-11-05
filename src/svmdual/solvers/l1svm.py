"""1-norm SVM as a linear program (Bennett & Mangasarian 1992; Bennett & Bredensteiner 2000).

    min_{w, b, xi} ||w||_1 + C sum xi   s.t. y_i (<w, x_i> + b) >= 1 - xi_i, xi >= 0

Geometry: the distance between the hyperplanes <w, x> + b = +-1 measured in
the l_inf norm is 2 / ||w||_1 (the dual norm of l_inf is l_1), so this is the
non-Euclidean counterpart of the L2 maximum-margin problem. Its LP dual is

    max_a sum a_i   s.t. ||X^T (a * y)||_inf <= 1, y^T a = 0, 0 <= a <= C.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from svmdual.kernels import Kernel
from svmdual.model import SVMSolution, validate_inputs
from svmdual.solvers.cvx import check_status

LP_SOLVER = "HIGHS"
WEIGHT_ZERO_TOL = 1e-8


def l1_svm(
    X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, solver: str = LP_SOLVER
) -> SVMSolution:
    """Fit the 1-norm SVM (linear kernel only)."""
    if kernel.name != "linear":
        raise ValueError("The 1-norm SVM is implemented for the linear kernel only")
    validate_inputs(X, y, C)
    start = time.perf_counter()
    w, b, xi = cp.Variable(X.shape[1]), cp.Variable(), cp.Variable(len(y))
    margin = cp.multiply(y, X @ w + b) >= 1 - xi
    problem = cp.Problem(cp.Minimize(cp.norm1(w) + C * cp.sum(xi)), [margin, xi >= 0])
    problem.solve(solver=solver)  # raises if the LP solver is not installed
    status = check_status(problem, "l1_svm")
    alpha = np.clip(np.asarray(margin.dual_value, dtype=float), 0.0, C)
    weights = np.asarray(w.value, dtype=float)
    extras = {
        "dual_norm_violation": float(max(0.0, np.abs(X.T @ (alpha * y)).max() - 1.0)),
        "n_nonzero_w": float(np.sum(np.abs(weights) > WEIGHT_ZERO_TOL)),
        "w_l1": float(np.abs(weights).sum()),
    }
    return SVMSolution(
        solver="l1_svm",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=float(b.value),
        X_train=X,
        y_train=y,
        primal_objective=float(problem.value),
        dual_objective=float(alpha.sum()),
        runtime_s=time.perf_counter() - start,
        n_iter=problem.solver_stats.num_iters,
        status=status,
        w=weights,
        extras=extras,
    )
