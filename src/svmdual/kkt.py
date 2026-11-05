"""KKT residuals, support-vector sparsity and margin geometry of a fitted SVM.

For the soft-margin L2 SVM with xi_i = max(0, 1 - y_i f_i) the KKT system is

    primal feasibility     y_i f_i >= 1 - xi_i,  xi_i >= 0          (holds by construction)
    dual feasibility       0 <= a_i <= C,  sum_i a_i y_i = 0
    stationarity           w = sum_i a_i y_i phi(x_i),  a_i + mu_i = C
    complementary slack.   a_i (y_i f_i - 1 + xi_i) = 0,  (C - a_i) xi_i = 0

Every residual below is the max absolute violation over i, so 0 means exact.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from svmdual.model import SVMSolution, hinge_losses
from svmdual.solvers.cvx import SV_TOL_REL

NAN = float("nan")
NORM_EPS = 1e-12  # below this ||f|| or ||w||_1 is treated as zero (infinite margin)


@dataclass(frozen=True)
class KKTReport:
    """Max-abs residual of each KKT condition on the training set (0 = exact)."""

    solver: str
    kernel: str
    C: float
    dual_box_violation: float
    equality_residual: float
    stationarity_w: float
    stationarity_xi: float
    comp_slack_margin: float
    comp_slack_slack: float
    max_violating_pair: float
    primal_objective: float
    dual_objective: float
    duality_gap: float
    relative_gap: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class Sparsity:
    """Support-vector counts: free (0 < a < C) and bounded (a = C)."""

    n: int
    n_sv: int
    n_free: int
    n_bounded: int

    @property
    def ratio(self) -> float:
        return self.n_sv / self.n


def _max_violating_pair(alpha: np.ndarray, g: np.ndarray, y: np.ndarray, C: float) -> float:
    """m(a) - M(a) from the SMO optimality condition (<= 0 at an exact optimum)."""
    score = y - g  # = -y_i * grad_i
    tol = SV_TOL_REL * C
    below_c, above_0 = alpha < C - tol, alpha > tol
    up = ((y > 0) & below_c) | ((y < 0) & above_0)
    low = ((y > 0) & above_0) | ((y < 0) & below_c)
    if not (np.any(up) and np.any(low)):
        return 0.0
    return float(max(score[up].max() - score[low].min(), 0.0))


def kkt_report(solution: SVMSolution) -> KKTReport:
    """Evaluate every KKT condition on the training set."""
    s = solution
    alpha, y, C = s.alpha, s.y_train, s.C
    f = s.decision_function(s.X_train)
    xi = hinge_losses(y * f)
    is_l2 = s.solver != "l1_svm"
    g = f - s.b
    gap = s.duality_gap
    return KKTReport(
        solver=s.solver,
        kernel=s.kernel.label,
        C=C,
        dual_box_violation=float(np.maximum(np.maximum(-alpha, alpha - C), 0.0).max()),
        equality_residual=float(abs(alpha @ y)),
        stationarity_w=s.extras.get("stationarity_w", NAN),
        stationarity_xi=s.extras.get("stationarity_xi", NAN),
        comp_slack_margin=float(np.abs(alpha * (y * f - 1.0 + xi)).max()),
        comp_slack_slack=float(np.abs((C - alpha) * xi).max()),
        max_violating_pair=_max_violating_pair(alpha, g, y, C) if is_l2 else NAN,
        primal_objective=s.primal_objective,
        dual_objective=s.dual_objective,
        duality_gap=gap,
        relative_gap=gap / max(1.0, abs(s.primal_objective)) if math.isfinite(gap) else NAN,
    )


def sparsity(alpha: np.ndarray, C: float, rel_tol: float = SV_TOL_REL) -> Sparsity:
    """Count support vectors: a_i > tol; free if also a_i < C - tol, bounded otherwise."""
    tol = rel_tol * C
    is_sv = alpha > tol
    is_free = is_sv & (alpha < C - tol)
    n_sv, n_free = int(is_sv.sum()), int(is_free.sum())
    return Sparsity(n=len(alpha), n_sv=n_sv, n_free=n_free, n_bounded=n_sv - n_free)


def rkhs_norm_sq(solution: SVMSolution) -> float:
    """||f||_H^2 = (a*y)^T K (a*y); equals ||w||^2 in the linear case."""
    if solution.w is not None and solution.solver != "l1_svm":
        return float(solution.w @ solution.w)
    coef = solution.dual_coef
    K = solution.kernel(solution.X_train, solution.X_train)
    return float(coef @ K @ coef)


def margin_width(solution: SVMSolution) -> float:
    """Distance between f = +1 and f = -1: 2/||f||_H (L2) or the l_inf width 2/||w||_1 (L1)."""
    norm = (
        float(np.abs(solution.w).sum())
        if solution.solver == "l1_svm"
        else math.sqrt(max(rkhs_norm_sq(solution), 0.0))
    )
    return math.inf if norm < NORM_EPS else 2.0 / norm


def reduced_hull_points(solution: SVMSolution) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest points of the two reduced convex hulls (Bennett & Bredensteiner 2000).

    With s = sum_{y_i = +1} a_i = sum_{y_i = -1} a_i, the points
    c = sum_{+} (a_i / s) x_i and d = sum_{-} (a_i / s) x_i are convex combinations
    with weights at most D = C / s, and w = s (c - d). Returns (c, d, D).
    """
    alpha, y, X = solution.alpha, solution.y_train, solution.X_train
    pos, neg = y > 0, y < 0
    s = float(alpha[pos].sum())
    if s <= 0.0 or float(alpha[neg].sum()) <= 0.0:
        raise ValueError("Reduced-hull points need a positive multiplier in each class")
    c = alpha[pos] @ X[pos] / s
    d = alpha[neg] @ X[neg] / float(alpha[neg].sum())
    return c, d, solution.C / s
