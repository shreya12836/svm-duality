"""The fitted-SVM container shared by all solvers, plus L2-SVM objective helpers.

Notation (soft-margin L2 SVM, labels y_i in {-1, +1}):

    primal  P(f, b) = 1/2 ||f||_H^2 + C * sum_i max(0, 1 - y_i (f(x_i) + b))
    dual    D(a)    = sum_i a_i - 1/2 a^T Q a,   Q_ij = y_i y_j k(x_i, x_j)
    f(x)    = sum_i a_i y_i k(x_i, x) + b
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from svmdual.kernels import Kernel

STATUS_OPTIMAL = "optimal"
STATUS_MAX_ITER = "max_iter"
STATUS_FIXED_BUDGET = "fixed_budget"
STATUS_INACCURATE = "optimal_inaccurate"


@dataclass(frozen=True)
class SVMSolution:
    """Result of one solver run on the training set.

    alpha: dual multipliers (n,). For Pegasos these are the implied expansion
        weights, which need not satisfy the dual constraints.
    w: explicit weight vector in input space when the model is linear and the
        solver produces one (CVXPY primal, L1-norm SVM); otherwise None.
    extras: solver-specific scalar diagnostics (e.g. stationarity residuals).
    history: per-iteration traces (e.g. duality gap for SMO).
    """

    solver: str
    kernel: Kernel
    C: float
    alpha: np.ndarray
    b: float
    X_train: np.ndarray
    y_train: np.ndarray
    primal_objective: float
    dual_objective: float
    runtime_s: float
    n_iter: int | None
    status: str
    w: np.ndarray | None = None
    extras: Mapping[str, float] = field(default_factory=dict)
    history: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def dual_coef(self) -> np.ndarray:
        """alpha_i * y_i, the coefficients of f in the kernel expansion."""
        return self.alpha * self.y_train

    @property
    def duality_gap(self) -> float:
        """P - D; NaN when the solver has no dual objective (Pegasos)."""
        return self.primal_objective - self.dual_objective

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) + b for each row of X."""
        if self.w is not None:
            return X @ self.w + self.b
        return self.kernel(X, self.X_train) @ self.dual_coef + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Labels in {-1, +1}, with f = 0 assigned to +1."""
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)


def validate_inputs(X: np.ndarray, y: np.ndarray, C: float) -> None:
    """Fail fast on what every solver assumes: 2-D X, both labels in {-1, +1}, C > 0."""
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be (n, d) with n == len(y)")
    if set(np.unique(y).tolist()) != {-1.0, 1.0}:
        raise ValueError("y must contain both labels -1 and +1 and nothing else")
    if not C > 0:
        raise ValueError("C must be positive")


def signed_gram(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Q = diag(y) K diag(y)."""
    return (y[:, None] * y[None, :]) * K


def hinge_losses(margins: np.ndarray) -> np.ndarray:
    """xi_i = max(0, 1 - y_i f(x_i)), given margins y_i f(x_i)."""
    return np.maximum(0.0, 1.0 - margins)


def l2_dual_objective(alpha: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
    coef = alpha * y
    return float(alpha.sum() - 0.5 * coef @ K @ coef)


def l2_primal_objective(
    alpha: np.ndarray, b: float, K: np.ndarray, y: np.ndarray, C: float
) -> float:
    """P evaluated at f = sum a_i y_i k(x_i, .) with offset b."""
    coef = alpha * y
    g = K @ coef
    return float(0.5 * coef @ g + C * hinge_losses(y * (g + b)).sum())


def best_offset(g: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Minimise sum_i max(0, 1 - y_i (g_i + b)) over b exactly in O(n log n).

    The objective is convex and piecewise linear with breakpoints c_i = y_i - g_i,
    so its minimum is attained at one of them. Returns (b, minimal hinge sum).
    """
    c = y - g
    order = np.argsort(c, kind="stable")
    cs, ys = c[order], y[order]
    pos, neg = (ys > 0).astype(float), (ys < 0).astype(float)
    pos_sum_after = (pos * cs).sum() - np.cumsum(pos * cs)
    pos_cnt_after = pos.sum() - np.cumsum(pos)
    neg_sum_before = np.cumsum(neg * cs) - neg * cs
    neg_cnt_before = np.cumsum(neg) - neg
    totals = (pos_sum_after - cs * pos_cnt_after) + (cs * neg_cnt_before - neg_sum_before)
    k = int(np.argmin(totals))
    return float(cs[k]), float(totals[k])


def offset_from_kkt(
    alpha: np.ndarray, g: np.ndarray, y: np.ndarray, C: float, tol: float
) -> float:
    """b from KKT: average of y_i - g_i over free SVs (0 < a_i < C).

    If no multiplier is free, b lies in an interval and the midpoint of its
    KKT bounds is returned.
    """
    free = (alpha > tol) & (alpha < C - tol)
    candidates = y - g
    if np.any(free):
        return float(candidates[free].mean())
    at_lower = alpha <= tol
    at_upper = ~at_lower
    # y_i f_i >= 1 for a_i = 0 and <= 1 for a_i = C give lower/upper bounds on b.
    lower = np.concatenate([candidates[at_lower & (y > 0)], candidates[at_upper & (y < 0)]])
    upper = np.concatenate([candidates[at_lower & (y < 0)], candidates[at_upper & (y > 0)]])
    lo = lower.max() if lower.size else upper.min()
    hi = upper.min() if upper.size else lower.max()
    return float((lo + hi) / 2.0)
