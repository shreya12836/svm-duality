"""Sequential Minimal Optimization with second-order working-set selection.

Implements WSS2 of Fan, Chen & Lin (JMLR 2005) for the L2-SVM dual

    min_a 1/2 a^T Q a - e^T a   s.t. 0 <= a_i <= C,  y^T a = 0.

Each iteration moves along d = y_i e_i - y_j e_j, which keeps y^T a fixed.
After every iteration we record:

* the duality gap P(a, b*) - D(a), where b* minimises the primal for the
  current f, so the gap is a valid upper bound on suboptimality;
* the maximal KKT violation m(a) - M(a) that drives the stopping rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from svmdual.kernels import Kernel
from svmdual.model import (
    STATUS_MAX_ITER,
    STATUS_OPTIMAL,
    SVMSolution,
    best_offset,
    l2_dual_objective,
    l2_primal_objective,
    signed_gram,
    validate_inputs,
)

TAU = 1e-12  # floor on the curvature a_ij for non-PD kernels / duplicate points
SNAP_REL = 1e-12  # snap alpha to 0 or C within this relative distance


@dataclass(frozen=True)
class SMOConfig:
    """Stopping rule m(a) - M(a) < tol, an iteration cap and whether to trace the gap."""

    tol: float = 1e-3
    max_iter: int = 200_000
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.tol <= 0 or self.max_iter < 1:
            raise ValueError("SMOConfig needs tol > 0 and max_iter >= 1")


DEFAULT_SMO = SMOConfig()


@dataclass(frozen=True)
class _WorkingSet:
    i: int
    j: int
    violation: float  # m(a) - M(a)


def _index_sets(alpha: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    """I_up and I_low of Keerthi et al.: directions in which a_t may increase y_t a_t or not."""
    below_c, above_0 = alpha < C, alpha > 0
    up = ((y > 0) & below_c) | ((y < 0) & above_0)
    low = ((y > 0) & above_0) | ((y < 0) & below_c)
    return up, low


def _select_working_set(
    alpha: np.ndarray, grad: np.ndarray, y: np.ndarray, K: np.ndarray, C: float
) -> _WorkingSet:
    up, low = _index_sets(alpha, y, C)
    score = -y * grad
    i = int(np.flatnonzero(up)[np.argmax(score[up])])
    m_alpha = score[i]
    violation = float(m_alpha - score[low].min())
    candidates = low & (score < m_alpha)
    if not np.any(candidates):
        return _WorkingSet(i, i, violation)
    curvature = K[i, i] + np.diag(K) - 2.0 * K[i]
    curvature = np.where(curvature > 0, curvature, TAU)
    decrease = np.where(candidates, -((m_alpha - score) ** 2) / curvature, np.inf)
    return _WorkingSet(i, int(np.argmin(decrease)), violation)


def _step_length(
    ws: _WorkingSet, alpha: np.ndarray, grad: np.ndarray, y: np.ndarray, K: np.ndarray, C: float
) -> float:
    """Exact line search along d = y_i e_i - y_j e_j, clipped to the box."""
    i, j = ws.i, ws.j
    slope = -y[i] * grad[i] + y[j] * grad[j]  # = b_ij > 0
    curvature = max(K[i, i] + K[j, j] - 2.0 * K[i, j], TAU)
    room_i = C - alpha[i] if y[i] > 0 else alpha[i]
    room_j = alpha[j] if y[j] > 0 else C - alpha[j]
    return float(min(slope / curvature, room_i, room_j))


def _gap_entry(
    alpha: np.ndarray, grad: np.ndarray, y: np.ndarray, C: float
) -> tuple[float, float]:
    """(primal with optimal b, dual) computed from the maintained gradient in O(n log n)."""
    quad = float(alpha @ (grad + 1.0))  # a^T Q a
    g = y * (grad + 1.0)  # g_i = sum_j a_j y_j K_ij
    _, hinge = best_offset(g, y)
    return 0.5 * quad + C * hinge, float(alpha.sum()) - 0.5 * quad


@dataclass(frozen=True)
class _LoopResult:
    n_iter: int
    status: str
    trace: dict[str, list[float]]


def smo(
    X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, config: SMOConfig = DEFAULT_SMO
) -> SVMSolution:
    """Solve the L2-SVM dual by SMO and return the fitted model with its gap trace.

    The returned offset b minimises the primal for the final f (as in the gap
    trace), so solution.duality_gap equals the last entry of history["gap"].
    """
    validate_inputs(X, y, C)
    start = time.perf_counter()
    K = kernel(X, X)
    alpha = np.zeros(len(y))
    loop = _iterate(alpha, signed_gram(K, y), K, y, C, config)
    g = K @ (alpha * y)
    b, _ = best_offset(g, y)
    return SVMSolution(
        solver="smo",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=b,
        X_train=X,
        y_train=y,
        primal_objective=l2_primal_objective(alpha, b, K, y, C),
        dual_objective=l2_dual_objective(alpha, K, y),
        runtime_s=time.perf_counter() - start,
        n_iter=loop.n_iter,
        status=loop.status,
        extras={"smo_tol": config.tol},
        history={key: np.asarray(values) for key, values in loop.trace.items()},
    )


def _iterate(
    alpha: np.ndarray, Q: np.ndarray, K: np.ndarray, y: np.ndarray, C: float, config: SMOConfig
) -> _LoopResult:
    """Main SMO loop; updates alpha in place (see _move) and returns the trace."""
    grad = -np.ones(len(y))
    trace: dict[str, list[float]] = {"gap": [], "violation": [], "primal": [], "dual": []}
    for iteration in range(config.max_iter):
        ws = _select_working_set(alpha, grad, y, K, C)
        if config.record_history:
            primal, dual = _gap_entry(alpha, grad, y, C)
            for key, value in zip(trace, (primal - dual, ws.violation, primal, dual), strict=True):
                trace[key].append(value)
        if ws.violation < config.tol or ws.i == ws.j:
            return _LoopResult(iteration, STATUS_OPTIMAL, trace)
        t = _step_length(ws, alpha, grad, y, K, C)
        delta_i = _move(alpha, ws.i, t * y[ws.i], C)
        delta_j = _move(alpha, ws.j, -t * y[ws.j], C)
        grad += Q[:, ws.i] * delta_i + Q[:, ws.j] * delta_j
    return _LoopResult(config.max_iter, STATUS_MAX_ITER, trace)


def _move(alpha: np.ndarray, idx: int, delta: float, C: float) -> float:
    """Apply a_idx += delta, snapping to {0, C} within rounding; return the actual change.

    Snapping matters: a value left at C - 1e-16 would stay in I_up with zero
    room and stall the iteration. Each snap perturbs y^T a by at most
    1e-12 * C, which is negligible next to the solver tolerance. This is the
    one in-place update in the package, kept local to the loop for O(1) steps.
    """
    snap = SNAP_REL * max(C, 1.0)
    new = alpha[idx] + delta
    new = 0.0 if new < snap else (C if new > C - snap else new)
    actual = new - alpha[idx]
    alpha[idx] = new
    return float(actual)
