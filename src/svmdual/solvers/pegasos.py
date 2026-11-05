"""Kernelized Pegasos: stochastic sub-gradient descent on the primal.

Shalev-Shwartz et al. (2011) minimise lambda/2 ||f||^2 + 1/n sum hinge with
no offset. With lambda = 1 / (n C) this is the L2-SVM primal divided by n C
with b fixed to 0, so the result is an *approximate* solution to a slightly
restricted problem. Its suboptimality P(f_T, 0) - P* is therefore always
non-negative and is what we report. There is no certificate of optimality,
so the run uses a fixed iteration budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from svmdual.kernels import Kernel
from svmdual.model import STATUS_FIXED_BUDGET, SVMSolution, l2_primal_objective, validate_inputs


@dataclass(frozen=True)
class PegasosConfig:
    """Fixed budget of epochs * n stochastic steps, sampled with the given seed."""

    epochs: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("PegasosConfig.epochs must be >= 1")


DEFAULT_PEGASOS = PegasosConfig()


def pegasos(
    X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, config: PegasosConfig = DEFAULT_PEGASOS
) -> SVMSolution:
    """Run T = epochs * n Pegasos steps; record the primal objective after each epoch."""
    validate_inputs(X, y, C)
    start = time.perf_counter()
    n = len(y)
    K = kernel(X, X)
    lam = 1.0 / (n * C)
    counts = np.zeros(n)
    unscaled = np.zeros(n)  # K @ (counts * y)
    samples = np.random.default_rng(config.seed).integers(0, n, size=config.epochs * n)
    objective_trace: list[float] = []
    for step, i in enumerate(samples, start=1):
        if y[i] * unscaled[i] / (lam * step) < 1.0:
            counts[i] += 1.0
            unscaled += y[i] * K[:, i]
        if step % n == 0:
            objective_trace.append(l2_primal_objective(counts / (lam * step), 0.0, K, y, C))
    alpha = counts / (lam * len(samples))
    return SVMSolution(
        solver="pegasos",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=0.0,
        X_train=X,
        y_train=y,
        primal_objective=l2_primal_objective(alpha, 0.0, K, y, C),
        dual_objective=float("nan"),
        runtime_s=time.perf_counter() - start,
        n_iter=len(samples),
        status=STATUS_FIXED_BUDGET,
        extras={"lambda": lam, "max_alpha_over_C": float(alpha.max() / C)},
        history={"primal": np.asarray(objective_trace)},
    )
