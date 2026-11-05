"""Sensitivity of sparsity, margin and accuracy to C and gamma, and model selection.

The grid is fitted with scikit-learn's SVC (libsvm), an independent SMO
implementation, so the sweep also serves as an external reference. Each fit is
converted to an SVMSolution and analysed with the same KKT and sparsity code
as our own solvers.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

from svmdual.data import Dataset
from svmdual.kernels import Kernel
from svmdual.kkt import margin_width, sparsity
from svmdual.metrics import classification_metrics
from svmdual.model import (
    STATUS_MAX_ITER,
    STATUS_OPTIMAL,
    SVMSolution,
    l2_dual_objective,
    l2_primal_objective,
    validate_inputs,
)

LIBSVM_TOL = 1e-5
LIBSVM_CACHE_MB = 500


def fit_libsvm(X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel) -> SVMSolution:
    """Fit sklearn.svm.SVC and express the result as an SVMSolution.

    Runtime covers libsvm's fit, including its own kernel evaluations; the Gram
    matrix built afterwards is only for evaluating the objectives.
    """
    validate_inputs(X, y, C)
    start = time.perf_counter()
    model = SVC(
        C=C,
        kernel=kernel.name,
        gamma=kernel.gamma if kernel.gamma is not None else "scale",
        tol=LIBSVM_TOL,
        cache_size=LIBSVM_CACHE_MB,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    runtime = time.perf_counter() - start
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    alpha = np.zeros(len(y))
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    b = float(model.intercept_[0])
    K = kernel(X, X)
    return SVMSolution(
        solver="libsvm",
        kernel=kernel,
        C=C,
        alpha=alpha,
        b=b,
        X_train=X,
        y_train=y,
        primal_objective=l2_primal_objective(alpha, b, K, y, C),
        dual_objective=l2_dual_objective(alpha, K, y),
        runtime_s=runtime,
        n_iter=int(model.n_iter_[0]),
        status=STATUS_OPTIMAL if converged else STATUS_MAX_ITER,
    )


def sweep_row(data: Dataset, kernel: Kernel, C: float) -> dict[str, float | str]:
    """Fit one grid point on train and summarise sparsity, margin and train/val metrics."""
    sol = fit_libsvm(data.train.X, data.train.y, C, kernel)
    sp = sparsity(sol.alpha, C)
    return {
        "kernel": kernel.name,
        "C": C,
        "gamma": kernel.gamma if kernel.gamma is not None else np.nan,
        "status": sol.status,
        "n_sv": sp.n_sv,
        "n_free": sp.n_free,
        "n_bounded": sp.n_bounded,
        "sparsity_ratio": sp.ratio,
        "margin_width": margin_width(sol),
        "primal_objective": sol.primal_objective,
        "duality_gap": sol.duality_gap,
        "fit_time_s": sol.runtime_s,
        **classification_metrics(data.train.y, sol.decision_function(data.train.X)).as_dict(
            "train_"
        ),
        **classification_metrics(data.val.y, sol.decision_function(data.val.X)).as_dict("val_"),
    }


def sweep(data: Dataset, C_grid: tuple[float, ...], gamma_grid: tuple[float, ...]) -> pd.DataFrame:
    """Linear kernel over C, and RBF over the full C x gamma grid."""
    kernels = [Kernel("linear")] + [Kernel("rbf", g) for g in gamma_grid]
    rows = [sweep_row(data, kernel, C) for kernel in kernels for C in C_grid]
    return pd.DataFrame(rows)


def select(results: pd.DataFrame, kernel_name: str) -> tuple[float, float | None]:
    """Best validation balanced accuracy; ties broken by ROC-AUC, then fewest SVs.

    Many grid cells reach the same validation accuracy on 270 points, so the
    tie-break matters. Fewest support vectors is principled: the leave-one-out
    error of an SVM is bounded by #SV / n (Vapnik), and sparsity is the
    property this project studies. Remaining ties go to smaller C, then gamma.
    Only the validation split is used; the test split is never consulted.
    """
    subset = results[results["kernel"] == kernel_name]
    if subset.empty:
        raise ValueError(f"No sweep results for kernel {kernel_name!r}")
    ranked = subset.sort_values(
        ["val_balanced_accuracy", "val_roc_auc", "n_sv", "C", "gamma"],
        ascending=[False, False, True, True, True],
    )
    best = ranked.iloc[0]
    gamma = None if np.isnan(best["gamma"]) else float(best["gamma"])
    return float(best["C"]), gamma
