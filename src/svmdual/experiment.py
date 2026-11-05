"""End-to-end experiment: selection sweep, solver comparison, KKT report, figures, log.

Order of work (the test split is used only in step 3, after selection):

1. load + verify + split the data;
2. sweep C (linear) and C x gamma (RBF) with libsvm, select on validation;
3. fit all solvers at the selected settings, evaluate on train / val / test;
4. draw the geometry, gap-curve, Pegasos and sensitivity figures;
5. write tables, a markdown summary and the experiment log.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from svmdual import plots
from svmdual.config import ExperimentConfig
from svmdual.data import Dataset, load_dataset
from svmdual.kernels import Kernel
from svmdual.kkt import kkt_report, margin_width, sparsity
from svmdual.metrics import bootstrap_balanced_accuracy_ci, classification_metrics
from svmdual.model import SVMSolution
from svmdual.sensitivity import fit_libsvm, select
from svmdual.solvers import PegasosConfig, SMOConfig, cvx_dual, cvx_primal, l1_svm, pegasos, smo

SolverFn = Callable[[np.ndarray, np.ndarray, float, Kernel], SVMSolution]
BYTES_PER_MB = 1024**2
WARMUP_PER_CLASS = 10


@dataclass(frozen=True)
class Run:
    solution: SVMSolution
    py_heap_peak_mb: float


def solver_suite(kernel: Kernel, cfg: ExperimentConfig) -> dict[str, SolverFn]:
    """The solvers compared for one kernel (the 1-norm LP only for the linear kernel)."""
    s = cfg.solvers
    smo_cfg = SMOConfig(tol=s.smo_tol, max_iter=s.smo_max_iter)
    peg_cfg = PegasosConfig(epochs=s.pegasos_epochs, seed=cfg.seed)
    suite: dict[str, SolverFn] = {
        "cvx_primal": lambda X, y, C, k: cvx_primal(X, y, C, k, s.cvx_solver),
        "cvx_dual": lambda X, y, C, k: cvx_dual(X, y, C, k, s.cvx_solver),
        "smo": lambda X, y, C, k: smo(X, y, C, k, smo_cfg),
        "pegasos": lambda X, y, C, k: pegasos(X, y, C, k, peg_cfg),
        "libsvm": fit_libsvm,
    }
    if kernel.name == "linear":
        suite["l1_svm"] = l1_svm
    return suite


def peak_memory_mb(fn: SolverFn, X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel) -> float:
    """Peak Python-heap allocation (NumPy included; native solver internals are not)."""
    tracemalloc.start()
    try:
        fn(X, y, C, kernel)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / BYTES_PER_MB


def warm_up(data: Dataset, cfg: ExperimentConfig) -> None:
    """Solve tiny problems once so CVXPY / solver start-up cost is not timed."""
    y = data.train.y
    idx = np.r_[np.flatnonzero(y > 0)[:WARMUP_PER_CLASS], np.flatnonzero(y < 0)[:WARMUP_PER_CLASS]]
    for fn in solver_suite(Kernel("linear"), cfg).values():
        fn(data.train.X[idx], y[idx], 1.0, Kernel("linear"))


def timed_solve(
    fn: SolverFn, X: np.ndarray, y: np.ndarray, C: float, kernel: Kernel, repeats: int
) -> SVMSolution:
    """Run `repeats` times; keep the first solution with the median runtime."""
    solutions = [fn(X, y, C, kernel) for _ in range(repeats)]
    median = float(np.median([sol.runtime_s for sol in solutions]))
    return replace(solutions[0], runtime_s=median)


def run_suite(
    data: Dataset, kernel: Kernel, C: float, cfg: ExperimentConfig, measure_memory: bool
) -> dict[str, Run]:
    """All solvers on the training split; memory is measured in a separate traced run."""
    X, y = data.train.X, data.train.y
    runs = {}
    for name, fn in solver_suite(kernel, cfg).items():
        solution = timed_solve(fn, X, y, C, kernel, cfg.solvers.timing_repeats)
        memory = peak_memory_mb(fn, X, y, C, kernel) if measure_memory else np.nan
        runs[name] = Run(solution, memory)
    return runs


def pegasos_seed_study(
    data: Dataset, kernel: Kernel, C: float, cfg: ExperimentConfig, optimum: float
) -> list[dict[str, float | str]]:
    """Pegasos over several seeds: is the gap to P* a property of the method or of one run?"""
    rows = []
    for offset in range(cfg.solvers.pegasos_seeds):
        seed = cfg.seed + offset
        sol = pegasos(
            data.train.X,
            data.train.y,
            C,
            kernel,
            PegasosConfig(epochs=cfg.solvers.pegasos_epochs, seed=seed),
        )
        test = classification_metrics(data.test.y, sol.decision_function(data.test.X))
        rows.append(
            {
                "kernel": kernel.label,
                "C": C,
                "seed": seed,
                "relative_suboptimality": (sol.primal_objective - optimum) / optimum,
                "sparsity_ratio": sparsity(sol.alpha, C).ratio,
                "test_balanced_accuracy": test.balanced_accuracy,
            }
        )
    return rows


def evaluate(
    run: Run, data: Dataset, optimum: float, cfg: ExperimentConfig
) -> dict[str, float | str | None]:
    """One row of the solver-comparison table."""
    sol = run.solution
    is_l1 = sol.solver == "l1_svm"
    sp = sparsity(sol.alpha, sol.C)
    row: dict[str, float | str | None] = {
        "solver": sol.solver,
        "kernel": sol.kernel.label,
        "C": sol.C,
        "status": sol.status,
        "n_iter": sol.n_iter,
        "runtime_s": sol.runtime_s,
        "py_heap_peak_mb": run.py_heap_peak_mb,
        "primal_objective": sol.primal_objective,
        "dual_objective": sol.dual_objective,
        "duality_gap": sol.duality_gap,
        "suboptimality": np.nan if is_l1 else sol.primal_objective - optimum,
        "n_sv": sp.n_sv,
        "n_free": sp.n_free,
        "n_bounded": sp.n_bounded,
        "sparsity_ratio": sp.ratio,
        "margin_width": margin_width(sol),
        "margin_norm": "linf" if is_l1 else "l2",
        "n_nonzero_w": sol.extras.get("n_nonzero_w", np.nan),
    }
    for split_name in ("train", "val", "test"):
        split = getattr(data, split_name)
        scores = sol.decision_function(split.X)
        row.update(classification_metrics(split.y, scores).as_dict(f"{split_name}_"))
    low, high = bootstrap_balanced_accuracy_ci(
        data.test.y, sol.decision_function(data.test.X), cfg.bootstrap_resamples, cfg.seed
    )
    return {**row, "test_bal_acc_ci_low": low, "test_bal_acc_ci_high": high}


def selected_settings(
    cfg: ExperimentConfig, grid: pd.DataFrame
) -> dict[str, tuple[float, float | None]]:
    """Validation-selected (C, gamma) per kernel, unless fixed in the config."""
    sel = cfg.selection
    linear = (sel.linear_C, None) if sel.linear_C else select(grid, "linear")
    rbf_auto = select(grid, "rbf")
    rbf = (sel.rbf_C or rbf_auto[0], sel.rbf_gamma or rbf_auto[1])
    return {"linear": linear, "rbf": rbf}


def geometry_figures(cfg: ExperimentConfig, figures: Path) -> list[Path]:
    """2-D models on the two plot features: hyperplane, reduced hulls, kernel boundary."""
    p = cfg.plots
    data_2d = load_dataset(
        cfg.data.path,
        p.features,
        cfg.data.val_fraction,
        cfg.data.test_fraction,
        cfg.seed,
        cfg.data.sha256,
    )
    X, y = data_2d.train.X, data_2d.train.y
    linear = cvx_primal(X, y, p.linear_C, Kernel("linear"), cfg.solvers.cvx_solver)
    rbf = cvx_dual(X, y, p.rbf_C, Kernel("rbf", p.rbf_gamma), cfg.solvers.cvx_solver)
    return [
        plots.hyperplane_figure(linear, p.features, figures / "separating_hyperplane.png"),
        plots.kernel_boundary_figure(rbf, p.features, figures / "rbf_margin_support_vectors.png"),
    ]


def solver_figures(runs: dict[str, dict[str, Run]], figures: Path) -> list[Path]:
    """Gap curves from SMO and Pegasos convergence, one line per kernel."""
    smo_runs = {kernel: r["smo"].solution for kernel, r in runs.items()}
    pegasos_traces = {
        kernel: (
            r["pegasos"].solution.history["primal"],
            r["cvx_primal"].solution.primal_objective,
        )
        for kernel, r in runs.items()
    }
    return [
        plots.gap_curve_figure(smo_runs, figures / "primal_dual_gap_curve.png"),
        plots.pegasos_figure(pegasos_traces, figures / "pegasos_vs_exact.png"),
    ]


def sensitivity_figures(grid: pd.DataFrame, figures: Path) -> list[Path]:
    return [
        plots.linear_sensitivity_figure(
            grid[grid["kernel"] == "linear"], figures / "sensitivity_linear_C.png"
        ),
        plots.rbf_heatmaps_figure(
            grid[grid["kernel"] == "rbf"], figures / "sensitivity_rbf_C_gamma.png"
        ),
    ]


def comparison_tables(
    runs: dict[str, dict[str, Run]], data: Dataset, cfg: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, kkt_rows = [], []
    for kernel_runs in runs.values():
        optimum = kernel_runs["cvx_primal"].solution.primal_objective
        for run in kernel_runs.values():
            rows.append(evaluate(run, data, optimum, cfg))
            kkt_rows.append(kkt_report(run.solution).as_dict())
    return pd.DataFrame(rows), pd.DataFrame(kkt_rows)


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
