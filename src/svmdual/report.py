"""Writing tables, the markdown summary and the experiment log."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "clarabel", "matplotlib")
SUMMARY_COLUMNS = (
    "solver",
    "kernel",
    "C",
    "status",
    "n_iter",
    "runtime_s",
    "py_heap_peak_mb",
    "primal_objective",
    "dual_objective",
    "duality_gap",
    "suboptimality",
    "n_sv",
    "sparsity_ratio",
    "margin_width",
    "margin_norm",
    "n_nonzero_w",
    "test_balanced_accuracy",
    "test_bal_acc_ci_low",
    "test_bal_acc_ci_high",
    "test_macro_f1",
    "test_roc_auc",
    "test_pr_auc",
)
KKT_COLUMNS = (
    "solver",
    "kernel",
    "dual_box_violation",
    "equality_residual",
    "stationarity_w",
    "stationarity_xi",
    "comp_slack_margin",
    "comp_slack_slack",
    "max_violating_pair",
    "duality_gap",
    "relative_gap",
)
BYTES_PER_GB = 1024**3
GIT_TIMEOUT_S = 10


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / BYTES_PER_GB, 2)
    except ImportError:
        info["ram_gb"] = None
    return info


def git_state(repo: Path) -> dict[str, Any]:
    """Commit hash and whether the working tree had uncommitted changes (None if no git)."""

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
                timeout=GIT_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    present = [c for c in columns if c in frame.columns]
    lines = ["| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
    for _, row in frame[present].iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in present) + " |")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):  # NumPy scalar
        return _jsonable(value.item())
    return value


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


NOTES = """\
Notes on reading the tables:

- `py_heap_peak_mb` is the peak Python/NumPy heap from `tracemalloc` in a separate run. It
  does **not** include memory allocated inside native solvers (Clarabel, HiGHS, libsvm), so it
  is not a full memory comparison.
- `runtime_s` is the median of repeated runs after a warm-up. `smo` and `pegasos` are pure
  Python loops; `libsvm` is compiled C++, so their runtimes compare implementations, not
  algorithms.
- `suboptimality` is P(solution) - P(cvx_primal). Pegasos has no offset b, so part of its
  suboptimality comes from the restricted model; its dual objective is undefined and its
  "KKT residuals" are large by construction (no box or equality constraint is enforced).
- `l1_svm` minimises ||w||_1 + C sum xi, so its objective is not comparable with the L2 rows,
  and its `margin_width` is measured in the l_inf norm (`margin_norm`).
- The CI columns are 95% percentile-bootstrap intervals for test balanced accuracy."""


def _pegasos_summary(seeds: pd.DataFrame) -> str:
    agg = seeds.groupby(["kernel", "C"], as_index=False).agg(
        seeds=("seed", "count"),
        rel_subopt_mean=("relative_suboptimality", "mean"),
        rel_subopt_sd=("relative_suboptimality", "std"),
        sparsity_mean=("sparsity_ratio", "mean"),
        test_bal_acc_mean=("test_balanced_accuracy", "mean"),
        test_bal_acc_sd=("test_balanced_accuracy", "std"),
    )
    return markdown_table(agg, tuple(agg.columns))


def write_summary(
    comparison: pd.DataFrame,
    kkt: pd.DataFrame,
    seeds: pd.DataFrame,
    data_info: dict[str, Any],
    selection: dict[str, Any],
    path: Path,
) -> Path:
    """Human-readable results page regenerated on every run."""
    chosen = "\n".join(
        f"- **{k}**: C = {v[0]:g}" + (f", gamma = {v[1]:g}" if v[1] else "")
        for k, v in selection.items()
    )
    data_lines = "\n".join(f"- **{k}**: {v}" for k, v in data_info.items())
    sections = [
        "# Results summary",
        "Generated by `run_experiment.py`. Every number below is regenerated on each run.",
        f"## Data\n\n{data_lines}",
        "## Selected hyperparameters\n\nValidation balanced accuracy, ties broken by "
        "validation ROC-AUC, then fewest support vectors, then smaller C and gamma.\n\n" + chosen,
        "## Solver comparison (test metrics on the untouched split)\n\n"
        + markdown_table(comparison, SUMMARY_COLUMNS),
        NOTES,
        "## KKT residual report (training set; 0 = exact)\n\n" + markdown_table(kkt, KKT_COLUMNS),
        "## Pegasos over seeds (relative suboptimality = (P - P*) / P*)\n\n"
        + _pegasos_summary(seeds),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
