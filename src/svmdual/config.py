"""Typed, validated experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from svmdual.data import FEATURE_NAMES


@dataclass(frozen=True)
class DataConfig:
    path: Path
    sha256: str | None
    features: tuple[str, ...]
    val_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class SelectionConfig:
    C_grid: tuple[float, ...]
    gamma_grid: tuple[float, ...]
    linear_C: float | None
    rbf_C: float | None
    rbf_gamma: float | None


@dataclass(frozen=True)
class SolverConfig:
    cvx_solver: str
    smo_tol: float
    smo_max_iter: int
    pegasos_epochs: int
    pegasos_seeds: int
    timing_repeats: int


@dataclass(frozen=True)
class PlotConfig:
    features: tuple[str, str]
    linear_C: float
    rbf_C: float
    rbf_gamma: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    output_dir: Path
    data: DataConfig
    selection: SelectionConfig
    solvers: SolverConfig
    plots: PlotConfig
    bootstrap_resamples: int


def _positive_tuple(values: Any, name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if not result or min(result) <= 0:
        raise ValueError(f"{name} must be a non-empty list of positive numbers")
    return result


def _optional_positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive or null")
    return float(value)


def _at_least_one(value: Any, name: str) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(value)


def _features(values: Any, name: str) -> tuple[str, ...]:
    result = tuple(str(v) for v in values)
    unknown = set(result) - set(FEATURE_NAMES)
    if unknown or not result:
        raise ValueError(f"{name}: unknown or empty features {sorted(unknown)}")
    return result


def _build(raw: dict[str, Any], root: Path) -> ExperimentConfig:
    data, sel, sol, plot = raw["data"], raw["selection"], raw["solvers"], raw["plots"]
    plot_features = _features(plot["features"], "plots.features")
    if len(plot_features) != 2:
        raise ValueError("plots.features must name exactly two features")
    return ExperimentConfig(
        seed=int(raw["seed"]),
        output_dir=root / raw.get("output_dir", "results"),
        data=DataConfig(
            path=root / data["path"],
            sha256=data.get("sha256"),
            features=_features(data["features"], "data.features"),
            val_fraction=float(data["val_fraction"]),
            test_fraction=float(data["test_fraction"]),
        ),
        selection=SelectionConfig(
            C_grid=_positive_tuple(sel["C_grid"], "selection.C_grid"),
            gamma_grid=_positive_tuple(sel["gamma_grid"], "selection.gamma_grid"),
            linear_C=_optional_positive(sel.get("linear_C"), "selection.linear_C"),
            rbf_C=_optional_positive(sel.get("rbf_C"), "selection.rbf_C"),
            rbf_gamma=_optional_positive(sel.get("rbf_gamma"), "selection.rbf_gamma"),
        ),
        solvers=SolverConfig(
            cvx_solver=str(sol["cvx_solver"]),
            smo_tol=float(sol["smo_tol"]),
            smo_max_iter=int(sol["smo_max_iter"]),
            pegasos_epochs=int(sol["pegasos_epochs"]),
            pegasos_seeds=_at_least_one(sol["pegasos_seeds"], "solvers.pegasos_seeds"),
            timing_repeats=_at_least_one(sol["timing_repeats"], "solvers.timing_repeats"),
        ),
        plots=PlotConfig(
            features=(plot_features[0], plot_features[1]),
            linear_C=float(plot["linear_C"]),
            rbf_C=float(plot["rbf_C"]),
            rbf_gamma=float(plot["rbf_gamma"]),
        ),
        bootstrap_resamples=_at_least_one(
            raw["evaluation"]["bootstrap_resamples"], "evaluation.bootstrap_resamples"
        ),
    )


def load_config(path: Path, root: Path | None = None) -> ExperimentConfig:
    """Load YAML; relative paths inside it resolve against `root` (default: cwd)."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        return _build(raw, root or Path.cwd())
    except KeyError as missing:
        raise ValueError(f"Missing config key: {missing}") from missing
