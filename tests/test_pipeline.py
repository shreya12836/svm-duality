import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from svmdual import plots
from svmdual.config import load_config
from svmdual.kernels import Kernel
from svmdual.report import _jsonable, markdown_table
from svmdual.sensitivity import fit_libsvm, select
from svmdual.solvers import SMOConfig, cvx_primal, smo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG, root=ROOT)
    assert cfg.data.path.is_file()


def test_config_rejects_nonpositive_grid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C_grid"):
        load_config(_write_config(tmp_path, selection__C_grid=[0, 1]), root=ROOT)


def test_config_rejects_three_plot_features(tmp_path: Path) -> None:
    path = _write_config(tmp_path, plots__features=["variance", "skewness", "entropy"])
    with pytest.raises(ValueError, match="exactly two"):
        load_config(path, root=ROOT)


def test_config_reports_missing_key(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    del raw["solvers"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Missing config key"):
        load_config(path, root=ROOT)


def test_libsvm_wrapper_matches_smo(toy) -> None:
    ref = fit_libsvm(toy.X, toy.y, 1.0, Kernel("rbf", 0.5))
    ours = smo(toy.X, toy.y, 1.0, Kernel("rbf", 0.5), SMOConfig(tol=1e-8))
    assert ref.dual_objective == pytest.approx(ours.dual_objective, rel=1e-5)


def _grid(n_sv: list[int], accuracy: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kernel": ["rbf"] * len(n_sv),
            "C": [10.0, 1.0, 0.1][: len(n_sv)],
            "gamma": [1.0] * len(n_sv),
            "n_sv": n_sv,
            "val_balanced_accuracy": accuracy,
            "val_roc_auc": [1.0] * len(n_sv),
        }
    )


def test_select_prefers_accuracy() -> None:
    assert select(_grid([5, 50, 1], [1.0, 1.0, 0.9]), "rbf")[0] == 10.0


def test_select_breaks_ties_by_fewest_support_vectors() -> None:
    assert select(_grid([50, 5, 1], [1.0, 1.0, 0.9]), "rbf")[0] == 1.0


def test_select_breaks_remaining_ties_by_smaller_c() -> None:
    assert select(_grid([5, 5], [1.0, 1.0]), "rbf")[0] == 1.0


def test_select_rejects_missing_kernel() -> None:
    with pytest.raises(ValueError):
        select(pd.DataFrame({"kernel": ["rbf"]}), "linear")


def test_markdown_table_formats_nan_as_dash() -> None:
    table = markdown_table(pd.DataFrame({"a": [float("nan")]}), ("a",))
    assert table.splitlines()[-1] == "| – |"


def test_jsonable_handles_numpy_and_paths() -> None:
    payload = _jsonable({"x": np.float64(1.5), "p": Path("a/b"), "n": float("nan")})
    assert payload == {"x": 1.5, "p": "a/b", "n": None}


def test_geometry_figures_are_written(toy, tmp_path: Path) -> None:
    linear = cvx_primal(toy.X, toy.y, 1.0, Kernel("linear"))
    rbf = smo(toy.X, toy.y, 1.0, Kernel("rbf", 0.5))
    paths = [
        plots.hyperplane_figure(linear, ("a", "b"), tmp_path / "h.png"),
        plots.kernel_boundary_figure(rbf, ("a", "b"), tmp_path / "k.png"),
        plots.gap_curve_figure({"rbf": rbf}, tmp_path / "g.png"),
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_hyperplane_figure_requires_w(toy, tmp_path: Path) -> None:
    rbf = smo(toy.X, toy.y, 1.0, Kernel("rbf", 0.5))
    with pytest.raises(ValueError):
        plots.hyperplane_figure(rbf, ("a", "b"), tmp_path / "h.png")


@pytest.mark.integration
def test_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline on the real data with a 2 x 2 grid."""
    config = _write_config(tmp_path, selection__C_grid=[1, 10], selection__gamma_grid=[0.5, 1])
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("run_experiment", None)
    import run_experiment

    exit_code = run_experiment.main(
        ["--config", str(config), "--out", str(tmp_path / "out"), "--no-memory"]
    )
    log = json.loads((tmp_path / "out" / "experiment_log.json").read_text())
    assert exit_code == 0 and len(log["figures"]) == 6 and log["seed"] == 42


def test_git_state_without_repository(tmp_path: Path) -> None:
    from svmdual.report import git_state

    assert git_state(tmp_path)["commit"] is None


def test_config_rejects_zero_timing_repeats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timing_repeats"):
        load_config(_write_config(tmp_path, solvers__timing_repeats=0), root=ROOT)
