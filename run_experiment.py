"""Reproduce every table and figure of the project.

Usage:
    python run_experiment.py                         # default config, seed 42
    python run_experiment.py --seed 7 --out results_seed7
    python run_experiment.py --config configs/default.yaml --no-memory
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from svmdual import experiment as exp  # noqa: E402
from svmdual import report  # noqa: E402
from svmdual.config import ExperimentConfig, load_config  # noqa: E402
from svmdual.data import Dataset, load_dataset  # noqa: E402
from svmdual.kernels import Kernel  # noqa: E402
from svmdual.sensitivity import sweep  # noqa: E402

log = logging.getLogger("svmdual")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--out", type=Path, default=None, help="override the output directory")
    parser.add_argument("--no-memory", action="store_true", help="skip the peak-memory pass")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config, root=ROOT)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.out is not None:
        cfg = replace(cfg, output_dir=args.out.resolve())
    return cfg


def load_data(cfg: ExperimentConfig) -> tuple[Dataset, dict[str, object]]:
    d = cfg.data
    data = load_dataset(d.path, d.features, d.val_fraction, d.test_fraction, cfg.seed, d.sha256)
    info = {
        "raw rows": data.n_raw,
        "duplicates removed": data.n_duplicates_removed,
        "train / val / test": f"{data.train.n} / {data.val.n} / {data.test.n}",
        "features": ", ".join(data.feature_names),
        "sha256": d.sha256,
    }
    return data, info


def run_selected(
    data: Dataset, cfg: ExperimentConfig, selection: dict, measure_memory: bool
) -> dict[str, dict[str, exp.Run]]:
    log.info("Warm-up solves (excluded from timing)")
    exp.warm_up(data, cfg)
    runs = {}
    for name, (C, gamma) in selection.items():
        kernel = Kernel("linear") if name == "linear" else Kernel("rbf", gamma)
        log.info("Running all solvers: %s, C = %g", kernel.label, C)
        runs[name] = exp.run_suite(data, kernel, C, cfg, measure_memory)
    return runs


def pegasos_study(data: Dataset, cfg: ExperimentConfig, runs: dict) -> pd.DataFrame:
    rows = []
    for kernel_runs in runs.values():
        exact = kernel_runs["cvx_primal"].solution
        rows += exp.pegasos_seed_study(data, exact.kernel, exact.C, cfg, exact.primal_objective)
    return pd.DataFrame(rows)


def write_log(cfg: ExperimentConfig, started: str, extra: dict[str, object]) -> None:
    payload = {
        "started_utc": started,
        "finished_utc": report.utc_now(),
        "seed": cfg.seed,
        "config": cfg,
        "hardware": report.hardware_info(),
        "packages": report.package_versions(),
        **extra,
    }
    report.write_json(payload, cfg.output_dir / "experiment_log.json")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    cfg, started = build_config(args), report.utc_now()
    git = report.git_state(ROOT)  # before any output is written, so "dirty" means code changes
    out = cfg.output_dir
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    log.info("Loading data (seed %d)", cfg.seed)
    data, data_info = load_data(cfg)

    log.info("Sensitivity sweep and validation-based selection")
    sel = cfg.selection
    grid, sweep_time = exp.timed(lambda: sweep(data, sel.C_grid, sel.gamma_grid))
    grid.to_csv(tables / "sensitivity_grid.csv", index=False)
    selection = exp.selected_settings(cfg, grid)
    log.info("Selected: %s", selection)

    runs = run_selected(data, cfg, selection, measure_memory=not args.no_memory)
    comparison, kkt = exp.comparison_tables(runs, data, cfg)
    comparison.to_csv(tables / "solver_comparison.csv", index=False)
    kkt.to_csv(tables / "kkt_residuals.csv", index=False)
    log.info("Pegasos seed study (%d seeds)", cfg.solvers.pegasos_seeds)
    seeds = pegasos_study(data, cfg, runs)
    seeds.to_csv(tables / "pegasos_seeds.csv", index=False)

    log.info("Drawing figures")
    figure_paths = (
        exp.geometry_figures(cfg, figures)
        + exp.solver_figures(runs, figures)
        + exp.sensitivity_figures(grid, figures)
    )
    report.write_summary(comparison, kkt, seeds, data_info, selection, out / "summary.md")
    run_columns = ["solver", "kernel", "C", "status", "n_iter", "runtime_s", "py_heap_peak_mb"]
    write_log(
        cfg,
        started,
        {
            "git": git,
            "data": data_info,
            "selection": selection,
            "sweep_runtime_s": sweep_time,
            "runs": comparison[run_columns].to_dict(orient="records"),
            "figures": [p.relative_to(out).as_posix() for p in figure_paths],
        },
    )
    log.info("Done. Results in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
