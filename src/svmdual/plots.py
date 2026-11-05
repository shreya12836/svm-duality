"""All report figures. Every function takes fitted results and writes one PNG.

Colours: the two classes use the first two slots of a CVD-validated categorical
palette and also differ by marker shape, so identity never relies on colour
alone. Heatmaps use one sequential hue, light to dark.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from svmdual.kkt import reduced_hull_points  # noqa: E402
from svmdual.model import SVMSolution  # noqa: E402
from svmdual.solvers.cvx import SV_TOL_REL  # noqa: E402

CLASS_POS = "#2a78d6"  # categorical slot 1
CLASS_NEG = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL_CMAP = "Blues"
DPI = 200
GRID_RESOLUTION = 300
PLOT_PAD = 0.5
CLASS_STYLE = {
    1.0: {"color": CLASS_POS, "marker": "o", "label": "class 1 (y = +1)"},
    -1.0: {"color": CLASS_NEG, "marker": "^", "label": "class 0 (y = -1)"},
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _scatter_classes(ax: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    for label, style in CLASS_STYLE.items():
        pts = X[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.55, linewidths=0, **style)


def _circle_support_vectors(ax: plt.Axes, sol: SVMSolution) -> None:
    tol = SV_TOL_REL * sol.C
    free = (sol.alpha > tol) & (sol.alpha < sol.C - tol)
    bounded = sol.alpha >= sol.C - tol
    for mask, style, name in ((free, "-", "free SV (0<a<C)"), (bounded, ":", "bounded SV (a=C)")):
        pts = sol.X_train[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=70,
            facecolors="none",
            edgecolors=INK,
            linewidths=1.0,
            linestyle=style,
            label=f"{name}: {mask.sum()}",
        )


def _legend_below(ax: plt.Axes) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)


def _decision_grid(sol: SVMSolution, X: np.ndarray) -> tuple[np.ndarray, ...]:
    lo, hi = X.min(axis=0) - PLOT_PAD, X.max(axis=0) + PLOT_PAD
    xx, yy = np.meshgrid(
        np.linspace(lo[0], hi[0], GRID_RESOLUTION), np.linspace(lo[1], hi[1], GRID_RESOLUTION)
    )
    zz = sol.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, zz


def _margin_contours(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> None:
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-1.0, 0.0, 1.0],
        colors=INK,
        linestyles=["--", "-", "--"],
        linewidths=[1.0, 2.0, 1.0],
    )


def hyperplane_figure(sol: SVMSolution, feature_names: tuple[str, str], path: Path) -> Path:
    """Linear SVM in 2-D: <w,x>+b = 0 and +-1, normal w, SVs and reduced-hull closest points."""
    if sol.w is None:
        raise ValueError("hyperplane_figure needs an explicit weight vector w")
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    X = sol.X_train
    _scatter_classes(ax, X, sol.y_train)
    _margin_contours(ax, *_decision_grid(sol, X))
    _circle_support_vectors(ax, sol)
    c, d, D = reduced_hull_points(sol)
    ax.plot(*np.c_[c, d], color=INK, linewidth=1.5, zorder=4)
    # d is drawn larger and underneath so that c stays visible when the points nearly coincide.
    for point, colour, size, layer, name in (
        (d, CLASS_NEG, 110, 5, "d (class 0 hull)"),
        (c, CLASS_POS, 40, 6, "c (class 1 hull)"),
    ):
        ax.scatter(
            *point,
            marker="s",
            s=size,
            color=colour,
            edgecolors=INK,
            linewidths=1.2,
            zorder=layer,
            label=f"reduced-hull closest point {name}",
        )
    ax.plot([], [], " ", label=f"reduced-hull weight bound D = C/s = {D:.3g};  w = s(c - d)")
    anchor = -sol.b * sol.w / (sol.w @ sol.w)  # foot of the normal on the hyperplane
    unit = sol.w / np.linalg.norm(sol.w)
    ax.annotate(
        "",
        xy=anchor + unit,
        xytext=anchor,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5},
    )
    ax.text(*(anchor + 1.08 * unit), "w", color=INK, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel(f"{feature_names[0]} (standardised)")
    ax.set_ylabel(f"{feature_names[1]} (standardised)")
    ax.set_title(f"Separating hyperplane <w,x> + b = 0 and margins ±1  (linear, C = {sol.C:g})")
    _legend_below(ax)
    return _save(fig, path)


def kernel_boundary_figure(sol: SVMSolution, feature_names: tuple[str, str], path: Path) -> Path:
    """RBF SVM in 2-D: level sets f = 0, +-1 and support vectors."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xx, yy, zz = _decision_grid(sol, sol.X_train)
    ax.contourf(
        xx, yy, np.clip(zz, -1, 1), levels=[-1, 0, 1], colors=[CLASS_NEG, CLASS_POS], alpha=0.08
    )
    _scatter_classes(ax, sol.X_train, sol.y_train)
    _margin_contours(ax, xx, yy, zz)
    _circle_support_vectors(ax, sol)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{feature_names[0]} (standardised)")
    ax.set_ylabel(f"{feature_names[1]} (standardised)")
    ax.set_title(
        f"Kernel SVM: f = 0 (solid), f = ±1 (dashed)  [{sol.kernel.label}, C = {sol.C:g}]"
    )
    _legend_below(ax)
    return _save(fig, path)


def gap_curve_figure(runs: Mapping[str, SVMSolution], path: Path) -> Path:
    """SMO traces: duality gap P - D and maximal KKT violation, both on log scales."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colours = (CLASS_POS, CLASS_NEG)
    for (label, sol), colour in zip(runs.items(), colours, strict=False):
        gap = np.maximum(sol.history["gap"], np.finfo(float).tiny)
        axes[0].semilogy(gap, color=colour, label=label, linewidth=0.8)
        axes[1].semilogy(sol.history["violation"], color=colour, label=label, linewidth=0.8)
    axes[0].set_title("Primal-dual gap  P(a, b*) - D(a)")
    axes[1].set_title("Max KKT violation  m(a) - M(a)")
    for ax in axes:
        ax.set_xlabel("SMO iteration")
        ax.legend()
    tol = next(iter(runs.values())).extras.get("smo_tol")
    if tol is not None:
        axes[1].axhline(tol, color=INK_MUTED, linestyle=":", linewidth=1.0)
    return _save(fig, path)


def pegasos_figure(traces: Mapping[str, tuple[np.ndarray, float]], path: Path) -> Path:
    """Relative suboptimality (P(f_t) - P*) / P* of Pegasos per epoch."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (label, (primal, optimum)), colour in zip(
        traces.items(), (CLASS_POS, CLASS_NEG), strict=False
    ):
        epochs = np.arange(1, len(primal) + 1)
        ax.loglog(epochs, (primal - optimum) / optimum, color=colour, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative suboptimality")
    ax.set_title("Approximate (Pegasos SGD) vs exact optimum")
    ax.legend()
    return _save(fig, path)


def linear_sensitivity_figure(sweep: pd.DataFrame, path: Path) -> Path:
    """Linear kernel: sparsity ratio, margin width and validation accuracy against C."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    series = (
        ("sparsity_ratio", "#SV / n", False),
        ("margin_width", "margin width 2/||w||", True),
        ("val_balanced_accuracy", "validation balanced accuracy", False),
    )
    for ax, (column, title, log_y) in zip(axes, series, strict=True):
        ax.plot(sweep["C"], sweep[column], color=CLASS_POS, marker="o", markersize=4)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("C")
        ax.set_title(title)
    return _save(fig, path)


def _heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, fmt: str) -> None:
    image = ax.imshow(table.to_numpy(), cmap=SEQUENTIAL_CMAP, origin="lower", aspect="auto")
    ax.set_xticks(range(table.shape[1]), [f"{v:g}" for v in table.columns], rotation=45)
    ax.set_yticks(range(table.shape[0]), [f"{v:g}" for v in table.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title(title)
    ax.grid(False)
    threshold = np.nanmean(table.to_numpy())
    for (row, col), value in np.ndenumerate(table.to_numpy()):
        ink = SURFACE if value > threshold else INK
        ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=6, color=ink)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def rbf_heatmaps_figure(sweep: pd.DataFrame, path: Path) -> Path:
    """RBF kernel: C x gamma heatmaps of sparsity ratio, log10 margin width and accuracy."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    panels = (
        ("sparsity_ratio", "#SV / n", ".2f"),
        ("log10_margin_width", "log10 margin width 2/||f||_H", ".1f"),
        ("val_balanced_accuracy", "validation balanced accuracy", ".3f"),
    )
    frame = sweep.assign(log10_margin_width=np.log10(sweep["margin_width"]))
    for ax, (column, title, fmt) in zip(axes, panels, strict=True):
        _heatmap(ax, frame.pivot(index="C", columns="gamma", values=column), title, fmt)
    return _save(fig, path)
