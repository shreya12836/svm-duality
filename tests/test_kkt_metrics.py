from dataclasses import replace

import numpy as np
import pytest

from svmdual.kernels import Kernel
from svmdual.kkt import kkt_report, margin_width, reduced_hull_points, rkhs_norm_sq, sparsity
from svmdual.metrics import classification_metrics
from svmdual.solvers import SMOConfig, cvx_primal, l1_svm, smo

C = 1.0
KKT_TOL = 1e-4


@pytest.fixture(scope="module")
def optimum(toy):
    return smo(toy.X, toy.y, C, Kernel("rbf", 0.5), SMOConfig(tol=1e-8))


@pytest.mark.parametrize(
    "field",
    ["dual_box_violation", "equality_residual", "comp_slack_margin", "comp_slack_slack"],
)
def test_kkt_residuals_vanish_at_optimum(optimum, field) -> None:
    assert getattr(kkt_report(optimum), field) < KKT_TOL


def test_kkt_detects_perturbed_solution(optimum) -> None:
    perturbed = replace(optimum, b=optimum.b + 0.5)
    assert kkt_report(perturbed).comp_slack_margin > 0.01


def test_cvx_primal_stationarity_reported(toy) -> None:
    report = kkt_report(cvx_primal(toy.X, toy.y, C, Kernel("linear")))
    assert report.stationarity_w < KKT_TOL


def test_l1_report_has_no_l2_pair_violation(toy) -> None:
    report = kkt_report(l1_svm(toy.X, toy.y, C, Kernel("linear")))
    assert np.isnan(report.max_violating_pair)


def test_sparsity_counts() -> None:
    s = sparsity(np.array([0.0, 0.3, 1.0, 1.0]), C=1.0)
    assert (s.n_sv, s.n_free, s.n_bounded, s.ratio) == (3, 1, 2, 0.75)


def test_rkhs_norm_matches_w_norm_for_linear(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("linear"), SMOConfig(tol=1e-8))
    w = toy.X.T @ sol.dual_coef
    assert rkhs_norm_sq(sol) == pytest.approx(w @ w)


def test_margin_width_of_known_hyperplane(separable) -> None:
    sol = cvx_primal(separable.X, separable.y, 1e4, Kernel("linear"))
    assert margin_width(sol) == pytest.approx(2.0, rel=1e-5)


def test_reduced_hull_direction_equals_w(toy) -> None:
    sol = cvx_primal(toy.X, toy.y, C, Kernel("linear"))
    c, d, _ = reduced_hull_points(sol)
    cosine = (c - d) @ sol.w / (np.linalg.norm(c - d) * np.linalg.norm(sol.w))
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_reduced_hull_weight_bound(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("linear"), SMOConfig(tol=1e-8))
    _, _, D = reduced_hull_points(sol)
    s = sol.alpha[toy.y > 0].sum()
    assert (sol.alpha / s).max() <= D + 1e-12


def test_metrics_perfect_scores() -> None:
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    m = classification_metrics(y, np.array([-2.0, -1.0, 1.0, 2.0]))
    assert (m.balanced_accuracy, m.macro_f1, m.roc_auc, m.pr_auc) == (1.0, 1.0, 1.0, 1.0)


def test_metrics_reject_single_class() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.ones(3), np.zeros(3))


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array([-1.0, 1.0]), np.zeros(3))


def test_margin_width_is_infinite_for_zero_function(optimum) -> None:
    zero = replace(optimum, alpha=np.zeros_like(optimum.alpha))
    assert margin_width(zero) == float("inf")


def test_reduced_hull_points_reject_all_zero_alpha(optimum) -> None:
    with pytest.raises(ValueError):
        reduced_hull_points(replace(optimum, alpha=np.zeros_like(optimum.alpha)))


def test_bootstrap_ci_contains_point_estimate() -> None:
    from svmdual.metrics import bootstrap_balanced_accuracy_ci

    rng = np.random.default_rng(0)
    y = np.repeat([-1.0, 1.0], 100)
    scores = y + rng.normal(scale=1.0, size=200)
    point = classification_metrics(y, scores).balanced_accuracy
    low, high = bootstrap_balanced_accuracy_ci(y, scores, n_resamples=500, seed=0)
    assert low <= point <= high


def test_bootstrap_ci_is_degenerate_for_perfect_scores() -> None:
    from svmdual.metrics import bootstrap_balanced_accuracy_ci

    y = np.repeat([-1.0, 1.0], 20)
    assert bootstrap_balanced_accuracy_ci(y, y, n_resamples=100, seed=0) == (1.0, 1.0)
