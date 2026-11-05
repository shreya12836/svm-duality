import numpy as np
import pytest
from sklearn.svm import SVC

from svmdual.kernels import Kernel
from svmdual.solvers import PegasosConfig, SMOConfig, cvx_dual, cvx_primal, l1_svm, pegasos, smo

C = 1.0
KERNELS = [Kernel("linear"), Kernel("rbf", 0.5)]
TIGHT = SMOConfig(tol=1e-6)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.name)
def test_cvx_primal_has_zero_duality_gap(toy, kernel) -> None:
    sol = cvx_primal(toy.X, toy.y, C, kernel)
    assert abs(sol.duality_gap) < 1e-5 * max(1.0, sol.primal_objective)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.name)
def test_cvx_primal_and_dual_agree(toy, kernel) -> None:
    primal = cvx_primal(toy.X, toy.y, C, kernel)
    dual = cvx_dual(toy.X, toy.y, C, kernel)
    assert primal.primal_objective == pytest.approx(dual.dual_objective, rel=1e-5)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.name)
def test_smo_matches_exact_dual(toy, kernel) -> None:
    exact = cvx_dual(toy.X, toy.y, C, kernel)
    sol = smo(toy.X, toy.y, C, kernel, TIGHT)
    assert sol.dual_objective == pytest.approx(exact.dual_objective, rel=1e-6)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.name)
def test_smo_matches_sklearn_decisions(toy, kernel) -> None:
    sol = smo(toy.X, toy.y, C, kernel, TIGHT)
    ref = SVC(C=C, kernel=kernel.name, gamma=kernel.gamma or "scale", tol=1e-8).fit(toy.X, toy.y)
    np.testing.assert_allclose(
        sol.decision_function(toy.X), ref.decision_function(toy.X), atol=1e-4
    )


def test_smo_converges(toy) -> None:
    assert smo(toy.X, toy.y, C, Kernel("rbf", 0.5), TIGHT).status == "optimal"


def test_smo_gap_history_ends_near_zero(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("rbf", 0.5), TIGHT)
    assert sol.history["gap"][-1] < 1e-4


def test_smo_gap_history_is_nonnegative(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("linear"), TIGHT)
    assert sol.history["gap"].min() > -1e-9


def test_smo_keeps_equality_constraint(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("linear"), TIGHT)
    assert abs(sol.alpha @ toy.y) < 1e-9


def test_smo_respects_box(toy) -> None:
    alpha = smo(toy.X, toy.y, C, Kernel("linear"), TIGHT).alpha
    assert alpha.min() >= 0.0 and alpha.max() <= C


def test_hard_margin_limit_recovers_known_hyperplane(separable) -> None:
    sol = cvx_primal(separable.X, separable.y, 1e4, Kernel("linear"))
    np.testing.assert_allclose(np.r_[sol.w, sol.b], [1.0, 0.0, 0.0], atol=1e-5)


def test_hard_margin_support_vectors_are_closest_points(separable) -> None:
    sol = smo(separable.X, separable.y, 1e4, Kernel("linear"), TIGHT)
    np.testing.assert_array_equal(sol.alpha > 1e-8, [True, False, True, False])


def test_pegasos_is_close_but_not_better_than_optimum(toy) -> None:
    exact = cvx_primal(toy.X, toy.y, C, Kernel("rbf", 0.5))
    approx = pegasos(toy.X, toy.y, C, Kernel("rbf", 0.5), PegasosConfig(epochs=200, seed=0))
    suboptimality = approx.primal_objective - exact.primal_objective
    assert -1e-6 < suboptimality < 0.1 * exact.primal_objective


def test_pegasos_is_reproducible(toy) -> None:
    def run():
        return pegasos(toy.X, toy.y, C, Kernel("linear"), PegasosConfig(epochs=5, seed=3))

    np.testing.assert_array_equal(run().alpha, run().alpha)


def test_l1_svm_strong_duality(toy) -> None:
    sol = l1_svm(toy.X, toy.y, C, Kernel("linear"))
    assert sol.duality_gap == pytest.approx(0.0, abs=1e-6)


def test_l1_svm_dual_feasible(toy) -> None:
    assert l1_svm(toy.X, toy.y, C, Kernel("linear")).extras["dual_norm_violation"] < 1e-6


def test_l1_svm_rejects_rbf(toy) -> None:
    with pytest.raises(ValueError):
        l1_svm(toy.X, toy.y, C, Kernel("rbf", 1.0))


def test_smo_reported_gap_matches_trace(toy) -> None:
    sol = smo(toy.X, toy.y, C, Kernel("linear"), SMOConfig(tol=1e-3))
    assert sol.duality_gap == pytest.approx(sol.history["gap"][-1], abs=1e-9)


def test_smo_reports_max_iter_when_budget_too_small(toy) -> None:
    assert smo(toy.X, toy.y, C, Kernel("linear"), SMOConfig(max_iter=2)).status == "max_iter"


def test_cvx_dual_offset_close_to_kkt_offset(toy) -> None:
    sol = cvx_dual(toy.X, toy.y, C, Kernel("rbf", 0.5))
    assert sol.b == pytest.approx(sol.extras["b_kkt"], abs=1e-3)


@pytest.mark.parametrize("solver", [smo, pegasos, cvx_primal, cvx_dual])
def test_solvers_reject_single_class(toy, solver) -> None:
    with pytest.raises(ValueError, match="labels"):
        solver(toy.X, np.ones_like(toy.y), C, Kernel("linear"))


def test_solvers_reject_nonpositive_c(toy) -> None:
    with pytest.raises(ValueError, match="C must be positive"):
        smo(toy.X, toy.y, 0.0, Kernel("linear"))


def test_configs_reject_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        PegasosConfig(epochs=0)
