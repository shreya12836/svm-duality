import numpy as np
import pytest

from svmdual.kernels import Kernel, empirical_feature_map, primal_features


def test_linear_kernel_is_inner_product(toy) -> None:
    np.testing.assert_allclose(Kernel("linear")(toy.X, toy.X), toy.X @ toy.X.T)


def test_rbf_diagonal_is_one(toy) -> None:
    np.testing.assert_allclose(np.diag(Kernel("rbf", 0.5)(toy.X, toy.X)), 1.0)


def test_rbf_matches_definition() -> None:
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 2.0]])
    assert Kernel("rbf", 0.3)(a, b)[0, 0] == pytest.approx(np.exp(-0.3 * 5.0))


def test_rbf_gram_is_positive_semidefinite(toy) -> None:
    assert np.linalg.eigvalsh(Kernel("rbf", 1.0)(toy.X, toy.X)).min() > -1e-10


def test_rbf_requires_positive_gamma() -> None:
    with pytest.raises(ValueError):
        Kernel("rbf", 0.0)


def test_unknown_kernel_rejected() -> None:
    with pytest.raises(ValueError):
        Kernel("poly")  # type: ignore[arg-type]


def test_feature_map_reconstructs_gram(toy) -> None:
    K = Kernel("rbf", 0.5)(toy.X, toy.X)
    Phi = empirical_feature_map(K)
    np.testing.assert_allclose(Phi @ Phi.T, K, atol=1e-8)


def test_feature_map_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        empirical_feature_map(np.zeros((2, 3)))


def test_primal_features_linear_is_identity(toy) -> None:
    assert primal_features(toy.X, Kernel("linear")) is toy.X
