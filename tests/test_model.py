import numpy as np
import pytest

from svmdual.model import best_offset, hinge_losses, offset_from_kkt


def brute_force_offset(g: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(-10, 10, 200_001)
    losses = np.maximum(0.0, 1.0 - y[None, :] * (g[None, :] + grid[:, None])).sum(axis=1)
    return float(losses.min())


def test_hinge_losses_values() -> None:
    np.testing.assert_allclose(hinge_losses(np.array([2.0, 1.0, 0.5, -1.0])), [0, 0, 0.5, 2.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_offset_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    g, y = rng.normal(size=25), rng.choice([-1.0, 1.0], size=25)
    _, hinge = best_offset(g, y)
    assert hinge == pytest.approx(brute_force_offset(g, y), abs=1e-3)


def test_best_offset_value_is_attained() -> None:
    rng = np.random.default_rng(3)
    g, y = rng.normal(size=25), rng.choice([-1.0, 1.0], size=25)
    b, hinge = best_offset(g, y)
    assert hinge == pytest.approx(hinge_losses(y * (g + b)).sum())


def test_offset_from_kkt_uses_free_vectors() -> None:
    alpha, y = np.array([0.5, 0.5, 0.0]), np.array([1.0, -1.0, 1.0])
    g = np.array([0.7, -1.3, 5.0])  # free SVs give b = 1 - 0.7 = 0.3 and -1 + 1.3 = 0.3
    assert offset_from_kkt(alpha, g, y, C=1.0, tol=1e-9) == pytest.approx(0.3)


def test_offset_from_kkt_without_free_vectors_is_within_bounds() -> None:
    alpha, y = np.array([0.0, 0.0]), np.array([1.0, -1.0])
    g = np.array([2.0, -2.0])  # b in [1 - 2, -1 + 2] = [-1, 1]
    assert offset_from_kkt(alpha, g, y, C=1.0, tol=1e-9) == pytest.approx(0.0)
