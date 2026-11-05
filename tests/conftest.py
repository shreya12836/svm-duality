from dataclasses import dataclass

import numpy as np
import pytest

N_PER_CLASS = 30


@dataclass(frozen=True)
class Toy:
    X: np.ndarray
    y: np.ndarray


@pytest.fixture(scope="session")
def toy() -> Toy:
    """Two overlapping Gaussian blobs in R^2 (not linearly separable)."""
    rng = np.random.default_rng(7)
    pos = rng.normal(loc=(1.0, 1.0), scale=1.0, size=(N_PER_CLASS, 2))
    neg = rng.normal(loc=(-1.0, -1.0), scale=1.0, size=(N_PER_CLASS, 2))
    return Toy(np.vstack([pos, neg]), np.repeat([1.0, -1.0], N_PER_CLASS))


@pytest.fixture(scope="session")
def separable() -> Toy:
    """Four points with a known maximum-margin line x1 = 0, w = (1, 0), b = 0."""
    X = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 0.0], [-2.0, -1.0]])
    return Toy(X, np.array([1.0, 1.0, -1.0, -1.0]))
