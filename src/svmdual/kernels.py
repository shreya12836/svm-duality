"""Linear and Gaussian (RBF) kernels and an exact empirical feature map.

The empirical feature map factorises a Gram matrix K = Phi Phi^T. By the
representer theorem the RKHS-norm problem over f in H_k restricted to the
training points is equivalent to a finite-dimensional problem in w = Phi^T c,
which is what lets CVXPY solve the kernel *primal* directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

KernelName = Literal["linear", "rbf"]
EIGEN_REL_TOL = 1e-10


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """k(x, x') = <x, x'>."""
    return A @ B.T


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """k(x, x') = exp(-gamma * ||x - x'||^2)."""
    sq_a = np.sum(A * A, axis=1)[:, None]
    sq_b = np.sum(B * B, axis=1)[None, :]
    sq_dist = np.maximum(sq_a + sq_b - 2.0 * (A @ B.T), 0.0)
    return np.exp(-gamma * sq_dist)


@dataclass(frozen=True)
class Kernel:
    """A positive-definite kernel identified by name and (for RBF) its gamma."""

    name: KernelName
    gamma: float | None = None

    def __post_init__(self) -> None:
        if self.name not in ("linear", "rbf"):
            raise ValueError(f"Unknown kernel {self.name!r}")
        if self.name == "rbf" and (self.gamma is None or self.gamma <= 0):
            raise ValueError("RBF kernel needs gamma > 0")

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.name == "linear":
            return linear_kernel(A, B)
        return rbf_kernel(A, B, float(self.gamma))

    @property
    def label(self) -> str:
        return "linear" if self.name == "linear" else f"rbf(gamma={self.gamma:g})"


def empirical_feature_map(K: np.ndarray, rel_tol: float = EIGEN_REL_TOL) -> np.ndarray:
    """Return Phi (n, r) with Phi @ Phi.T == K, dropping eigenvalues below rel_tol * max."""
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("Gram matrix must be square")
    eigvals, eigvecs = np.linalg.eigh((K + K.T) / 2.0)
    keep = eigvals > rel_tol * max(eigvals.max(), 0.0)
    return eigvecs[:, keep] * np.sqrt(eigvals[keep])


def primal_features(X: np.ndarray, kernel: Kernel) -> np.ndarray:
    """Features in which the primal is a plain linear SVM: X itself or the empirical map."""
    if kernel.name == "linear":
        return X
    return empirical_feature_map(kernel(X, X))
