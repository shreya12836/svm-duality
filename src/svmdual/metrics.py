"""Classification metrics on decision values, for labels in {-1, +1}."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

CI_LEVEL = 0.95


@dataclass(frozen=True)
class ClassificationMetrics:
    balanced_accuracy: float
    macro_f1: float
    roc_auc: float
    pr_auc: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> ClassificationMetrics:
    """Threshold-free (ROC-AUC, PR-AUC) and thresholded-at-0 metrics."""
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    if set(np.unique(y_true).tolist()) != {-1.0, 1.0}:
        raise ValueError("y_true must contain both classes -1 and +1")
    y_pred = np.where(scores >= 0.0, 1.0, -1.0)
    return ClassificationMetrics(
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores, pos_label=1.0)),
    )


def bootstrap_balanced_accuracy_ci(
    y_true: np.ndarray, scores: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI (level CI_LEVEL) for balanced accuracy at threshold 0.

    Vectorised: balanced accuracy = (TPR + TNR) / 2 per resample. Resamples
    missing a class have an undefined rate and are dropped.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    truth, correct = y_true[idx], (np.where(scores >= 0.0, 1.0, -1.0) == y_true)[idx]
    n_pos, n_neg = (truth > 0).sum(axis=1), (truth < 0).sum(axis=1)
    valid = (n_pos > 0) & (n_neg > 0)
    if not np.any(valid):
        raise ValueError("No bootstrap resample contained both classes")
    tpr = (correct & (truth > 0)).sum(axis=1)[valid] / n_pos[valid]
    tnr = (correct & (truth < 0)).sum(axis=1)[valid] / n_neg[valid]
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = np.percentile((tpr + tnr) / 2.0, [tail, 100.0 - tail])
    return float(low), float(high)
