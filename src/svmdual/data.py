"""Loading, verification, deduplication, splitting and scaling of the banknote data.

The raw UCI file has no header and five comma-separated columns: four wavelet
features and a 0/1 class label. Labels are mapped to y in {-1, +1} with
class 1 -> +1 and class 0 -> -1, which is the sign convention used by every
solver in this package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES: tuple[str, ...] = ("variance", "skewness", "curtosis", "entropy")
LABEL_COLUMN = "class"
RAW_COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, LABEL_COLUMN)
VALID_LABELS = frozenset({0, 1})
HASH_CHUNK_BYTES = 1 << 16


@dataclass(frozen=True)
class Split:
    """One data partition: features X (n, d) and signed labels y (n,)."""

    X: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class Dataset:
    """Standardised train / validation / test partitions plus provenance."""

    train: Split
    val: Split
    test: Split
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    n_raw: int
    n_duplicates_removed: int


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str | None) -> None:
    """Raise ValueError if the file's SHA-256 differs from the expected one."""
    if expected_sha256 is None:
        return
    actual = sha256_of(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}")


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw headerless CSV and validate its shape, types and labels."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns, found {frame.shape[1]}")
    frame.columns = list(RAW_COLUMNS)
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values")
    features = frame[list(FEATURE_NAMES)]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All feature columns must be numeric")
    labels = set(frame[LABEL_COLUMN].unique().tolist())
    if not labels <= VALID_LABELS:
        raise ValueError(f"Unexpected labels {sorted(labels - VALID_LABELS)}")
    return frame


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows (keeping the first) and report how many were removed."""
    unique = frame.drop_duplicates(keep="first").reset_index(drop=True)
    return unique, len(frame) - len(unique)


def to_signed_labels(labels: np.ndarray) -> np.ndarray:
    """Map 0/1 labels to -1/+1 floats."""
    return np.where(np.asarray(labels) == 1, 1.0, -1.0)


def stratified_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> tuple[Split, Split, Split]:
    """Split into train / validation / test with class proportions preserved."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Fractions must be in (0, 1) and sum to less than 1")
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=seed
    )
    val_of_rest = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_of_rest, stratify=y_rest, random_state=seed
    )
    return Split(X_train, y_train), Split(X_val, y_val), Split(X_test, y_test)


def standardize(
    train: Split, val: Split, test: Split
) -> tuple[Split, Split, Split, np.ndarray, np.ndarray]:
    """Scale all partitions with mean and std estimated on the training split only."""
    mean = train.X.mean(axis=0)
    scale = train.X.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("A feature is constant on the training split")

    def transform(split: Split) -> Split:
        return Split((split.X - mean) / scale, split.y)

    return transform(train), transform(val), transform(test), mean, scale


def load_dataset(
    path: Path,
    features: tuple[str, ...] = FEATURE_NAMES,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    expected_sha256: str | None = None,
) -> Dataset:
    """Full pipeline: verify, load, deduplicate, split (stratified) and standardise."""
    unknown = set(features) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown features {sorted(unknown)}")
    verify_checksum(path, expected_sha256)
    raw = load_raw(path)
    unique, n_removed = deduplicate(raw)
    X = unique[list(features)].to_numpy(dtype=float)
    y = to_signed_labels(unique[LABEL_COLUMN].to_numpy())
    train, val, test = stratified_split(X, y, val_fraction, test_fraction, seed)
    train, val, test, mean, scale = standardize(train, val, test)
    return Dataset(train, val, test, tuple(features), mean, scale, len(raw), n_removed)
