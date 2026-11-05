from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from svmdual.data import (
    deduplicate,
    load_dataset,
    load_raw,
    sha256_of,
    standardize,
    stratified_split,
    to_signed_labels,
    verify_checksum,
)

N_PER_CLASS = 50


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(2 * N_PER_CLASS, 4))
    labels = np.repeat([0, 1], N_PER_CLASS)
    frame = pd.DataFrame(np.column_stack([features, labels]))
    frame[4] = frame[4].astype(int)
    frame = pd.concat([frame, frame.iloc[:3]], ignore_index=True)  # 3 duplicates
    path = tmp_path / "raw.csv"
    frame.to_csv(path, header=False, index=False)
    return path


def test_load_raw_names_columns(raw_csv: Path) -> None:
    assert list(load_raw(raw_csv).columns) == [
        "variance",
        "skewness",
        "curtosis",
        "entropy",
        "class",
    ]


def test_load_raw_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_bad_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("1,2,3,4,7\n")
    with pytest.raises(ValueError, match="Unexpected labels"):
        load_raw(path)


def test_load_raw_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="columns"):
        load_raw(path)


def test_deduplicate_counts_removed_rows(raw_csv: Path) -> None:
    _, n_removed = deduplicate(load_raw(raw_csv))
    assert n_removed == 3


def test_to_signed_labels_maps_to_plus_minus_one() -> None:
    np.testing.assert_array_equal(to_signed_labels(np.array([0, 1, 1])), [-1.0, 1.0, 1.0])


def test_checksum_mismatch_raises(raw_csv: Path) -> None:
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksum(raw_csv, "0" * 64)


def test_checksum_match_passes(raw_csv: Path) -> None:
    verify_checksum(raw_csv, sha256_of(raw_csv))


def test_split_sizes_are_60_20_20() -> None:
    y = np.repeat([-1.0, 1.0], N_PER_CLASS)
    X = np.arange(2 * N_PER_CLASS, dtype=float)[:, None]
    train, val, test = stratified_split(X, y, 0.2, 0.2, seed=0)
    assert (train.n, val.n, test.n) == (60, 20, 20)


def test_split_partitions_are_disjoint_and_complete() -> None:
    y = np.repeat([-1.0, 1.0], N_PER_CLASS)
    X = np.arange(2 * N_PER_CLASS, dtype=float)[:, None]
    parts = stratified_split(X, y, 0.2, 0.2, seed=0)
    ids = np.concatenate([p.X.ravel() for p in parts])
    assert sorted(ids.tolist()) == X.ravel().tolist()


def test_split_is_stratified() -> None:
    y = np.repeat([-1.0, 1.0], N_PER_CLASS)
    X = np.zeros((2 * N_PER_CLASS, 1))
    _, val, _ = stratified_split(X, y, 0.2, 0.2, seed=0)
    assert np.mean(val.y == 1.0) == pytest.approx(0.5)


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        stratified_split(np.zeros((10, 1)), np.ones(10), 0.6, 0.5, seed=0)


def test_standardize_uses_training_statistics_only() -> None:
    from svmdual.data import Split

    train = Split(np.array([[0.0], [2.0]]), np.array([-1.0, 1.0]))
    other = Split(np.array([[100.0]]), np.array([1.0]))
    _, val, _, mean, scale = standardize(train, other, other)
    assert val.X[0, 0] == pytest.approx((100.0 - 1.0) / 1.0)


def test_load_dataset_end_to_end(raw_csv: Path) -> None:
    data = load_dataset(raw_csv, seed=1)
    assert data.train.n + data.val.n + data.test.n == 2 * N_PER_CLASS


def test_load_dataset_train_is_standardised(raw_csv: Path) -> None:
    data = load_dataset(raw_csv, seed=1)
    np.testing.assert_allclose(data.train.X.mean(axis=0), 0.0, atol=1e-12)


def test_load_dataset_rejects_unknown_feature(raw_csv: Path) -> None:
    with pytest.raises(ValueError, match="Unknown features"):
        load_dataset(raw_csv, features=("variance", "colour"))
