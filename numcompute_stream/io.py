"""
Small I/O helpers using only NumPy and the Python standard library.
"""

from __future__ import annotations

import csv
import numpy as np


def read_csv(path, target_column: int = -1, delimiter: str = ",", skip_header: bool = True, dtype=float):
    """
    Load a numeric CSV file and split it into X and y.

    Parameters
    ----------
    path : str
        CSV file path.
    target_column : int, default=-1
        Index of the target column.
    delimiter : str, default=","
        CSV delimiter.
    skip_header : bool, default=True
        Whether to skip the first row.
    dtype : type, default=float
        Data type for NumPy loading.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    """
    skip = 1 if skip_header else 0
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip, dtype=dtype)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and one target column.")

    target_column = int(target_column)
    y = data[:, target_column]
    X = np.delete(data, target_column, axis=1)

    if np.all(np.isfinite(y)) and np.all(y == y.astype(int)):
        y = y.astype(int)
    return X, y


def chunk_iter(X, y, chunk_size: int, shuffle: bool = False, random_state: int | None = None):
    """
    Yield (X_chunk, y_chunk) pairs.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    chunk_size : int
        Number of rows per chunk.
    shuffle : bool, default=False
        Whether to shuffle before chunking.
    random_state : int or None
        Random seed used when shuffle=True.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    indices = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    for start in range(0, X.shape[0], chunk_size):
        idx = indices[start : start + chunk_size]
        yield X[idx], y[idx]


def write_csv(path, X, y, header=None, delimiter: str = ","):
    """
    Write X and y to a CSV file. Useful for creating demo data.

    Parameters
    ----------
    header : list[str] or None
        Optional column names. If provided, length must equal X.shape[1] + 1.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows.")

    if header is not None and len(header) != X.shape[1] + 1:
        raise ValueError("header length must equal number of features plus one target.")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        for row, target in zip(X, y):
            writer.writerow(list(row) + [target])
# IO: CSV and Parquet readers
# CSVStream reader
# ParquetStream reader
# chunked_read() generator
# JSONStream reader
# KafkaStream stub
