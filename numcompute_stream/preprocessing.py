"""
Incremental preprocessing utilities.

Classes
-------
StandardScaler
    Updates running column means and variances using a numerically stable
    batch-combine form of Welford's algorithm.

Imputer
    Maintains running column means and replaces NaN values during transform.

OneHotEncoder
    Expands categorical columns incrementally as new categories are observed.
"""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(X, name: str = "X") -> np.ndarray:
    """Convert input to a 2D float NumPy array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return arr


class StandardScaler:
    """
    Streaming standardisation transformer.

    The transformer keeps per-feature running mean and variance. NaN values are
    ignored when updating the statistics and remain NaN during transformation.
    Use Imputer before this scaler if missing values should be replaced.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, subtract the running mean.
    with_std : bool, default=True
        If True, divide by the running standard deviation.
    eps : float, default=1e-12
        Minimum scale value used to avoid division by zero.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.M2_ = None

    def _check_features(self, X: np.ndarray) -> None:
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(self.n_features_in_, dtype=float)
            self.mean_ = np.zeros(self.n_features_in_, dtype=float)
            self.M2_ = np.zeros(self.n_features_in_, dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Update running statistics from a chunk."""
        X = _as_2d_float_array(X)
        self._check_features(X)

        mask = ~np.isnan(X)
        batch_count = mask.sum(axis=0).astype(float)

        safe_X = np.where(mask, X, 0.0)
        batch_sum = safe_X.sum(axis=0)

        batch_mean = np.divide(
            batch_sum,
            batch_count,
            out=np.zeros_like(batch_sum, dtype=float),
            where=batch_count > 0,
        )

        centered = np.where(mask, X - batch_mean, 0.0)
        batch_M2 = (centered ** 2).sum(axis=0)

        old_count = self.count_.copy()
        old_mean = self.mean_.copy()
        old_M2 = self.M2_.copy()

        total_count = old_count + batch_count
        valid = batch_count > 0

        delta = batch_mean - old_mean
        new_mean = old_mean.copy()
        new_M2 = old_M2.copy()

        new_mean[valid] = old_mean[valid] + (
            delta[valid] * batch_count[valid] / total_count[valid]
        )
        new_M2[valid] = (
            old_M2[valid]
            + batch_M2[valid]
            + (delta[valid] ** 2)
            * old_count[valid]
            * batch_count[valid]
            / total_count[valid]
        )

        self.count_ = total_count
        self.mean_ = new_mean
        self.M2_ = new_M2
        return self

    @property
    def var_(self) -> np.ndarray:
        """Population variance for each feature."""
        if self.count_ is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return np.divide(
            self.M2_,
            self.count_,
            out=np.zeros_like(self.M2_, dtype=float),
            where=self.count_ > 0,
        )

    @property
    def scale_(self) -> np.ndarray:
        """Standard deviation for each feature with zero-variance protection."""
        scale = np.sqrt(self.var_)
        return np.where(scale < self.eps, 1.0, scale)

    def transform(self, X) -> np.ndarray:
        """Standardise a chunk using the current running statistics."""
        if self.mean_ is None:
            raise ValueError("StandardScaler must be fitted before transform().")
        X = _as_2d_float_array(X)
        self._check_features(X)

        out = X.astype(float, copy=True)
        if self.with_mean:
            out = out - self.mean_
        if self.with_std:
            out = out / self.scale_
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Update statistics and transform the same chunk."""
        return self.partial_fit(X).transform(X)


class Imputer:
    """
    Streaming mean imputer for numeric data.

    NaN values are ignored when updating the running column means. During
    transform, NaNs are replaced by the latest running mean for that column.

    Parameters
    ----------
    fill_value : float, default=0.0
        Value used for a column if no non-missing value has been observed yet.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value
        self.n_features_in_ = None
        self.count_ = None
        self.sum_ = None
        self.statistics_ = None

    def _check_features(self, X: np.ndarray) -> None:
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(self.n_features_in_, dtype=float)
            self.sum_ = np.zeros(self.n_features_in_, dtype=float)
            self.statistics_ = np.full(self.n_features_in_, self.fill_value, dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

    def partial_fit(self, X, y=None) -> "Imputer":
        """Update missing-value estimates from a chunk."""
        X = _as_2d_float_array(X)
        self._check_features(X)

        mask = ~np.isnan(X)
        self.count_ += mask.sum(axis=0)
        self.sum_ += np.where(mask, X, 0.0).sum(axis=0)

        self.statistics_ = np.divide(
            self.sum_,
            self.count_,
            out=np.full_like(self.sum_, self.fill_value, dtype=float),
            where=self.count_ > 0,
        )
        return self

    def transform(self, X) -> np.ndarray:
        """Replace NaN values with current running column means."""
        if self.statistics_ is None:
            raise ValueError("Imputer must be fitted before transform().")
        X = _as_2d_float_array(X)
        self._check_features(X)

        out = X.astype(float, copy=True)
        rows, cols = np.where(np.isnan(out))
        if rows.size:
            out[rows, cols] = self.statistics_[cols]
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Update imputation statistics and transform the same chunk."""
        return self.partial_fit(X).transform(X)


class OneHotEncoder:
    """
    Incremental one-hot encoder for categorical data.

    Parameters
    ----------
    handle_unknown : {"ignore", "error"}, default="ignore"
        Behaviour when transform sees a category not yet observed by partial_fit.
    dtype : type, default=float
        Output array dtype.
    """

    def __init__(self, handle_unknown: str = "ignore", dtype=float):
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.n_features_in_ = None
        self.categories_ = None

    def _as_2d_object_array(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"X must be a 1D or 2D array; got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one row.")
        return arr

    def _check_features(self, X: np.ndarray) -> None:
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(self.n_features_in_)]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Update the known category list using a chunk."""
        X = self._as_2d_object_array(X)
        self._check_features(X)

        for col in range(X.shape[1]):
            for value in np.unique(X[:, col]):
                if value not in self.categories_[col]:
                    self.categories_[col].append(value)
        return self

    def transform(self, X) -> np.ndarray:
        """One-hot encode a chunk using the current category mapping."""
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform().")
        X = self._as_2d_object_array(X)
        self._check_features(X)

        n_rows = X.shape[0]
        total_width = sum(len(cats) for cats in self.categories_)
        out = np.zeros((n_rows, total_width), dtype=self.dtype)

        offset = 0
        for col, cats in enumerate(self.categories_):
            mapping = {cat: i for i, cat in enumerate(cats)}
            for row, value in enumerate(X[:, col]):
                idx = mapping.get(value)
                if idx is None:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category {value!r} in column {col}.")
                    continue
                out[row, offset + idx] = 1
            offset += len(cats)
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Update known categories and transform the same chunk."""
        return self.partial_fit(X).transform(X)
# Preprocessing module
# MinMaxScaler implementation
# StandardScaler added
# LabelEncoder for categorical features
# OneHotEncoder implementation
