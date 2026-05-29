"""
Streaming statistics for chunk-based data analysis.

The main API is StreamingStats.update_stats(X_chunk). It maintains stable
running mean/variance and provides optional rolling-window quantiles and
histograms.
"""

from __future__ import annotations

from collections import deque
import numpy as np


def _as_2d_float_array(X, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return arr


class StreamingStats:
    """
    Incremental mean, variance, quantile and histogram helper.

    Parameters
    ----------
    window_size : int or None, default=None
        Number of latest rows retained for rolling quantiles/histograms.
        If None, a bounded internal buffer of max_buffer rows is kept only for
        quantile/histogram convenience; mean/variance remain fully cumulative.
    max_buffer : int, default=10000
        Maximum rows stored when window_size is None.
    """

    def __init__(self, window_size: int | None = None, max_buffer: int = 10000):
        self.window_size = window_size
        self.max_buffer = max_buffer
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        maxlen = window_size if window_size is not None else max_buffer
        self._buffer = deque(maxlen=maxlen)

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

    def update_stats(self, X_chunk) -> "StreamingStats":
        """Update cumulative statistics using a new chunk."""
        X = _as_2d_float_array(X_chunk)
        self._check_features(X)

        mask = ~np.isnan(X)
        batch_count = mask.sum(axis=0).astype(float)
        batch_sum = np.where(mask, X, 0.0).sum(axis=0)
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

        self.mean_[valid] = old_mean[valid] + (
            delta[valid] * batch_count[valid] / total_count[valid]
        )
        self.M2_[valid] = (
            old_M2[valid]
            + batch_M2[valid]
            + (delta[valid] ** 2)
            * old_count[valid]
            * batch_count[valid]
            / total_count[valid]
        )
        self.count_ = total_count

        for row in X:
            self._buffer.append(row.astype(float, copy=True))

        return self

    @property
    def variance_(self) -> np.ndarray:
        """Cumulative population variance."""
        if self.count_ is None:
            raise ValueError("No statistics available. Call update_stats() first.")
        return np.divide(
            self.M2_,
            self.count_,
            out=np.zeros_like(self.M2_, dtype=float),
            where=self.count_ > 0,
        )

    def result(self) -> dict:
        """Return current cumulative statistics."""
        if self.mean_ is None:
            return {"count": None, "mean": None, "variance": None}
        return {
            "count": self.count_.copy(),
            "mean": self.mean_.copy(),
            "variance": self.variance_.copy(),
        }

    def _buffer_array(self) -> np.ndarray:
        if not self._buffer:
            raise ValueError("No buffered data available.")
        return np.vstack(list(self._buffer))

    def quantiles(self, q=(0.25, 0.5, 0.75), feature: int | None = None) -> np.ndarray:
        """
        Return quantiles from the retained buffer.

        This is exact for the retained rolling window and approximate for the
        entire stream when max_buffer truncates old observations.
        """
        data = self._buffer_array()
        if feature is not None:
            data = data[:, feature]
        return np.nanquantile(data, q, axis=0)

    def histogram(self, bins=10, feature: int = 0, range=None):
        """Return histogram counts and bin edges for one feature from the buffer."""
        data = self._buffer_array()[:, feature]
        data = data[~np.isnan(data)]
        return np.histogram(data, bins=bins, range=range)

    def reset(self) -> "StreamingStats":
        """Clear all accumulated statistics."""
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        maxlen = self.window_size if self.window_size is not None else self.max_buffer
        self._buffer = deque(maxlen=maxlen)
        return self
# Descriptive statistics helpers
# RunningMean class
# Welford running variance
# HistogramSketch for numeric features
# CountMinSketch
# StreamingKMeans (Lloyd online)
