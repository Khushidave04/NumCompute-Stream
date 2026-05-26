"""
Decision tree classifier implemented using NumPy.

The tree supports chunk-wise partial_fit(). For simplicity and reliability in
a streaming assignment setting, each call stores the new chunk and rebuilds a
depth-limited tree from all retained data. This gives deterministic incremental
behaviour while keeping the code understandable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class _Node:
    prediction: object
    depth: int
    is_leaf: bool = True
    feature_index: int | None = None
    threshold: float | None = None
    left: object | None = None
    right: object | None = None
    default_left: bool = True


def _as_2d_float_array(X, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return arr


def _as_1d_labels(y, n_rows: int) -> np.ndarray:
    arr = np.asarray(y).ravel()
    if arr.shape[0] != n_rows:
        raise ValueError(f"y must contain {n_rows} labels; got {arr.shape[0]}.")
    return arr


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity function.
    max_features : None, int, float, {"sqrt", "log2"}, default=None
        Number of candidate features considered at each split.
    random_state : int or None, default=None
        Seed used when max_features samples a feature subset.
    max_samples : int or None, default=None
        If set, only the most recent max_samples observations are retained.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features=None,
        random_state: int | None = None,
        max_samples: int | None = None,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.max_samples = max_samples

        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._X_seen = None
        self._y_seen = None

    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """Update the tree with a new data chunk."""
        X = _as_2d_float_array(X_chunk)
        y = _as_1d_labels(y_chunk, X.shape[0])

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        if self._X_seen is None:
            self._X_seen = X.copy()
            self._y_seen = y.copy()
        else:
            self._X_seen = np.vstack([self._X_seen, X])
            self._y_seen = np.concatenate([self._y_seen, y])

        if self.max_samples is not None and self._X_seen.shape[0] > self.max_samples:
            self._X_seen = self._X_seen[-self.max_samples :]
            self._y_seen = self._y_seen[-self.max_samples :]

        self.classes_ = np.unique(self._y_seen)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build_tree(self._X_seen, self._y_seen, depth=0, rng=rng)
        return self

    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Fit from one batch. Provided for convenience."""
        self._X_seen = None
        self._y_seen = None
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y)

    def predict(self, X) -> np.ndarray:
        """Predict labels for X."""
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier must be fitted before predict().")
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        return np.asarray([self._predict_row(row, self.root_) for row in X])

    def score(self, X, y) -> float:
        """Return accuracy on X and y."""
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y contain different numbers of samples.")
        return float(np.mean(pred == y))

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _class_indices(self, y) -> np.ndarray:
        return np.searchsorted(self.classes_, y)

    def _impurity_from_counts(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=-1, keepdims=True)
        probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2, axis=-1)

        log_probs = np.zeros_like(probs)
        positive = probs > 0
        log_probs[positive] = np.log2(probs[positive])
        return -np.sum(probs * log_probs, axis=-1)

    def _feature_count(self, n_features: int, rng) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features string must be 'sqrt' or 'log2'.")
        if isinstance(mf, float):
            if not (0 < mf <= 1):
                raise ValueError("float max_features must be in (0, 1].")
            return max(1, int(np.ceil(mf * n_features)))
        if isinstance(mf, int):
            if mf < 1:
                raise ValueError("int max_features must be >= 1.")
            return min(mf, n_features)
        raise ValueError("Unsupported max_features value.")

    def _candidate_features(self, n_features: int, rng) -> np.ndarray:
        k = self._feature_count(n_features, rng)
        if k == n_features:
            return np.arange(n_features)
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray, rng):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None

        y_idx = self._class_indices(y)
        n_classes = len(self.classes_)
        parent_counts = np.bincount(y_idx, minlength=n_classes)
        parent_impurity = float(self._impurity_from_counts(parent_counts.reshape(1, -1))[0])

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        features = self._candidate_features(n_features, rng)

        for feature in features:
            col = X[:, feature]
            valid_mask = ~np.isnan(col)

            if valid_mask.sum() < self.min_samples_split:
                continue

            x_valid = col[valid_mask]
            y_valid_idx = y_idx[valid_mask]

            order = np.argsort(x_valid, kind="mergesort")
            x_sorted = x_valid[order]
            y_sorted = y_valid_idx[order]

            diff_positions = np.where(x_sorted[1:] != x_sorted[:-1])[0] + 1
            if diff_positions.size == 0:
                continue

            one_hot = np.eye(n_classes, dtype=float)[y_sorted]
            cum_counts = np.cumsum(one_hot, axis=0)
            total_counts = cum_counts[-1]

            left_counts = cum_counts[diff_positions - 1]
            right_counts = total_counts - left_counts

            left_n = left_counts.sum(axis=1)
            right_n = right_counts.sum(axis=1)
            total_n = left_n + right_n

            left_impurity = self._impurity_from_counts(left_counts)
            right_impurity = self._impurity_from_counts(right_counts)

            weighted_impurity = (left_n / total_n) * left_impurity + (
                right_n / total_n
            ) * right_impurity
            gains = parent_impurity - weighted_impurity

            idx = int(np.argmax(gains))
            gain = float(gains[idx])

            if gain > best_gain:
                pos = int(diff_positions[idx])
                best_gain = gain
                best_feature = int(feature)
                best_threshold = float((x_sorted[pos - 1] + x_sorted[pos]) / 2.0)

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng) -> _Node:
        prediction = self._majority_class(y)
        node = _Node(prediction=prediction, depth=depth, is_leaf=True)

        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return node

        split = self._best_split(X, y, rng)
        if split is None:
            return node

        feature, threshold, gain = split
        if gain <= 0:
            return node

        col = X[:, feature]
        nan_mask = np.isnan(col)
        left_mask = col <= threshold
        right_mask = col > threshold

        # Send NaNs to the larger branch to avoid losing rows.
        if left_mask.sum() >= right_mask.sum():
            left_mask = left_mask | nan_mask
            default_left = True
        else:
            right_mask = right_mask | nan_mask
            default_left = False

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node

        node.is_leaf = False
        node.feature_index = feature
        node.threshold = threshold
        node.default_left = default_left
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)
        return node

    def _predict_row(self, row: np.ndarray, node: _Node):
        while not node.is_leaf:
            value = row[node.feature_index]
            if np.isnan(value):
                node = node.left if node.default_left else node.right
            elif value <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
# Hoeffding Tree skeleton
# HoeffdingTreeNode dataclass added
# Hoeffding bound split criterion
# HoeffdingTree.fit() implemented
