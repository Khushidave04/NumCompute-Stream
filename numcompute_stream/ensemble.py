"""
Tree-based ensemble classifier.

This module implements a bagging / random-forest style ensemble built from
DecisionTreeClassifier objects. It supports partial_fit() for chunk-wise
streaming adaptation.
"""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier


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


class EnsembleClassifier:
    """
    Bagging / random-forest style ensemble of decision trees.

    Parameters
    ----------
    n_estimators : int, default=5
        Number of decision trees.
    method : {"bagging", "random_forest"}, default="bagging"
        Ensemble mode. random_forest uses max_features="sqrt" unless specified.
    max_depth : int, default=5
        Maximum depth for each tree.
    min_samples_split : int, default=2
        Minimum samples required to split a tree node.
    criterion : {"gini", "entropy"}, default="gini"
        Tree impurity criterion.
    max_features : None, int, float, {"sqrt", "log2"}, default=None
        Number of features considered by each tree at each split.
    bootstrap : bool, default=True
        If True, each tree receives a bootstrap sample of the current chunk.
    sample_fraction : float, default=1.0
        Fraction of the current chunk sampled for each tree.
    random_state : int or None, default=None
        Random seed.
    max_samples : int or None, default=None
        Per-tree retained sample limit.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        method: str = "bagging",
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features=None,
        bootstrap: bool = True,
        sample_fraction: float = 1.0,
        random_state: int | None = None,
        max_samples: int | None = None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if method not in {"bagging", "random_forest"}:
            raise ValueError("method must be 'bagging' or 'random_forest'.")
        if sample_fraction <= 0:
            raise ValueError("sample_fraction must be positive.")

        self.n_estimators = int(n_estimators)
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.max_samples = max_samples

        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def _make_estimators(self):
        if self.max_features is None and self.method == "random_forest":
            tree_max_features = "sqrt"
        else:
            tree_max_features = self.max_features

        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=tree_max_features,
                random_state=None if self.random_state is None else self.random_state + i,
                max_samples=self.max_samples,
            )
            for i in range(self.n_estimators)
        ]

    def partial_fit(self, X_chunk, y_chunk) -> "EnsembleClassifier":
        """Update every tree using a sampled version of the new chunk."""
        X = _as_2d_float_array(X_chunk)
        y = _as_1d_labels(y_chunk, X.shape[0])

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        if not self.estimators_:
            self._make_estimators()

        self.classes_ = np.unique(y) if self.classes_ is None else np.unique(
            np.concatenate([self.classes_, np.unique(y)])
        )

        n = X.shape[0]
        sample_size = max(1, int(np.ceil(self.sample_fraction * n)))

        for tree in self.estimators_:
            if self.bootstrap:
                indices = self._rng.integers(0, n, size=sample_size)
            else:
                indices = np.arange(n)
                if sample_size < n:
                    indices = self._rng.choice(indices, size=sample_size, replace=False)
            tree.partial_fit(X[indices], y[indices])

        return self

    def fit(self, X, y) -> "EnsembleClassifier":
        """Fit from one batch. Provided for convenience."""
        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(self.random_state)
        return self.partial_fit(X, y)

    def predict(self, X) -> np.ndarray:
        """Predict labels by majority voting."""
        if not self.estimators_:
            raise ValueError("EnsembleClassifier must be fitted before predict().")
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        predictions = np.asarray([tree.predict(X) for tree in self.estimators_])
        return self._majority_vote(predictions)

    def predict_proba(self, X) -> np.ndarray:
        """
        Estimate class probabilities from vote proportions.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if self.classes_ is None:
            raise ValueError("EnsembleClassifier must be fitted before predict_proba().")
        X = _as_2d_float_array(X)
        predictions = np.asarray([tree.predict(X) for tree in self.estimators_])
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        for col, class_label in enumerate(self.classes_):
            proba[:, col] = np.mean(predictions == class_label, axis=0)

        row_sums = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, row_sums, out=np.zeros_like(proba), where=row_sums > 0)

    def score(self, X, y) -> float:
        """Return accuracy on X and y."""
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y contain different numbers of samples.")
        return float(np.mean(pred == y))

    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        # predictions shape: (n_estimators, n_samples)
        labels = self.classes_
        out = []
        for sample_preds in predictions.T:
            counts = np.array([np.sum(sample_preds == label) for label in labels])
            out.append(labels[np.argmax(counts)])
        return np.asarray(out)
# Ensemble base
# BaggingClassifier for streaming
# BoostingClassifier stub
# majority vote predict
# Poisson sampling in bagging
