"""
Streaming pipeline.

The Pipeline chains transformers and a final estimator. Transformers should
implement partial_fit() and transform(). The final estimator should implement
partial_fit() and predict().
"""

from __future__ import annotations

import inspect
import numpy as np


class Pipeline:
    """
    Sequential streaming pipeline.

    Example
    -------
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier())
    ])

    pipe.partial_fit(X_chunk, y_chunk)
    y_pred = pipe.predict(X_chunk)
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        self.steps = list(steps)
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")
        for name, step in self.steps:
            if not isinstance(name, str) or not name:
                raise ValueError("Each pipeline step name must be a non-empty string.")
            if step is None:
                raise ValueError(f"Step {name!r} cannot be None.")

    @property
    def named_steps(self) -> dict:
        """Dictionary mapping step names to step objects."""
        return dict(self.steps)

    def _call_partial_fit(self, step, X, y=None):
        if not hasattr(step, "partial_fit"):
            return step
        try:
            sig = inspect.signature(step.partial_fit)
            if "y" in sig.parameters or len(sig.parameters) >= 2:
                return step.partial_fit(X, y)
        except (TypeError, ValueError):
            pass
        try:
            return step.partial_fit(X, y)
        except TypeError:
            return step.partial_fit(X)

    def partial_fit(self, X, y) -> "Pipeline":
        """Incrementally fit transformers and the final estimator."""
        X_current = np.asarray(X)

        for name, step in self.steps[:-1]:
            self._call_partial_fit(step, X_current, y)
            if not hasattr(step, "transform"):
                raise ValueError(f"Transformer step {name!r} must implement transform().")
            X_current = step.transform(X_current)

        final_name, final_estimator = self.steps[-1]
        if not hasattr(final_estimator, "partial_fit"):
            raise ValueError(f"Final estimator {final_name!r} must implement partial_fit().")
        final_estimator.partial_fit(X_current, y)
        return self

    def predict(self, X):
        """Transform X through all transformers and predict with the final estimator."""
        X_current = np.asarray(X)

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Transformer step {name!r} must implement transform().")
            X_current = step.transform(X_current)

        final_name, final_estimator = self.steps[-1]
        if not hasattr(final_estimator, "predict"):
            raise ValueError(f"Final estimator {final_name!r} must implement predict().")
        return final_estimator.predict(X_current)

    def transform(self, X):
        """Apply transformer steps only and return transformed features."""
        X_current = np.asarray(X)
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(f"Transformer step {name!r} must implement transform().")
            X_current = step.transform(X_current)
        return X_current

    def score(self, X, y) -> float:
        """Return prediction accuracy."""
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y contain different numbers of samples.")
        return float(np.mean(pred == y))
# Pipeline orchestrator
# Pipeline.fit_transform()
# Pipeline step chaining
