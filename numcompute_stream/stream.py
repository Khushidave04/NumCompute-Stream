"""
Stream training manager.

StreamTrainer coordinates chunk-wise fitting, prediction, metric updates and
logging for a streaming pipeline.
"""

from __future__ import annotations

import sys
import time
import numpy as np

from .metrics import StreamingAccuracy, StreamingClassificationMetrics


def _estimate_size(obj, seen=None) -> int:
    """Rough recursive memory estimate in bytes, including NumPy array buffers."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, np.ndarray):
        return size + obj.nbytes

    if isinstance(obj, dict):
        size += sum(_estimate_size(k, seen) + _estimate_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, "__dict__"):
        size += _estimate_size(vars(obj), seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_estimate_size(item, seen) for item in obj)

    return size


class StreamTrainer:
    """
    Manages chunk-wise training and logging.

    Parameters
    ----------
    pipeline : object
        Object implementing partial_fit(X, y) and predict(X).
    metrics : dict or None
        Mapping from metric name to metric object. Metric objects should support
        update(y_true, y_pred), result() and reset().
    evaluate_before_fit : bool, default=False
        If True, performs prequential evaluation: predict first, then train.
        The first chunk will be trained without scoring if the model is not yet fitted.
    """

    def __init__(self, pipeline, metrics: dict | None = None, evaluate_before_fit: bool = False):
        self.pipeline = pipeline
        self.metrics = metrics or {"accuracy": StreamingAccuracy()}
        self.evaluate_before_fit = evaluate_before_fit
        self.chunk_index = 0
        self.samples_seen = 0
        self.correct_seen = 0
        self.history = []
        self.metric_history = {name: [] for name in self.metrics}

    def fit_chunk(self, X_chunk, y_chunk) -> dict:
        """
        Process one stream chunk.

        Returns
        -------
        dict
            Log record for the chunk.
        """
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk).ravel()
        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError("X_chunk and y_chunk must contain the same number of rows.")
        if X_chunk.shape[0] == 0:
            raise ValueError("Empty chunks are not allowed.")

        start = time.perf_counter()
        y_pred = None

        if self.evaluate_before_fit:
            try:
                y_pred = self.pipeline.predict(X_chunk)
            except Exception:
                y_pred = None
            self.pipeline.partial_fit(X_chunk, y_chunk)
            if y_pred is None:
                y_pred = self.pipeline.predict(X_chunk)
        else:
            self.pipeline.partial_fit(X_chunk, y_chunk)
            y_pred = self.pipeline.predict(X_chunk)

        elapsed = time.perf_counter() - start
        chunk_accuracy = float(np.mean(y_pred == y_chunk))

        self.samples_seen += int(y_chunk.shape[0])
        self.correct_seen += int(np.sum(y_pred == y_chunk))
        cumulative_accuracy = self.correct_seen / self.samples_seen

        metric_results = {}
        for name, metric in self.metrics.items():
            metric.update(y_chunk, y_pred)
            value = metric.result()
            metric_results[name] = value
            self.metric_history[name].append(value)

        self.chunk_index += 1
        log = {
            "chunk": self.chunk_index,
            "n_samples": int(y_chunk.shape[0]),
            "samples_seen": int(self.samples_seen),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": float(cumulative_accuracy),
            "elapsed_seconds": float(elapsed),
            "memory_bytes": int(_estimate_size(self.pipeline)),
            "metrics": metric_results,
        }
        self.history.append(log)
        return log

    def score_chunk(self, X_chunk, y_chunk, update_metrics: bool = False) -> dict:
        """Score a chunk without fitting. Optionally update streaming metrics."""
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk).ravel()
        y_pred = self.pipeline.predict(X_chunk)
        chunk_accuracy = float(np.mean(y_pred == y_chunk))

        metric_results = {}
        if update_metrics:
            for name, metric in self.metrics.items():
                metric.update(y_chunk, y_pred)
                value = metric.result()
                metric_results[name] = value
                self.metric_history[name].append(value)

        return {
            "n_samples": int(y_chunk.shape[0]),
            "chunk_accuracy": chunk_accuracy,
            "metrics": metric_results,
            "predictions": y_pred,
        }

    def logs(self) -> list[dict]:
        """Return all chunk logs."""
        return list(self.history)

    def reset_metrics(self):
        """Reset metric objects and metric history."""
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()
        self.metric_history = {name: [] for name in self.metrics}
        return self
# Stream reader/generator
# DataStream generator
# ConceptDriftDetector stub
# ADWIN implementation
# drift hook callbacks
