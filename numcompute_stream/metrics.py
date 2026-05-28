"""
Streaming classification metrics.

Each metric supports:
- update(y_true_chunk, y_pred_chunk)
- result()
- reset()

Some metrics also support a rolling window using window_size.
"""

from __future__ import annotations

from collections import deque
import numpy as np


def _as_1d(a, name: str) -> np.ndarray:
    arr = np.asarray(a).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _validate_pair(y_true, y_pred):
    y_true = _as_1d(y_true, "y_true")
    y_pred = _as_1d(y_pred, "y_pred")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have same length; got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )
    return y_true, y_pred


class StreamingAccuracy:
    """Streaming accuracy with optional rolling-window accuracy."""

    def __init__(self, window_size: int | None = None):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.correct_ = 0
        self.total_ = 0
        self._window = deque(maxlen=self.window_size) if self.window_size else None
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = _validate_pair(y_true_chunk, y_pred_chunk)
        hits = (y_true == y_pred).astype(int)
        self.correct_ += int(hits.sum())
        self.total_ += int(hits.size)
        if self._window is not None:
            self._window.extend(hits.tolist())
        return self

    def result(self) -> float:
        if self._window is not None and len(self._window) > 0:
            return float(np.mean(self._window))
        return float(self.correct_ / self.total_) if self.total_ else 0.0


class StreamingConfusionMatrix:
    """Accumulates a confusion matrix over chunks."""

    def __init__(self, labels=None, window_size: int | None = None):
        self.initial_labels = None if labels is None else list(labels)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.labels_ = [] if self.initial_labels is None else list(self.initial_labels)
        self.matrix_ = np.zeros((len(self.labels_), len(self.labels_)), dtype=int)
        self._true_window = deque(maxlen=self.window_size) if self.window_size else None
        self._pred_window = deque(maxlen=self.window_size) if self.window_size else None
        return self

    def _ensure_labels(self, values):
        for value in values:
            if value not in self.labels_:
                self.labels_.append(value)
        new_size = len(self.labels_)
        if self.matrix_.shape[0] != new_size:
            new_matrix = np.zeros((new_size, new_size), dtype=int)
            old_size = self.matrix_.shape[0]
            new_matrix[:old_size, :old_size] = self.matrix_
            self.matrix_ = new_matrix

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = _validate_pair(y_true_chunk, y_pred_chunk)

        self._ensure_labels(np.concatenate([y_true, y_pred]))
        label_to_idx = {label: idx for idx, label in enumerate(self.labels_)}

        for true_value, pred_value in zip(y_true, y_pred):
            self.matrix_[label_to_idx[true_value], label_to_idx[pred_value]] += 1

        if self._true_window is not None:
            self._true_window.extend(y_true.tolist())
            self._pred_window.extend(y_pred.tolist())
        return self

    def result(self) -> np.ndarray:
        if self._true_window is not None and len(self._true_window) > 0:
            labels = self.labels_
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            mat = np.zeros((len(labels), len(labels)), dtype=int)
            for true_value, pred_value in zip(self._true_window, self._pred_window):
                mat[label_to_idx[true_value], label_to_idx[pred_value]] += 1
            return mat
        return self.matrix_.copy()


def _precision_recall_f1_from_matrix(matrix: np.ndarray, average: str = "macro"):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0.0, 0.0, 0.0

    tp = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    precision_per_class = np.divide(
        tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0
    )
    recall_per_class = np.divide(
        tp, actual, out=np.zeros_like(tp, dtype=float), where=actual > 0
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp, dtype=float),
        where=(precision_per_class + recall_per_class) > 0,
    )

    if average == "macro":
        return (
            float(np.mean(precision_per_class)) if precision_per_class.size else 0.0,
            float(np.mean(recall_per_class)) if recall_per_class.size else 0.0,
            float(np.mean(f1_per_class)) if f1_per_class.size else 0.0,
        )

    if average == "micro":
        total_tp = tp.sum()
        total = matrix.sum()
        micro = float(total_tp / total) if total else 0.0
        return micro, micro, micro

    raise ValueError("average must be 'macro' or 'micro'.")


class _MatrixMetric:
    def __init__(self, average: str = "macro", labels=None, window_size: int | None = None):
        self.average = average
        self.cm = StreamingConfusionMatrix(labels=labels, window_size=window_size)

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self


class StreamingPrecision(_MatrixMetric):
    """Streaming precision, macro or micro averaged."""

    def result(self) -> float:
        precision, _, _ = _precision_recall_f1_from_matrix(self.cm.result(), self.average)
        return precision


class StreamingRecall(_MatrixMetric):
    """Streaming recall, macro or micro averaged."""

    def result(self) -> float:
        _, recall, _ = _precision_recall_f1_from_matrix(self.cm.result(), self.average)
        return recall


class StreamingF1(_MatrixMetric):
    """Streaming F1 score, macro or micro averaged."""

    def result(self) -> float:
        _, _, f1 = _precision_recall_f1_from_matrix(self.cm.result(), self.average)
        return f1


class StreamingAUC:
    """
    Streaming binary AUC.

    The metric stores scores over time and computes the rank-based AUC. It can
    accept probability scores or discrete predictions. If only one class is
    present, result() returns np.nan.
    """

    def __init__(self, positive_label=1, window_size: int | None = None):
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        maxlen = self.window_size if self.window_size else None
        self.y_true_ = deque(maxlen=maxlen)
        self.y_score_ = deque(maxlen=maxlen)
        return self

    def update(self, y_true_chunk, y_score_chunk):
        y_true = _as_1d(y_true_chunk, "y_true")
        y_score = _as_1d(y_score_chunk, "y_score")
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score must have the same length.")
        self.y_true_.extend(y_true.tolist())
        self.y_score_.extend(np.asarray(y_score, dtype=float).tolist())
        return self

    def result(self) -> float:
        y_true = np.asarray(self.y_true_)
        y_score = np.asarray(self.y_score_, dtype=float)
        if y_true.size == 0:
            return np.nan

        binary = (y_true == self.positive_label).astype(int)
        n_pos = int(binary.sum())
        n_neg = int(binary.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.nan

        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

        # Average ranks for tied scores.
        unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
        for group_idx, count in enumerate(counts):
            if count > 1:
                tied = inverse == group_idx
                ranks[tied] = ranks[tied].mean()

        rank_sum_pos = ranks[binary == 1].sum()
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)


class StreamingClassificationMetrics:
    """
    Convenience container for common classification metrics.

    result() returns a dictionary containing accuracy, precision, recall, f1 and
    confusion_matrix.
    """

    def __init__(self, labels=None, average: str = "macro", window_size: int | None = None):
        self.accuracy = StreamingAccuracy(window_size=window_size)
        self.confusion_matrix = StreamingConfusionMatrix(labels=labels, window_size=window_size)
        self.average = average

    def reset(self):
        self.accuracy.reset()
        self.confusion_matrix.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.accuracy.update(y_true_chunk, y_pred_chunk)
        self.confusion_matrix.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self) -> dict:
        cm = self.confusion_matrix.result()
        precision, recall, f1 = _precision_recall_f1_from_matrix(cm, self.average)
        return {
            "accuracy": self.accuracy.result(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "labels": list(self.confusion_matrix.labels_),
        }
# Metrics: accuracy, kappa
# accuracy_score function
# kappa_score function
# precision_score
# recall_score
# f1_score
# roc_auc_score streaming
# geometric_mean_score
