"""
Reusable matplotlib visualisation helpers for streaming experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _finish_plot(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    """
    Plot a metric across stream chunks.

    Parameters
    ----------
    metric_values : sequence of float
        Metric values in chunk order.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str or None
        Optional file path for saving the figure.
    show : bool
        Whether to display the figure.
    """
    values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, values.size + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(chunks, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finish_plot(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison", ylabel="Metric", save_path=None, show=True):
    """
    Compare two models across streaming metric histories.

    Parameters
    ----------
    metric1, metric2 : sequence of float
        Metric values for two models.
    labels : tuple/list of str
        Names for the two models.
    """
    values1 = np.asarray(metric1, dtype=float)
    values2 = np.asarray(metric2, dtype=float)
    n = min(values1.size, values2.size)
    if n == 0:
        raise ValueError("metric1 and metric2 must contain at least one value.")

    chunks = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(chunks, values1[:n], marker="o", label=labels[0])
    ax.plot(chunks, values2[:n], marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _finish_plot(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth", save_path=None, show=True):
    """
    Visualise predicted and actual labels for the latest chunk.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    idx = np.arange(y_true.size)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(idx, y_true, marker="o", linestyle="-", label="Ground truth")
    ax.plot(idx, y_pred, marker="x", linestyle="--", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Class label")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _finish_plot(fig, save_path, show)
# Visualisation helpers
# line_plot() for accuracy over time
# plot_confusion_matrix() helper
# scatter_plot() for feature space
# animated_accuracy_curve()
