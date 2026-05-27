"""
benchmark_stream.py

Benchmark script for the NumCompute-Stream assignment.

This script compares:
1. A single streaming DecisionTreeClassifier
2. A streaming EnsembleClassifier using bagging / random-forest style voting
3. A loop-based preprocessing operation vs. a NumPy-vectorised version

It uses only Python standard library, NumPy and the package modules.

Run from the project root:

    python benchmark/benchmark_stream.py

Expected outputs inside benchmark/:
    benchmark_results.csv
    benchmark_summary.txt
    model_accuracy_comparison.png
"""

from __future__ import annotations

import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Make imports work whether this file is run from the project root or
# directly from inside benchmark/.
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.io import chunk_iter
from numcompute_stream.visualise import compare_models


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------
def make_synthetic_classification(
    n_samples: int = 2000,
    n_features: int = 8,
    random_state: int = 42,
    missing_rate: float = 0.03,
):
    """
    Create a numeric binary classification dataset.

    The target is generated from a noisy linear relationship. This is enough
    for testing streaming behaviour without using scikit-learn or pandas.
    """
    rng = np.random.default_rng(random_state)

    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    weights = rng.normal(0.0, 1.0, size=n_features)

    logits = X @ weights + rng.normal(0.0, 0.5, size=n_samples)
    y = (logits > np.median(logits)).astype(int)

    # Add missing values to test streaming imputation.
    missing_mask = rng.random(X.shape) < missing_rate
    X[missing_mask] = np.nan

    return X, y


# ---------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------
def make_single_tree_pipeline(random_state: int = 42):
    """Create a streaming pipeline using one decision tree."""
    return Pipeline(
        [
            ("imputer", Imputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=5,
                    criterion="gini",
                    random_state=random_state,
                    max_samples=3000,
                ),
            ),
        ]
    )


def make_ensemble_pipeline(random_state: int = 42):
    """Create a streaming pipeline using a tree ensemble."""
    return Pipeline(
        [
            ("imputer", Imputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                EnsembleClassifier(
                    n_estimators=7,
                    method="random_forest",
                    max_depth=6,
                    min_samples_split=5,
                    criterion="gini",
                    bootstrap=True,
                    sample_fraction=0.9,
                    random_state=random_state,
                    max_samples=3000,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------
def run_streaming_model_benchmark(
    name: str,
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int = 100,
):
    """
    Train and evaluate one model chunk by chunk.

    Returns
    -------
    summary : dict
        Final timing, accuracy and memory summary.
    history : list[dict]
        Per-chunk logs produced by StreamTrainer.
    """
    trainer = StreamTrainer(
        pipeline=pipeline,
        metrics={"accuracy": StreamingAccuracy()},
        evaluate_before_fit=False,
    )

    tracemalloc.start()
    start_time = time.perf_counter()

    for X_chunk, y_chunk in chunk_iter(X, y, chunk_size=chunk_size, shuffle=False):
        trainer.fit_chunk(X_chunk, y_chunk)

    elapsed = time.perf_counter() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logs = trainer.logs()
    final_log = logs[-1]

    summary = {
        "model": name,
        "chunks": len(logs),
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "chunk_size": int(chunk_size),
        "elapsed_seconds": float(elapsed),
        "final_cumulative_accuracy": float(final_log["cumulative_accuracy"]),
        "final_chunk_accuracy": float(final_log["chunk_accuracy"]),
        "peak_memory_mb": float(peak_memory / (1024 * 1024)),
        "avg_seconds_per_chunk": float(elapsed / len(logs)),
    }

    return summary, logs


# ---------------------------------------------------------------------
# Loop vs vectorised benchmark
# ---------------------------------------------------------------------
def loop_standardize(X: np.ndarray):
    """
    Slow loop-based standardisation.

    This function intentionally uses Python loops to provide a baseline
    comparison against NumPy vectorisation.
    """
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X, dtype=float)

    rows, cols = X.shape
    for j in range(cols):
        values = []
        for i in range(rows):
            if not np.isnan(X[i, j]):
                values.append(X[i, j])

        if len(values) == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            if std == 0:
                std = 1.0

        for i in range(rows):
            if np.isnan(X[i, j]):
                out[i, j] = np.nan
            else:
                out[i, j] = (X[i, j] - mean) / std

    return out


def vectorized_standardize(X: np.ndarray):
    """Fast NumPy-vectorised standardisation."""
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def run_loop_vs_vectorized_benchmark(X: np.ndarray, repeats: int = 5):
    """
    Compare loop-based and vectorised preprocessing speed.
    """
    loop_times = []
    vectorized_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        loop_result = loop_standardize(X)
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        vectorized_result = vectorized_standardize(X)
        vectorized_times.append(time.perf_counter() - start)

    # Compare finite values only because NaNs remain NaNs in both outputs.
    finite_mask = np.isfinite(loop_result) & np.isfinite(vectorized_result)
    max_difference = float(
        np.max(np.abs(loop_result[finite_mask] - vectorized_result[finite_mask]))
    )

    return {
        "loop_avg_seconds": float(np.mean(loop_times)),
        "vectorized_avg_seconds": float(np.mean(vectorized_times)),
        "speedup": float(np.mean(loop_times) / max(np.mean(vectorized_times), 1e-12)),
        "max_difference": max_difference,
    }


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------
def write_model_results_csv(path: Path, summaries: list[dict]):
    """Write model benchmark summaries to CSV."""
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_text(
    path: Path,
    model_summaries: list[dict],
    loop_vectorized_summary: dict,
):
    """Write a readable benchmark summary."""
    lines = []
    lines.append("NumCompute-Stream Benchmark Summary")
    lines.append("=" * 40)
    lines.append("")

    lines.append("Streaming model comparison")
    lines.append("-" * 40)
    for item in model_summaries:
        lines.append(f"Model: {item['model']}")
        lines.append(f"  Samples: {item['samples']}")
        lines.append(f"  Features: {item['features']}")
        lines.append(f"  Chunks: {item['chunks']}")
        lines.append(f"  Chunk size: {item['chunk_size']}")
        lines.append(f"  Final cumulative accuracy: {item['final_cumulative_accuracy']:.4f}")
        lines.append(f"  Final chunk accuracy: {item['final_chunk_accuracy']:.4f}")
        lines.append(f"  Total elapsed seconds: {item['elapsed_seconds']:.4f}")
        lines.append(f"  Average seconds per chunk: {item['avg_seconds_per_chunk']:.6f}")
        lines.append(f"  Peak memory MB: {item['peak_memory_mb']:.4f}")
        lines.append("")

    lines.append("Loop vs vectorised preprocessing")
    lines.append("-" * 40)
    lines.append(f"Loop average seconds: {loop_vectorized_summary['loop_avg_seconds']:.6f}")
    lines.append(
        f"Vectorised average seconds: {loop_vectorized_summary['vectorized_avg_seconds']:.6f}"
    )
    lines.append(f"Speedup: {loop_vectorized_summary['speedup']:.2f}x")
    lines.append(f"Max numerical difference: {loop_vectorized_summary['max_difference']:.12f}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def extract_accuracy_history(logs: list[dict]):
    """Extract cumulative accuracy from trainer logs."""
    return [row["cumulative_accuracy"] for row in logs]


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 2000
    n_features = 8
    chunk_size = 100
    random_state = 42

    print("Generating synthetic streaming dataset...")
    X, y = make_synthetic_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        missing_rate=0.03,
    )

    print("Running single decision tree benchmark...")
    tree_summary, tree_logs = run_streaming_model_benchmark(
        name="DecisionTreeClassifier",
        pipeline=make_single_tree_pipeline(random_state=random_state),
        X=X,
        y=y,
        chunk_size=chunk_size,
    )

    print("Running ensemble classifier benchmark...")
    ensemble_summary, ensemble_logs = run_streaming_model_benchmark(
        name="EnsembleClassifier",
        pipeline=make_ensemble_pipeline(random_state=random_state),
        X=X,
        y=y,
        chunk_size=chunk_size,
    )

    print("Running loop vs vectorised preprocessing benchmark...")
    # Use a subset to keep the intentionally slow loop benchmark fast enough.
    loop_vectorized_summary = run_loop_vs_vectorized_benchmark(X[:1000], repeats=5)

    model_summaries = [tree_summary, ensemble_summary]

    csv_path = output_dir / "benchmark_results.csv"
    txt_path = output_dir / "benchmark_summary.txt"
    plot_path = output_dir / "model_accuracy_comparison.png"

    write_model_results_csv(csv_path, model_summaries)
    write_summary_text(txt_path, model_summaries, loop_vectorized_summary)

    print("Saving model accuracy comparison plot...")
    compare_models(
        extract_accuracy_history(tree_logs),
        extract_accuracy_history(ensemble_logs),
        labels=("Single tree", "Ensemble"),
        title="Streaming accuracy comparison",
        ylabel="Cumulative accuracy",
        save_path=str(plot_path),
        show=False,
    )

    print("")
    print("Benchmark complete.")
    print(f"CSV results: {csv_path}")
    print(f"Summary text: {txt_path}")
    print(f"Accuracy plot: {plot_path}")
    print("")
    print(txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
# Benchmark: throughput & accuracy
# measure instances per second
