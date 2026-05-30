# NumCompute-Stream

## Project Title

**NumCompute-Stream: A Modular Ensemble Tree-Based Streaming Machine Learning Framework**

## Overview

NumCompute-Stream is a modular streaming machine-learning framework developed for Assignment 2.2: Programming Task 2. The project extends the NumCompute concept by implementing a decision tree-based classification system that supports streaming data, incremental preprocessing, model ensembling, streaming metrics, benchmarking and visualisation.

The framework is designed to process data in chunks rather than training on the full dataset at once. This simulates a real-world online learning scenario where new data arrives continuously. The main learning interface is based on `partial_fit()` for models and transformers, while metrics use `update()`, `result()` and `reset()` methods.

Only **Python, NumPy and matplotlib** are used. External machine-learning or data-processing libraries such as scikit-learn, pandas, TensorFlow and PyTorch are not used.

---

## Project Structure

```text
NumCompute-Stream/
│
├── numcompute_stream/
│   ├── __init__.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── stats.py
│   ├── pipeline.py
│   ├── stream.py
│   ├── visualise.py
│   └── io.py
│
├── tests/
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   ├── test_stats.py
│   ├── test_pipeline.py
│   └── test_stream.py
│
├── demo/
│   ├── stream_demo.ipynb
│   ├── sample_data.csv
│   ├── tree_accuracy_over_time.png
│   ├── ensemble_accuracy_over_time.png
│   ├── tree_vs_ensemble_accuracy.png
│   └── predictions_vs_ground_truth.png
│
├── benchmark/
│   ├── benchmark_stream.py
│   ├── benchmark_results.csv
│   ├── benchmark_summary.txt
│   └── model_accuracy_comparison.png
│
├── README.md
└── report.pdf
```

---

## Main Features

* Streaming classification using chunk-wise `partial_fit()`
* Depth-limited decision tree classifier
* Gini and entropy split criteria
* Bagging/random-forest style ensemble classifier
* Streaming preprocessing using imputation and standardisation
* Incremental one-hot encoding for categorical values
* Streaming metrics including accuracy, precision, recall, F1, confusion matrix and AUC
* Streaming statistical summaries including mean, variance, quantiles and histograms
* Pipeline support for chaining preprocessing and modelling steps
* Stream training manager with chunk-level logs
* Matplotlib-based visualisation functions
* Benchmarking for model comparison and loop vs vectorised preprocessing
* Unit testing with 48 tests

---

## Module Summary

| Module             | Description                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `tree.py`          | Implements `DecisionTreeClassifier` with depth limit, Gini/entropy splitting and streaming `partial_fit()` |
| `ensemble.py`      | Implements `EnsembleClassifier` using multiple decision trees, bootstrap sampling and majority voting      |
| `preprocessing.py` | Provides `Imputer`, `StandardScaler` and `OneHotEncoder` with incremental update support                   |
| `metrics.py`       | Provides streaming classification metrics with `update()`, `result()` and `reset()`                        |
| `stats.py`         | Provides streaming mean, variance, quantile and histogram support                                          |
| `pipeline.py`      | Chains preprocessing and model steps into one streaming workflow                                           |
| `stream.py`        | Provides `StreamTrainer` for chunk-wise training, scoring and logging                                      |
| `visualise.py`     | Provides reusable matplotlib plotting functions                                                            |
| `io.py`            | Provides CSV loading and chunk generation functions                                                        |

---

## Installation and Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install numpy matplotlib notebook
```

No other machine-learning or data-processing libraries are required.

---

## Running the Unit Tests

To run all tests, open a terminal in the project root folder and execute:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected output:

```text
Ran 48 tests in 1.143s

OK
```

The test suite covers the decision tree, ensemble classifier, preprocessing, metrics, statistics, pipeline and stream trainer modules.

---

## Running the Benchmark

To run the benchmark script:

```bash
python benchmark/benchmark_stream.py
```

The script compares:

1. `DecisionTreeClassifier`
2. `EnsembleClassifier`
3. Loop-based preprocessing vs NumPy-vectorised preprocessing

The benchmark generates the following files:

```text
benchmark/benchmark_results.csv
benchmark/benchmark_summary.txt
benchmark/model_accuracy_comparison.png
```

### Benchmark Summary

| Model                  | Samples | Chunks | Final Cumulative Accuracy | Final Chunk Accuracy |     Time | Peak Memory |
| ---------------------- | ------: | -----: | ------------------------: | -------------------: | -------: | ----------: |
| DecisionTreeClassifier |    2000 |     20 |                    0.9225 |               0.8800 |  7.8793s |   0.6528 MB |
| EnsembleClassifier     |    2000 |     20 |                    0.9230 |               0.9000 | 25.6257s |   1.4460 MB |

The vectorised preprocessing benchmark showed a speedup of approximately **99.05x** compared with the loop-based implementation.

---

## Running the Demo Notebook

To run the demo notebook:

```bash
jupyter notebook
```

Then open:

```text
demo/stream_demo.ipynb
```

Run all cells using:

```text
Kernel → Restart & Run All
```

The notebook demonstrates:

* Loading CSV data using `io.py`
* Splitting data into chunks
* Incremental training using `partial_fit()`
* Single decision tree training
* Ensemble model training
* Streaming accuracy, precision, recall and F1 tracking
* Memory and chunk-level logging
* Visualisation of accuracy and prediction results

The notebook creates the following plots:

```text
demo/tree_accuracy_over_time.png
demo/ensemble_accuracy_over_time.png
demo/tree_vs_ensemble_accuracy.png
demo/predictions_vs_ground_truth.png
```

---

## Example Usage

```python
from numcompute_stream import (
    Pipeline,
    Imputer,
    StandardScaler,
    EnsembleClassifier,
    StreamTrainer,
    StreamingAccuracy,
)
from numcompute_stream.io import read_csv, chunk_iter

X, y = read_csv("demo/sample_data.csv", target_column=-1)

pipe = Pipeline([
    ("imputer", Imputer()),
    ("scaler", StandardScaler()),
    ("model", EnsembleClassifier(
        n_estimators=7,
        method="random_forest",
        max_depth=5,
        random_state=42
    )),
])

trainer = StreamTrainer(
    pipeline=pipe,
    metrics={"accuracy": StreamingAccuracy()}
)

for X_chunk, y_chunk in chunk_iter(X, y, chunk_size=30):
    log = trainer.fit_chunk(X_chunk, y_chunk)
    print(log)
```

---

## Dataset

The demo uses a synthetic CSV dataset stored in:

```text
demo/sample_data.csv
```

The dataset contains:

* 300 rows
* 6 numeric features
* Binary target labels: `0` and `1`
* Some missing values to test the streaming imputer

The benchmark script generates its own synthetic dataset with:

* 2000 samples
* 8 features
* 20 chunks
* Chunk size of 100

---

## Edge Cases Handled

The implementation handles several important edge cases:

* Missing values using running mean imputation
* Zero-variance features in standardisation
* Empty chunks
* Mismatched feature and label lengths
* Feature-size mismatches
* Unknown categories in one-hot encoding
* Prediction before fitting
* Voting ties in ensemble prediction
* Single-class AUC cases

---

## Limitations

The decision tree uses chunk-wise accumulated retraining rather than a fully advanced online tree algorithm such as a Hoeffding tree. This design was selected to keep the implementation clear, testable and aligned with the assignment requirements. The AUC implementation is basic and mainly suitable for binary classification. Quantile estimation is based on a retained buffer rather than a specialised streaming quantile algorithm.

Despite these limitations, the framework successfully demonstrates modular streaming machine learning, incremental preprocessing, tree-based classification, ensemble learning, streaming metrics, benchmarking and visualisation using only NumPy and matplotlib.

---

## Author

Student Name:
Student ID:
Course/Module:
Assignment: Assignment 2.2 Programming Task 2
# NumCompute-Stream
## Installation
## API Reference
## Contributing
## Usage Examples
![Python](https://img.shields.io/badge/python-3.10+-blue)
## Known Limitations
## Future Work
<!-- v0.1.0 release -->
