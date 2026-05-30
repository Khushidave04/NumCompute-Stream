"""
NumCompute-Stream: a small streaming machine-learning framework.

Only NumPy and matplotlib are used for numerical and visualisation work.
The package provides incremental preprocessing, streaming metrics,
decision-tree learning, tree-based bagging ensembles, pipelines, logging,
statistics and plotting helpers.
"""

from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    StreamingAUC,
    StreamingClassificationMetrics,
)
from .stats import StreamingStats
from .pipeline import Pipeline
from .stream import StreamTrainer
from .io import read_csv, chunk_iter

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "StandardScaler",
    "Imputer",
    "OneHotEncoder",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingConfusionMatrix",
    "StreamingAUC",
    "StreamingClassificationMetrics",
    "StreamingStats",
    "Pipeline",
    "StreamTrainer",
    "read_csv",
    "chunk_iter",
]
# NumCompute-Stream: streaming ML library
# type hints added to public API
__all__ = ["HoeffdingTree", "ARF", "Pipeline"]
__version__ = "0.1.0"
