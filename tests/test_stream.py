import unittest
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import StreamingAccuracy


class TestStreamTrainer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(789)
        self.X = rng.normal(size=(40, 3))
        self.X[1, 2] = np.nan
        self.y = (self.X[:, 0] > 0).astype(int)

    def _make_trainer(self):
        pipe = Pipeline([
            ("imputer", Imputer()),
            ("scale", StandardScaler()),
            ("model", EnsembleClassifier(n_estimators=3, max_depth=3, random_state=0)),
        ])
        return StreamTrainer(pipe, metrics={"accuracy": StreamingAccuracy()})

    def test_fit_chunk_returns_log_dictionary(self):
        trainer = self._make_trainer()

        log = trainer.fit_chunk(self.X[:10], self.y[:10])

        self.assertIn("chunk", log)
        self.assertIn("chunk_accuracy", log)
        self.assertIn("cumulative_accuracy", log)
        self.assertIn("memory_bytes", log)

    def test_trainer_logs_multiple_chunks(self):
        trainer = self._make_trainer()

        trainer.fit_chunk(self.X[:20], self.y[:20])
        trainer.fit_chunk(self.X[20:], self.y[20:])

        logs = trainer.logs()

        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[-1]["samples_seen"], 40)

    def test_score_chunk_does_not_fit_new_data(self):
        trainer = self._make_trainer()
        trainer.fit_chunk(self.X[:20], self.y[:20])

        result = trainer.score_chunk(self.X[20:], self.y[20:])

        self.assertIn("chunk_accuracy", result)
        self.assertIn("predictions", result)
        self.assertEqual(result["predictions"].shape[0], 20)

    def test_empty_chunk_raises_error(self):
        trainer = self._make_trainer()

        with self.assertRaises(ValueError):
            trainer.fit_chunk(np.empty((0, 3)), np.array([]))

    def test_metric_history_is_recorded(self):
        trainer = self._make_trainer()

        trainer.fit_chunk(self.X[:20], self.y[:20])
        trainer.fit_chunk(self.X[20:], self.y[20:])

        self.assertIn("accuracy", trainer.metric_history)
        self.assertEqual(len(trainer.metric_history["accuracy"]), 2)


if __name__ == "__main__":
    unittest.main()
# test_stream: DataStream test
# ADWIN and PageHinkley tests
