import unittest
import numpy as np

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    StreamingAUC,
    StreamingClassificationMetrics,
)


class TestStreamingMetrics(unittest.TestCase):
    def test_accuracy_accumulates_across_chunks(self):
        metric = StreamingAccuracy()
        metric.update([1, 0, 1], [1, 1, 1])
        metric.update([0, 0], [0, 1])

        self.assertAlmostEqual(metric.result(), 3 / 5)

    def test_accuracy_rolling_window(self):
        metric = StreamingAccuracy(window_size=3)
        metric.update([1, 1, 1], [1, 0, 0])
        metric.update([0, 0], [0, 0])

        self.assertAlmostEqual(metric.result(), 2 / 3)

    def test_confusion_matrix_accumulates(self):
        cm = StreamingConfusionMatrix(labels=[0, 1])
        cm.update([0, 0, 1, 1], [0, 1, 1, 0])

        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(cm.result(), expected)

    def test_precision_recall_f1_macro(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]

        precision = StreamingPrecision(labels=[0, 1])
        recall = StreamingRecall(labels=[0, 1])
        f1 = StreamingF1(labels=[0, 1])

        precision.update(y_true, y_pred)
        recall.update(y_true, y_pred)
        f1.update(y_true, y_pred)

        self.assertGreaterEqual(precision.result(), 0.0)
        self.assertLessEqual(precision.result(), 1.0)
        self.assertGreaterEqual(recall.result(), 0.0)
        self.assertLessEqual(recall.result(), 1.0)
        self.assertGreaterEqual(f1.result(), 0.0)
        self.assertLessEqual(f1.result(), 1.0)

    def test_metric_reset(self):
        metric = StreamingAccuracy()
        metric.update([1, 1], [1, 0])
        metric.reset()

        self.assertEqual(metric.result(), 0.0)

    def test_auc_perfect_scores(self):
        auc = StreamingAUC(positive_label=1)
        auc.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        self.assertAlmostEqual(auc.result(), 1.0)

    def test_auc_single_class_returns_nan(self):
        auc = StreamingAUC(positive_label=1)
        auc.update([1, 1, 1], [0.7, 0.8, 0.9])

        self.assertTrue(np.isnan(auc.result()))

    def test_classification_metrics_container_returns_dictionary(self):
        metrics = StreamingClassificationMetrics(labels=[0, 1])
        metrics.update([0, 1, 1], [0, 1, 0])

        result = metrics.result()

        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1", result)
        self.assertIn("confusion_matrix", result)

    def test_mismatched_metric_lengths_raise_error(self):
        metric = StreamingAccuracy()

        with self.assertRaises(ValueError):
            metric.update([1, 0], [1])


if __name__ == "__main__":
    unittest.main()
# test_metrics: accuracy assertion
# test precision, recall, f1
# roc_auc test case
