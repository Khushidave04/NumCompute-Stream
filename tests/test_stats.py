import unittest
import numpy as np

from numcompute_stream.stats import StreamingStats


class TestStreamingStats(unittest.TestCase):
    def test_update_stats_mean_and_variance(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0]])
        X_all = np.vstack([X1, X2])

        stats = StreamingStats()
        stats.update_stats(X1)
        stats.update_stats(X2)
        result = stats.result()

        np.testing.assert_allclose(result["mean"], X_all.mean(axis=0))
        np.testing.assert_allclose(result["variance"], X_all.var(axis=0))

    def test_update_stats_ignores_nan_values(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        stats = StreamingStats()
        stats.update_stats(X)
        result = stats.result()

        np.testing.assert_allclose(result["mean"], np.array([3.0, 5.0]))
        np.testing.assert_allclose(result["count"], np.array([3.0, 2.0]))

    def test_quantiles_from_buffer(self):
        X = np.arange(10).reshape(10, 1)
        stats = StreamingStats()
        stats.update_stats(X)

        q = stats.quantiles(q=(0.5,), feature=0)

        np.testing.assert_allclose(q, np.array([4.5]))

    def test_histogram_counts_all_buffered_rows(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        stats = StreamingStats()
        stats.update_stats(X)

        counts, edges = stats.histogram(bins=2, feature=0)

        self.assertEqual(counts.sum(), 4)
        self.assertEqual(len(edges), 3)

    def test_reset_clears_statistics(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1.0], [2.0]]))
        stats.reset()

        self.assertIsNone(stats.result()["mean"])


if __name__ == "__main__":
    unittest.main()
# test_stats: RunningMean assertions
