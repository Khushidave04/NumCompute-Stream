import unittest
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ])
        self.y = np.array([0, 0, 1, 1, 1, 1])

    def test_tree_partial_fit_predicts_training_data_reasonably(self):
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.partial_fit(self.X, self.y)

        pred = tree.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)
        self.assertGreaterEqual(np.mean(pred == self.y), 0.8)

    def test_tree_supports_entropy_criterion(self):
        tree = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=0)
        tree.partial_fit(self.X, self.y)

        pred = tree.predict(self.X)

        self.assertEqual(pred.shape[0], self.X.shape[0])

    def test_tree_updates_across_chunks(self):
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.partial_fit(self.X[:3], self.y[:3])
        tree.partial_fit(self.X[3:], self.y[3:])

        pred = tree.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)
        self.assertIsNotNone(tree.root_)

    def test_tree_handles_nan_values(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.partial_fit(X, self.y)

        pred = tree.predict(X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_predict_before_fit_raises_error(self):
        tree = DecisionTreeClassifier()

        with self.assertRaises(ValueError):
            tree.predict(self.X)

    def test_invalid_criterion_raises_error(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="invalid")

    def test_feature_mismatch_raises_error(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.partial_fit(self.X, self.y)

        with self.assertRaises(ValueError):
            tree.predict(np.array([[1.0, 2.0, 3.0]]))


if __name__ == "__main__":
    unittest.main()
# test_tree: basic HoeffdingTree tests
# EFDT test suite
