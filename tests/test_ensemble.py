import unittest
import numpy as np

from numcompute_stream.ensemble import EnsembleClassifier


class TestEnsembleClassifier(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(60, 4))
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

    def test_ensemble_partial_fit_predict_shape(self):
        model = EnsembleClassifier(n_estimators=5, max_depth=3, random_state=1)
        model.partial_fit(self.X, self.y)

        pred = model.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_ensemble_score_is_between_zero_and_one(self):
        model = EnsembleClassifier(n_estimators=5, max_depth=3, random_state=1)
        model.partial_fit(self.X, self.y)

        score = model.score(self.X, self.y)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ensemble_updates_across_chunks(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)
        model.partial_fit(self.X[:30], self.y[:30])
        model.partial_fit(self.X[30:], self.y[30:])

        pred = model.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_random_forest_method_uses_feature_subsets(self):
        model = EnsembleClassifier(
            n_estimators=3,
            method="random_forest",
            max_depth=3,
            random_state=1,
        )
        model.partial_fit(self.X, self.y)

        pred = model.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_predict_proba_rows_sum_to_one(self):
        model = EnsembleClassifier(n_estimators=5, max_depth=3, random_state=1)
        model.partial_fit(self.X, self.y)

        proba = model.predict_proba(self.X[:5])

        self.assertEqual(proba.shape[0], 5)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(5))

    def test_invalid_number_of_estimators_raises_error(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(n_estimators=0)

    def test_predict_before_fit_raises_error(self):
        model = EnsembleClassifier(n_estimators=3)

        with self.assertRaises(ValueError):
            model.predict(self.X)


if __name__ == "__main__":
    unittest.main()
# test_ensemble: BaggingClassifier
