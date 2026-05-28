import unittest
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier


class TestPipeline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(456)
        self.X = rng.normal(size=(50, 3))
        self.X[0, 0] = np.nan
        self.y = (self.X[:, 1] > 0).astype(int)

    def test_pipeline_partial_fit_and_predict_with_tree(self):
        pipe = Pipeline([
            ("imputer", Imputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])

        pipe.partial_fit(self.X, self.y)
        pred = pipe.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_pipeline_partial_fit_and_predict_with_ensemble(self):
        pipe = Pipeline([
            ("imputer", Imputer()),
            ("scale", StandardScaler()),
            ("model", EnsembleClassifier(n_estimators=3, max_depth=3, random_state=0)),
        ])

        pipe.partial_fit(self.X, self.y)
        pred = pipe.predict(self.X)

        self.assertEqual(pred.shape, self.y.shape)

    def test_pipeline_score_between_zero_and_one(self):
        pipe = Pipeline([
            ("imputer", Imputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])

        pipe.partial_fit(self.X, self.y)
        score = pipe.score(self.X, self.y)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_named_steps_returns_dictionary(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2)),
        ])

        self.assertIn("scale", pipe.named_steps)
        self.assertIn("model", pipe.named_steps)

    def test_duplicate_step_names_raise_error(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("scale", StandardScaler()),
                ("scale", StandardScaler()),
                ("model", DecisionTreeClassifier()),
            ])


if __name__ == "__main__":
    unittest.main()
# end-to-end pipeline test
