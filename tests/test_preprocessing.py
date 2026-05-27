import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler, Imputer, OneHotEncoder


class TestStandardScaler(unittest.TestCase):
    def test_partial_fit_matches_numpy_mean_variance(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        X_all = np.vstack([X1, X2])

        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        np.testing.assert_allclose(scaler.mean_, X_all.mean(axis=0))
        np.testing.assert_allclose(scaler.var_, X_all.var(axis=0))

    def test_transform_has_zero_mean_after_fit(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)

        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(2), atol=1e-12)

    def test_zero_variance_does_not_divide_by_zero(self):
        X = np.array([[4.0, 2.0], [4.0, 3.0], [4.0, 4.0]])
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)

        self.assertTrue(np.all(np.isfinite(Xt)))
        np.testing.assert_allclose(Xt[:, 0], np.zeros(3))

    def test_nan_values_are_ignored_in_running_statistics(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.partial_fit(X)

        np.testing.assert_allclose(scaler.mean_, np.array([3.0, 5.0]))
        np.testing.assert_allclose(scaler.count_, np.array([3.0, 2.0]))


class TestImputer(unittest.TestCase):
    def test_imputer_replaces_nan_with_column_mean(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        imputer = Imputer()
        Xt = imputer.fit_transform(X)

        self.assertFalse(np.isnan(Xt).any())
        self.assertEqual(Xt[0, 1], 5.0)

    def test_imputer_updates_across_chunks(self):
        imputer = Imputer()
        imputer.partial_fit(np.array([[1.0], [3.0]]))
        imputer.partial_fit(np.array([[5.0], [np.nan]]))

        np.testing.assert_allclose(imputer.statistics_, np.array([3.0]))

    def test_imputer_uses_fill_value_when_column_all_nan(self):
        X = np.array([[np.nan], [np.nan]])
        imputer = Imputer(fill_value=-1.0)
        Xt = imputer.fit_transform(X)

        np.testing.assert_allclose(Xt, np.array([[-1.0], [-1.0]]))


class TestOneHotEncoder(unittest.TestCase):
    def test_one_hot_encoder_expands_categories_incrementally(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([["red"], ["blue"]], dtype=object))
        enc.partial_fit(np.array([["green"]], dtype=object))

        Xt = enc.transform(np.array([["red"], ["green"]], dtype=object))

        self.assertEqual(Xt.shape, (2, 3))
        self.assertTrue(np.all(Xt.sum(axis=1) == 1))

    def test_one_hot_encoder_ignores_unknown_when_configured(self):
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.partial_fit(np.array([["yes"], ["no"]], dtype=object))

        Xt = enc.transform(np.array([["maybe"]], dtype=object))

        np.testing.assert_allclose(Xt, np.zeros((1, 2)))

    def test_one_hot_encoder_errors_on_unknown_when_configured(self):
        enc = OneHotEncoder(handle_unknown="error")
        enc.partial_fit(np.array([["yes"], ["no"]], dtype=object))

        with self.assertRaises(ValueError):
            enc.transform(np.array([["maybe"]], dtype=object))


if __name__ == "__main__":
    unittest.main()
# test_preprocessing: scaler tests
# LabelEncoder and OneHotEncoder tests
