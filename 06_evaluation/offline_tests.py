"""Deterministic tests; do not call a model or require an API key."""
import math
import unittest
from data_tools import load_dataset, inspect_dataframe, describe_column, fit_ols
from diagnostic_tools import check_ols_diagnostics

class TestStatisticalTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset('biomarker')

    def test_dataset(self):
        info = inspect_dataframe(self.df)
        self.assertEqual(info['rows'], 200)
        self.assertIn('age', info['columns'])

    def test_mean(self):
        observed = describe_column(self.df, 'age')['mean']
        self.assertTrue(math.isclose(observed, float(self.df.age.mean()), abs_tol=1e-12))

    def test_ols(self):
        m = fit_ols(self.df, 'biomarker', ['age', 'sex'])
        self.assertEqual(m['n_observations'], 200)
        self.assertAlmostEqual(m['coefficients']['age'], 0.772, places=2)
        self.assertAlmostEqual(m['coefficients']['sex'], 11.281, places=2)

    def test_diagnostics(self):
        d = check_ols_diagnostics(self.df, 'biomarker', ['age', 'sex'])
        self.assertGreaterEqual(d['shapiro_wilk_p'], 0)
        self.assertLessEqual(d['shapiro_wilk_p'], 1)
        self.assertGreaterEqual(d['breusch_pagan_lm_p'], 0)
        self.assertLessEqual(d['breusch_pagan_lm_p'], 1)

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            load_dataset('phantom')

    def test_unsupported_outcome(self):
        with self.assertRaises(ValueError):
            fit_ols(self.df, 'age', ['sex'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
