"""Offline correctness checks. No SDK, API key or external network needed."""
import math,unittest
from tools.statistical_tools import load_dataset,inspect_dataset,describe_numeric,regression,regression_diagnostics
from specialists.schemas import StatisticalFinding

class Tests(unittest.TestCase):
    def test_data(self):
        self.assertEqual(inspect_dataset('biomarker')['rows'],200)
        with self.assertRaises(ValueError):load_dataset('../secret')
    def test_descriptives(self):
        z=describe_numeric('biomarker','age');self.assertEqual(z['n'],200)
        self.assertTrue(math.isfinite(z['mean']))
    def test_regression(self):
        m=regression('biomarker','biomarker',['age','sex'])
        self.assertEqual(m['n'],200)
        self.assertIn('age',m['coefficients'])
        self.assertTrue(math.isfinite(m['coefficients']['age']))
    def test_diagnostics(self):
        d=regression_diagnostics('biomarker','biomarker',['age','sex'])
        self.assertTrue(0<=d['breusch_pagan_p']<=1)
    def test_invalid(self):
        with self.assertRaises(ValueError):regression('biomarker','biomarker',['age','age'])
    def test_schema(self):
        x=StatisticalFinding(dataset='biomarker',outcome='biomarker',predictors=['age'],
            sample_size=200,coefficients={'age':0.7},limitations=['observational data'])
        self.assertEqual(x.sample_size,200)

if __name__=='__main__':unittest.main(verbosity=2)
