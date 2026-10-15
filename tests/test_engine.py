"""
FinRisk-Engine: Automated Unit & Integration Test Suite
Validates feature engineering, model scoring bounds, underwriting decisions,
and database persistence.
"""

import sys
import os
import unittest
import pandas as pd

# Add src and database to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from feature_engineering import engineer_features
from risk_engine import CreditRiskEngine
from db import init_db, log_evaluation, get_recent_evaluations

class TestFinRiskEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CreditRiskEngine()
        init_db()

    def test_feature_engineering_ratios(self):
        sample = pd.DataFrame([{
            'person_age': 30,
            'person_income': 60000,
            'person_emp_length': 5,
            'loan_amnt': 12000,
            'loan_int_rate': 10.0,
            'loan_percent_income': 0.20,
            'cb_person_cred_hist_length': 8,
            'credit_score': 700,
            'person_home_ownership': 'RENT',
            'loan_intent': 'PERSONAL',
            'loan_grade': 'B',
            'cb_person_default_on_file': 'N'
        }])
        
        df_out = engineer_features(sample)
        self.assertIn('monthly_income', df_out.columns)
        self.assertIn('monthly_loan_payment', df_out.columns)
        self.assertIn('debt_to_income_ratio', df_out.columns)
        self.assertIn('credit_stability_index', df_out.columns)
        
        # Monthly income should be 60000 / 12 = 5000
        self.assertAlmostEqual(df_out['monthly_income'].iloc[0], 5000.0)

    def test_prime_applicant_approval(self):
        prime_applicant = {
            'person_age': 40,
            'person_income': 150000,
            'person_home_ownership': 'MORTGAGE',
            'person_emp_length': 12,
            'loan_intent': 'HOMEIMPROVEMENT',
            'loan_grade': 'A',
            'loan_amnt': 15000,
            'loan_int_rate': 6.5,
            'loan_percent_income': 0.10,
            'cb_person_default_on_file': 'N',
            'cb_person_cred_hist_length': 15,
            'credit_score': 800
        }
        res = self.engine.evaluate_applicant(prime_applicant)
        self.assertLess(res['probability_of_default'], 0.25)
        self.assertEqual(res['decision'], 'APPROVED')

    def test_subprime_applicant_rejection(self):
        subprime_applicant = {
            'person_age': 22,
            'person_income': 22000,
            'person_home_ownership': 'RENT',
            'person_emp_length': 0,
            'loan_intent': 'DEBTCONSOLIDATION',
            'loan_grade': 'F',
            'loan_amnt': 20000,
            'loan_int_rate': 24.0,
            'loan_percent_income': 0.91,
            'cb_person_default_on_file': 'Y',
            'cb_person_cred_hist_length': 2,
            'credit_score': 420
        }
        res = self.engine.evaluate_applicant(subprime_applicant)
        self.assertGreaterEqual(res['probability_of_default'], 0.35)
        self.assertEqual(res['decision'], 'REJECTED')

    def test_database_logging(self):
        applicant_data = {
            'person_age': 35,
            'person_income': 80000,
            'loan_amnt': 15000,
            'loan_intent': 'PERSONAL',
            'credit_score': 720
        }
        mock_eval = {
            'dti_ratio': 0.22,
            'probability_of_default': 0.08,
            'risk_tier': 'Tier AA (Prime)',
            'decision': 'APPROVED'
        }
        insert_id = log_evaluation("Test Applicant Unit", applicant_data, mock_eval, "Test suite log")
        self.assertGreater(insert_id, 0)
        
        recent = get_recent_evaluations(limit=5)
        names = [r['applicant_name'] for r in recent]
        self.assertIn("Test Applicant Unit", names)

if __name__ == '__main__':
    unittest.main()
