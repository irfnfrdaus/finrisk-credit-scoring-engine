"""
FinRisk-Engine: Automated FastAPI Endpoint & Integration Test Suite
Validates health checks, Pydantic validation gates, single and batch underwriting endpoints.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure api, src, and database are in Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'api'))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from main import app

class TestFinRiskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("system"), "FinRisk-Engine")
        self.assertEqual(data.get("status"), "ONLINE")

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "HEALTHY")
        self.assertTrue(data.get("model_loaded"))
        self.assertTrue(data.get("database_connected"))

    def test_score_valid_applicant(self):
        payload = {
            "applicant_name": "Test Prime Candidate",
            "person_age": 35,
            "person_income": 95000.0,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 8.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "A",
            "loan_amnt": 20000.0,
            "loan_int_rate": 7.5,
            "loan_percent_income": 0.21,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 10,
            "credit_score": 760,
            "notes": "Automated CI/CD Integration Test"
        }
        response = self.client.post("/api/v1/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("evaluation_id", data)
        self.assertIn("probability_of_default", data)
        self.assertGreaterEqual(data["probability_of_default"], 0.0)
        self.assertLessEqual(data["probability_of_default"], 1.0)
        self.assertIn(data["decision"], ["APPROVED", "REFERRED", "REJECTED"])

    def test_score_invalid_underage_applicant(self):
        # Under 18 should fail Pydantic validation (422 Unprocessable Entity)
        invalid_payload = {
            "applicant_name": "Minor Applicant",
            "person_age": 16, # Violates ge=18
            "person_income": 30000.0,
            "person_home_ownership": "RENT",
            "person_emp_length": 1.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "C",
            "loan_amnt": 5000.0,
            "loan_int_rate": 12.0,
            "loan_percent_income": 0.15,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 1,
            "credit_score": 650
        }
        response = self.client.post("/api/v1/score", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_batch_scoring_endpoint(self):
        batch_payload = [
            {
                "applicant_name": "Batch Applicant 1",
                "person_age": 28,
                "person_income": 50000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 10000.0,
                "loan_int_rate": 10.5,
                "loan_percent_income": 0.20,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 5,
                "credit_score": 690
            },
            {
                "applicant_name": "Batch Applicant 2",
                "person_age": 45,
                "person_income": 120000.0,
                "person_home_ownership": "OWN",
                "person_emp_length": 15.0,
                "loan_intent": "VENTURE",
                "loan_grade": "A",
                "loan_amnt": 25000.0,
                "loan_int_rate": 6.0,
                "loan_percent_income": 0.15,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 18,
                "credit_score": 810
            }
        ]
        response = self.client.post("/api/v1/batch-score", json=batch_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("batch_size", data)
        self.assertEqual(data["batch_size"], 2)
        self.assertEqual(len(data.get("results", [])), 2)

    def test_audit_log_endpoint(self):
        response = self.client.get("/api/v1/audit-log?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_portfolio_metrics_endpoint(self):
        response = self.client.get("/api/v1/portfolio-metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

if __name__ == '__main__':
    unittest.main()
