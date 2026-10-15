"""
FinRisk-Engine: Underwriting & Credit Risk Scoring Decision Engine
Translates raw model Probability of Default (PD) into actionable banking decisions,
risk tiers, affordability caps, and transparent risk explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from feature_engineering import engineer_features

class CreditRiskEngine:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'credit_risk_model.joblib')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")
            
        self.model = joblib.load(model_path)
        
    def evaluate_applicant(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw applicant dictionary, engineers features, predicts PD, 
        and executes underwriting policy rules.
        """
        # Convert dictionary to single-row DataFrame
        df_input = pd.DataFrame([applicant_data])
        
        # Apply banking feature engineering
        df_engineered = engineer_features(df_input)
        
        # Predict Default Probability (PD)
        pd_score = float(self.model.predict_proba(df_engineered)[0, 1])
        
        # 1. Determine Basel Risk Rating Tier
        if pd_score < 0.05:
            risk_tier = "Tier AAA (Super-Prime)"
            risk_label = "Very Low Risk"
        elif pd_score < 0.10:
            risk_tier = "Tier AA (Prime)"
            risk_label = "Low Risk"
        elif pd_score < 0.18:
            risk_tier = "Tier A (Good)"
            risk_label = "Moderate-Low Risk"
        elif pd_score < 0.28:
            risk_tier = "Tier BBB (Moderate)"
            risk_label = "Acceptable Risk"
        elif pd_score < 0.40:
            risk_tier = "Tier BB (Near-Prime)"
            risk_label = "Elevated Risk"
        elif pd_score < 0.55:
            risk_tier = "Tier B (Subprime)"
            risk_label = "High Risk"
        else:
            risk_tier = "Tier CCC/D (Distressed)"
            risk_label = "Critical Default Risk"
            
        # 2. Extract Key Financial Ratios
        dti = float(df_engineered['debt_to_income_ratio'].iloc[0])
        monthly_income = float(df_engineered['monthly_income'].iloc[0])
        monthly_payment = float(df_engineered['monthly_loan_payment'].iloc[0])
        credit_score = int(applicant_data.get('credit_score', 650))
        loan_amount = float(applicant_data.get('loan_amnt', 10000))
        annual_income = float(applicant_data.get('person_income', 50000))
        default_history = str(applicant_data.get('cb_person_default_on_file', 'N')).upper()
        
        # 3. Maximum Recommended Loan (Affordability Ceiling based on 40% maximum allowable DTI)
        max_allowable_monthly_debt = monthly_income * 0.40
        # Inverse amortization to find principal
        monthly_rate = (float(applicant_data.get('loan_int_rate', 10.0)) / 100.0) / 12.0
        n_months = 36
        if monthly_rate > 0:
            max_loan_cap = max_allowable_monthly_debt * ((1 + monthly_rate)**n_months - 1) / (monthly_rate * (1 + monthly_rate)**n_months)
        else:
            max_loan_cap = max_allowable_monthly_debt * n_months
        max_recommended_loan = float(np.clip(np.round(max_loan_cap, -2), 1000, 100000))
        
        # 4. Underwriting Policy Decision Logic
        risk_factors = []
        positive_factors = []
        
        # Analyze Risk Factors
        if pd_score >= 0.35:
            risk_factors.append(f"High statistical Probability of Default ({pd_score*100:.1f}%) exceeds policy threshold (35%).")
        if dti > 0.45:
            risk_factors.append(f"Debt-to-Income ratio ({dti*100:.1f}%) exceeds prudent ceiling (45%).")
        if credit_score < 580:
            risk_factors.append(f"Credit Bureau Score ({credit_score}) is below institutional benchmark (580).")
        if default_history == 'Y':
            risk_factors.append("Applicant has documented historical default on file.")
        if (loan_amount / annual_income) > 0.50:
            risk_factors.append(f"Requested loan exceeds 50% of gross annual income.")
            
        # Analyze Positive Factors
        if credit_score >= 720:
            positive_factors.append(f"Strong Credit Bureau Score ({credit_score}).")
        if dti <= 0.25:
            positive_factors.append(f"Healthy Debt-to-Income coverage ({dti*100:.1f}%).")
        if int(applicant_data.get('person_emp_length', 0)) >= 5:
            positive_factors.append(f"Stable employment history ({applicant_data.get('person_emp_length')} years).")
        if default_history == 'N':
            positive_factors.append("Clean credit track record with zero historical defaults.")
            
        # Decision Matrix
        if pd_score < 0.25 and dti <= 0.42 and credit_score >= 600 and default_history != 'Y':
            decision = "APPROVED"
            decision_color = "#28a745"
            summary_statement = "Application meets institutional underwriting criteria for automated approval."
        elif pd_score < 0.40 and dti <= 0.50 and credit_score >= 540:
            decision = "MANUAL_REVIEW"
            decision_color = "#ffc107"
            summary_statement = "Application flagged for Senior Underwriter review due to borderline risk metrics."
        else:
            decision = "REJECTED"
            decision_color = "#dc3545"
            summary_statement = "Application declined due to elevated credit risk exceeding institutional appetite."
            
        return {
            'probability_of_default': round(pd_score, 4),
            'probability_of_default_pct': f"{pd_score * 100:.2f}%",
            'risk_tier': risk_tier,
            'risk_label': risk_label,
            'decision': decision,
            'decision_color': decision_color,
            'summary_statement': summary_statement,
            'dti_ratio': round(dti, 4),
            'dti_ratio_pct': f"{dti * 100:.2f}%",
            'monthly_income': round(monthly_income, 2),
            'monthly_payment': round(monthly_payment, 2),
            'max_recommended_loan': max_recommended_loan,
            'risk_factors': risk_factors if risk_factors else ["No severe risk flags detected."],
            'positive_factors': positive_factors
        }
