"""
FinRisk-Engine: Feature Engineering & Preprocessing Pipeline
Computes key financial risk indicators and builds production-grade transformations.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = [
    'person_home_ownership', 
    'loan_intent', 
    'loan_grade', 
    'cb_person_default_on_file'
]

NUMERICAL_FEATURES = [
    'person_age', 
    'person_income', 
    'person_emp_length', 
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_cred_hist_length', 
    'credit_score',
    'monthly_income',
    'monthly_loan_payment',
    'debt_to_income_ratio',
    'credit_stability_index'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes banking credit risk ratios:
    - monthly_income
    - monthly_loan_payment (Amortized formula at 36-month loan term)
    - debt_to_income_ratio
    - credit_stability_index
    """
    data = df.copy()
    
    # 1. Monthly Gross Income
    data['monthly_income'] = data['person_income'] / 12.0
    
    # 2. Monthly Loan Payment (standard amortization formula for 36-month personal loan)
    monthly_rate = (data['loan_int_rate'] / 100.0) / 12.0
    n_months = 36
    data['monthly_loan_payment'] = data['loan_amnt'] * (
        (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
    )
    
    # 3. Debt to Income Ratio (Payment-to-Income / DTI)
    data['debt_to_income_ratio'] = data['monthly_loan_payment'] / (data['monthly_income'] + 1e-5)
    
    # 4. Credit Stability Index: Normalized score weighing credit history and bureau score
    norm_score = data['credit_score'] / 850.0
    norm_hist = np.clip(data['cb_person_cred_hist_length'] / 25.0, 0, 1.0)
    data['credit_stability_index'] = (0.70 * norm_score) + (0.30 * norm_hist)
    
    return data

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an scikit-learn ColumnTransformer for categorical and numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor
