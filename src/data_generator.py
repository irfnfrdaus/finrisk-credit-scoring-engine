"""
FinRisk-Engine: Synthetic Banking Loan Portfolio Generator
Simulates realistic enterprise credit risk data calibrated to Basel II/III standards.
"""

import os
import numpy as np
import pandas as pd

def generate_loan_portfolio(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Age and Employment Length
    age = np.random.normal(loc=35, scale=10, size=n_samples).astype(int)
    age = np.clip(age, 20, 68)
    
    emp_length = (age - 18) * np.random.uniform(0.1, 0.85, size=n_samples)
    emp_length = np.clip(np.round(emp_length), 0, 42).astype(int)
    
    # 2. Annual Income (Log-normal distribution for realistic income skew)
    income = np.random.lognormal(mean=10.9, sigma=0.6, size=n_samples)
    income = np.clip(np.round(income, -2), 18000, 350000)
    
    # 3. Home Ownership
    home_ownership_types = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    home_ownership = np.random.choice(
        home_ownership_types, 
        size=n_samples, 
        p=[0.48, 0.42, 0.08, 0.02]
    )
    
    # 4. Loan Intent
    intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
    loan_intent = np.random.choice(
        intents,
        size=n_samples,
        p=[0.18, 0.17, 0.16, 0.18, 0.20, 0.11]
    )
    
    # 5. Loan Amount
    loan_amnt = np.random.lognormal(mean=9.2, sigma=0.75, size=n_samples)
    loan_amnt = np.clip(np.round(loan_amnt, -2), 1500, 45000)
    
    # 6. Credit Bureau Score (FICO-like: 300 to 850)
    credit_score = np.random.normal(loc=670, scale=65, size=n_samples).astype(int)
    credit_score = np.clip(credit_score, 320, 850)
    
    # 7. Credit History Length (years)
    cred_hist_length = np.clip(np.round((age - 18) * np.random.uniform(0.3, 0.9, size=n_samples)), 1, 35).astype(int)
    
    # 8. Historical Default on File
    # Lower credit score significantly increases probability of having default on file
    default_prob_historical = 1 / (1 + np.exp((credit_score - 580) / 45))
    default_on_file = np.where(np.random.uniform(0, 1, size=n_samples) < default_prob_historical, 'Y', 'N')
    
    # 9. Interest Rate (Determined by credit score + loan grade base)
    base_rate = 19.5 - (credit_score - 300) * (14.0 / 550)
    interest_rate = base_rate + np.random.normal(0, 1.2, size=n_samples)
    interest_rate = np.clip(np.round(interest_rate, 2), 5.25, 23.50)
    
    # 10. Loan Percent of Income (DTI proxy)
    loan_percent_income = np.round(loan_amnt / income, 2)
    
    # 11. Loan Grade Assignment (A through G)
    def assign_grade(score, rate):
        if score >= 750 and rate <= 8.5:
            return 'A'
        elif score >= 700 and rate <= 11.5:
            return 'B'
        elif score >= 650 and rate <= 14.5:
            return 'C'
        elif score >= 600 and rate <= 17.5:
            return 'D'
        elif score >= 540 and rate <= 20.0:
            return 'E'
        else:
            return 'F'
            
    loan_grade = [assign_grade(s, r) for s, r in zip(credit_score, interest_rate)]
    
    # 12. Synthetic Ground-Truth Probability of Default (Logit Equation)
    # Calibrated realistic banking risk formula
    logit = (
        - 0.0085 * (credit_score - 650)
        + 4.2 * (loan_percent_income - 0.20)
        + 0.08 * (interest_rate - 11.0)
        - 0.04 * emp_length
        + 1.15 * (default_on_file == 'Y').astype(int)
        + 0.35 * (home_ownership == 'RENT').astype(int)
        - 0.40 * (home_ownership == 'OWN').astype(int)
        + 0.25 * (loan_intent == 'DEBTCONSOLIDATION').astype(int)
        - 2.10 # Base intercept yielding ~18-22% portfolio default rate
    )
    
    prob_default = 1 / (1 + np.exp(-logit))
    # Add idiosyncratic random shock
    loan_status = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': interest_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': default_on_file,
        'cb_person_cred_hist_length': cred_hist_length,
        'credit_score': credit_score,
        'loan_status': loan_status
    })
    
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'data', 'raw', 'loan_data.csv')
    df_loans = generate_loan_portfolio(n_samples=16000, random_state=42)
    df_loans.to_csv(output_path, index=False)
    print(f"Generated {len(df_loans)} loan records saved to: {output_path}")
    print(f"Portfolio Default Rate: {df_loans['loan_status'].mean()*100:.2f}%")
