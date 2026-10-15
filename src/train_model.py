"""
FinRisk-Engine: Model Training & Validation Pipeline
Trains risk scoring models and evaluates using Basel II/III banking benchmarks:
ROC-AUC, PR-AUC, Brier Calibration Score, and Kolmogorov-Smirnov (KS) Statistic.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    roc_curve
)

from data_generator import generate_loan_portfolio
from feature_engineering import engineer_features, build_preprocessor

def calculate_ks_statistic(y_true, y_prob) -> float:
    """
    Computes Kolmogorov-Smirnov (KS) statistic:
    The maximum separation between Cumulative True Positive Rate and False Positive Rate.
    Industry standard used by central banks and credit rating agencies.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = float(np.max(tpr - fpr))
    return ks_stat

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'loan_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load or Generate Dataset
    if not os.path.exists(raw_data_path):
        print("Generating fresh banking portfolio dataset...")
        df = generate_loan_portfolio(n_samples=16000, random_state=42)
        df.to_csv(raw_data_path, index=False)
    else:
        df = pd.read_csv(raw_data_path)
        
    print(f"Dataset loaded: {len(df)} records. Base default rate: {df['loan_status'].mean()*100:.2f}%")
    
    # 2. Feature Engineering
    df_engineered = engineer_features(df)
    
    # Save processed features
    processed_path = os.path.join(base_dir, 'data', 'processed', 'loan_features.parquet')
    df_engineered.to_parquet(processed_path, index=False)
    
    # 3. Train-Test Split (80/20 Stratified)
    X = df_engineered.drop(columns=['loan_status'])
    y = df_engineered['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Model Architectures
    # Candidate 1: Basel-Calibrated Logistic Regression (High Interpretability)
    pipe_lr = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    # Candidate 2: Gradient Boosting Classifier (High Non-Linear Discriminative Power)
    pipe_gb = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.08, 
            max_depth=4, 
            subsample=0.85, 
            random_state=42
        ))
    ])
    
    models = {
        'Logistic Regression (Basel Baseline)': pipe_lr,
        'Gradient Boosting Risk Engine': pipe_gb
    }
    
    results = {}
    best_model = None
    best_auc = 0.0
    best_model_name = ""
    
    for name, pipeline in models.items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)
        
        # Predicted probabilities for default class (1)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.35).astype(int) # Conservative underwriting threshold (35% PD cutoff)
        
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))
        ks_stat = calculate_ks_statistic(y_test, y_prob)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        
        results[name] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'ks_statistic': ks_stat,
            'brier_score': brier,
            'accuracy': acc,
            'f1_score': f1
        }
        
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"KS-Statistic:  {ks_stat*100:.2f}% (Industry benchmark: >40%)")
        print(f"PR-AUC Score:  {pr_auc:.4f}")
        print(f"Brier Score:   {brier:.4f}")
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = pipeline
            best_model_name = name
            
    # 5. Persist the Winning Production Model
    model_save_path = os.path.join(models_dir, 'credit_risk_model.joblib')
    joblib.dump(best_model, model_save_path)
    print(f"\n[Production Deployment] Selected Champion Model: {best_model_name}")
    print(f"Model serialized successfully to: {model_save_path}")
    
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Metrics saved to: {metrics_path}")
    return results

if __name__ == '__main__':
    train_and_evaluate()
