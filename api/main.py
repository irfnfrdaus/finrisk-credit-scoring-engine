"""
FinRisk-Engine: FastAPI High-Performance Underwriting REST API
Exposes endpoints for automated loan scoring, risk tiering, batch portfolio screening,
and regulatory audit retrieval with Swagger / OpenAPI interactive documentation.
"""

import sys
import os
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src and database are in Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database'))

from risk_engine import CreditRiskEngine
from db import log_evaluation, get_recent_evaluations, get_portfolio_stats, init_db

# Initialize FastAPI App
app = FastAPI(
    title="FinRisk Credit Underwriting & Risk Scoring Engine",
    description="Automated Basel-compliant credit default scoring, PD calculation, and loan policy enforcement API.",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine & DB Setup
init_db()
risk_engine = CreditRiskEngine()

# Pydantic Schemas
class LoanApplicationRequest(BaseModel):
    applicant_name: str = Field(default="John Doe", example="Ahmad Zulkarnain")
    person_age: int = Field(ge=18, le=100, example=32, description="Applicant age in years")
    person_income: float = Field(gt=0, example=75000.0, description="Annual gross income in currency units")
    person_home_ownership: Literal['RENT', 'MORTGAGE', 'OWN', 'OTHER'] = Field(example="MORTGAGE")
    person_emp_length: float = Field(ge=0, le=60, example=6.0, description="Years in current employment")
    loan_intent: Literal['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'] = Field(example="DEBTCONSOLIDATION")
    loan_grade: Literal['A', 'B', 'C', 'D', 'E', 'F'] = Field(default="B", example="B")
    loan_amnt: float = Field(gt=0, example=15000.0, description="Requested principal loan amount")
    loan_int_rate: float = Field(ge=1.0, le=50.0, example=9.5, description="Annual interest rate in %")
    loan_percent_income: float = Field(ge=0.0, le=2.0, default=0.20, example=0.20)
    cb_person_default_on_file: Literal['Y', 'N'] = Field(example="N", description="Historical default record")
    cb_person_cred_hist_length: int = Field(ge=0, le=60, example=8, description="Credit history length in years")
    credit_score: int = Field(ge=300, le=850, example=710, description="Credit Bureau / FICO score (300-850)")
    notes: Optional[str] = Field(default="", example="Applicant applying for debt consolidation")

class ScoringResponse(BaseModel):
    evaluation_id: int
    applicant_name: str
    probability_of_default: float
    probability_of_default_pct: str
    risk_tier: str
    risk_label: str
    decision: str
    decision_color: str
    summary_statement: str
    dti_ratio_pct: str
    max_recommended_loan: float
    risk_factors: List[str]
    positive_factors: List[str]

@app.get("/", tags=["Overview"])
def root():
    return {
        "system": "FinRisk-Engine",
        "status": "ONLINE",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "model_loaded": True,
        "database_connected": True
    }

@app.post("/api/v1/score", response_model=ScoringResponse, tags=["Underwriting Engine"])
def score_loan_application(application: LoanApplicationRequest):
    """
    Evaluates a single credit application and returns the Probability of Default,
    Risk Rating Tier, Underwriting Decision, Affordability Ceiling, and Factor Explanations.
    """
    try:
        app_dict = application.model_dump()
        eval_result = risk_engine.evaluate_applicant(app_dict)
        
        # Log to audit database
        eval_id = log_evaluation(
            applicant_name=application.applicant_name,
            applicant_input=app_dict,
            eval_output=eval_result,
            notes=application.notes or ""
        )
        
        return {
            "evaluation_id": eval_id,
            "applicant_name": application.applicant_name,
            **eval_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-score", tags=["Underwriting Engine"])
def batch_score_applications(applications: List[LoanApplicationRequest]):
    """
    Evaluates multiple loan applications in batch and returns aggregated risk metrics.
    """
    results = []
    for app_req in applications:
        app_dict = app_req.model_dump()
        res = risk_engine.evaluate_applicant(app_dict)
        eval_id = log_evaluation(
            applicant_name=app_req.applicant_name,
            applicant_input=app_dict,
            eval_output=res,
            notes=app_req.notes or ""
        )
        results.append({
            "evaluation_id": eval_id,
            "applicant_name": app_req.applicant_name,
            **res
        })
    return {
        "batch_size": len(results),
        "results": results
    }

@app.get("/api/v1/audit-log", tags=["Regulatory Audit"])
def get_audit_trail(limit: int = Query(default=25, ge=1, le=200)):
    """
    Returns recent scored applications from the compliance database.
    """
    return get_recent_evaluations(limit=limit)

@app.get("/api/v1/portfolio-metrics", tags=["Portfolio Intelligence"])
def get_portfolio_metrics():
    """
    Returns aggregate underwriting metrics across the evaluation database.
    """
    return get_portfolio_stats()
