"""
FinRisk-Engine: Streamlit Executive Underwriting & Credit Risk Portal
Interactive dashboard for loan underwriters, credit risk managers, and risk committees.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from risk_engine import CreditRiskEngine
from db import log_evaluation, get_recent_evaluations, get_portfolio_stats

st.set_page_config(
    page_title="FinRisk-Engine | Credit Risk Scoring Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .decision-badge-approved {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #86EFAC;
    }
    .decision-badge-review {
        background-color: #FEF9C3;
        color: #A16207;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #FDE047;
    }
    .decision-badge-rejected {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_risk_engine():
    try:
        return CreditRiskEngine()
    except Exception as e:
        return None

engine = load_risk_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("### **FinRisk Underwriting Engine**")
    st.markdown("Institutional Credit Risk & Basel II/III Default Scoring System")
    st.markdown("---")
    
    # Load Model Metrics if available
    metrics_path = os.path.join(base_dir, 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        best_metrics = metrics.get('Gradient Boosting Risk Engine', list(metrics.values())[0])
        st.markdown("**Production Model Telemetry:**")
        st.write(f"• **ROC-AUC:** `{best_metrics.get('roc_auc', 0.88):.4f}`")
        st.write(f"• **KS-Statistic:** `{best_metrics.get('ks_statistic', 0.62)*100:.1f}%`")
        st.write(f"• **Accuracy:** `{best_metrics.get('accuracy', 0.84)*100:.1f}%`")
        st.write(f"• **Brier Score:** `{best_metrics.get('brier_score', 0.08):.4f}`")
    
    st.markdown("---")
    st.markdown("**Quick Preset Profiles:**")
    preset = st.selectbox(
        "Select Demo Profile",
        ["Custom Input", "Prime Homebuyer", "Young Professional", "Overleveraged Applicant", "High-Risk Subprime"]
    )
    st.markdown("---")
    st.caption("Designed for Banking, FinTech, & Enterprise Risk Analytics.")

# Map Presets
preset_values = {
    "Custom Input": {"name": "Muhammad Irfan", "age": 28, "income": 72000, "home": "RENT", "emp": 4, "intent": "PERSONAL", "amnt": 15000, "rate": 8.5, "score": 710, "hist": 6, "default": "N"},
    "Prime Homebuyer": {"name": "Dato' Seri Rahman", "age": 45, "income": 180000, "home": "MORTGAGE", "emp": 15, "intent": "HOMEIMPROVEMENT", "amnt": 35000, "rate": 6.2, "score": 810, "hist": 20, "default": "N"},
    "Young Professional": {"name": "Sarah Tan", "age": 25, "income": 48000, "home": "RENT", "emp": 2, "intent": "EDUCATION", "amnt": 10000, "rate": 9.2, "score": 670, "hist": 4, "default": "N"},
    "Overleveraged Applicant": {"name": "Kevin Lee", "age": 36, "income": 45000, "home": "RENT", "emp": 3, "intent": "DEBTCONSOLIDATION", "amnt": 28000, "rate": 15.8, "score": 580, "hist": 7, "default": "N"},
    "High-Risk Subprime": {"name": "Azman Abdullah", "age": 42, "income": 32000, "home": "RENT", "emp": 1, "intent": "PERSONAL", "amnt": 20000, "rate": 21.0, "score": 460, "hist": 5, "default": "Y"}
}
current_preset = preset_values[preset]

# Main Title
st.markdown('<div class="main-header">💳 FinRisk-Engine: Enterprise Credit Risk Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Underwriting, Probability of Default (PD) Estimation, and Stress Testing</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 Live Underwriting Console", 
    "📈 Scenario & Sensitivity Testing", 
    "📊 Portfolio Intelligence & Basel Analytics", 
    "📋 Compliance Audit Log"
])

# ----------------- TAB 1: LIVE UNDERWRITING CONSOLE -----------------
with tabs[0]:
    if engine is None:
        st.warning("⚠️ Model not found on disk. Please run `python src/train_model.py` first to train and serialize the model.")
    else:
        st.markdown("#### **Applicant Financial Profile**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            applicant_name = st.text_input("Applicant Name", value=current_preset["name"])
            person_age = st.number_input("Age", min_value=18, max_value=85, value=current_preset["age"])
            person_income = st.number_input("Annual Gross Income (MYR/USD)", min_value=12000, max_value=1000000, value=current_preset["income"], step=1000)
            person_home_ownership = st.selectbox("Residential Status", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(current_preset["home"]))
            
        with col2:
            loan_amnt = st.number_input("Loan Amount Requested", min_value=1000, max_value=250000, value=current_preset["amnt"], step=500)
            loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(current_preset["intent"]))
            loan_int_rate = st.slider("Interest Rate (%)", min_value=4.0, max_value=28.0, value=float(current_preset["rate"]), step=0.1)
            person_emp_length = st.number_input("Employment Length (Years)", min_value=0, max_value=45, value=current_preset["emp"])
            
        with col3:
            credit_score = st.slider("Credit Bureau / FICO Score", min_value=300, max_value=850, value=current_preset["score"])
            cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=1, max_value=40, value=current_preset["hist"])
            cb_person_default_on_file = st.selectbox("Historical Default on Record?", ["N", "Y"], index=0 if current_preset["default"] == "N" else 1)
            underwriter_notes = st.text_input("Underwriter Notes", value="Standard online loan application evaluation.")

        # Determine loan grade proxy
        if credit_score >= 740 and loan_int_rate <= 8.5:
            loan_grade = 'A'
        elif credit_score >= 690:
            loan_grade = 'B'
        elif credit_score >= 640:
            loan_grade = 'C'
        elif credit_score >= 580:
            loan_grade = 'D'
        else:
            loan_grade = 'E'

        eval_payload = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': round(loan_amnt / person_income, 2),
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score
        }

        st.markdown("---")
        if st.button("🚀 Evaluate Credit Risk & Run Underwriting Model", use_container_width=True, type="primary"):
            evaluation = engine.evaluate_applicant(eval_payload)
            
            # Log to DB
            eval_id = log_evaluation(
                applicant_name=applicant_name,
                applicant_input=eval_payload,
                eval_output=evaluation,
                notes=underwriter_notes
            )
            
            st.markdown(f"### **Underwriting Assessment Result (Ref ID: #{eval_id})**")
            
            # Top Banner Decision
            col_dec, col_pd, col_dti, col_cap = st.columns(4)
            with col_dec:
                st.markdown("**Underwriting Decision:**")
                dec = evaluation['decision']
                if dec == "APPROVED":
                    st.markdown(f'<div class="decision-badge-approved">✅ {dec}</div>', unsafe_allow_html=True)
                elif dec == "MANUAL_REVIEW":
                    st.markdown(f'<div class="decision-badge-review">⚠️ {dec}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="decision-badge-rejected">❌ {dec}</div>', unsafe_allow_html=True)
                st.caption(evaluation['summary_statement'])
                
            with col_pd:
                st.metric("Probability of Default (PD)", evaluation['probability_of_default_pct'], delta=evaluation['risk_label'], delta_color="inverse")
                st.caption(f"Basel Rating: **{evaluation['risk_tier']}**")
                
            with col_dti:
                st.metric("Debt-to-Income (DTI)", evaluation['dti_ratio_pct'], delta=f"Payment: ${evaluation['monthly_payment']:,.0f}/mo")
                st.caption(f"Gross Income: ${evaluation['monthly_income']:,.0f}/mo")
                
            with col_cap:
                st.metric("Prudent Loan Ceiling", f"${evaluation['max_recommended_loan']:,.0f}")
                st.caption("Maximum allowable credit line at 40% DTI limit")

            # Risk Factors & Visual Gauges
            col_g, col_exp = st.columns([1, 1])
            with col_g:
                # Gauge Chart for PD
                pd_val = evaluation['probability_of_default'] * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pd_val,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Credit Risk Severity Gauge (PD %)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1E293B"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 15], 'color': '#86EFAC'},
                            {'range': [15, 35], 'color': '#FDE047'},
                            {'range': [35, 100], 'color': '#FCA5A5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 35.0
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_exp:
                st.markdown("**Underwriter Diagnostic Factors:**")
                st.markdown("🔴 **Key Risk Drivers:**")
                for rf in evaluation['risk_factors']:
                    st.write(f"- {rf}")
                    
                st.markdown("🟢 **Compensating Strengths:**")
                if evaluation['positive_factors']:
                    for pf in evaluation['positive_factors']:
                        st.write(f"- {pf}")
                else:
                    st.write("- No notable compensating strengths identified.")

# ----------------- TAB 2: SCENARIO STRESS-TESTING -----------------
with tabs[1]:
    st.markdown("#### **Macroeconomic & Financial Stress-Testing**")
    st.markdown("Simulate central bank rate hikes and borrower income shocks to analyze default probability sensitivity.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        base_income = st.number_input("Stress Test: Base Annual Income", min_value=20000, max_value=250000, value=60000, step=5000)
        base_score = st.slider("Stress Test: Credit Score", 350, 850, 680)
        base_loan = st.number_input("Stress Test: Loan Principal", 5000, 50000, 20000, step=2000)
    
    with col_s2:
        st.markdown("**Simulate Rate Shock Curve (+100 bps to +1,000 bps):**")
        rate_range = np.linspace(5.0, 25.0, 21)
        sim_pd = []
        
        for r in rate_range:
            payload = {
                'person_age': 32,
                'person_income': base_income,
                'person_home_ownership': 'RENT',
                'person_emp_length': 4,
                'loan_intent': 'PERSONAL',
                'loan_grade': 'B',
                'loan_amnt': base_loan,
                'loan_int_rate': r,
                'loan_percent_income': base_loan / base_income,
                'cb_person_default_on_file': 'N',
                'cb_person_cred_hist_length': 6,
                'credit_score': base_score
            }
            if engine:
                res = engine.evaluate_applicant(payload)
                sim_pd.append(res['probability_of_default'] * 100)
            else:
                sim_pd.append(0)
                
        df_sim = pd.DataFrame({'Interest Rate (%)': rate_range, 'Probability of Default (%)': sim_pd})
        fig_sim = px.line(
            df_sim, 
            x='Interest Rate (%)', 
            y='Probability of Default (%)',
            title="Default Risk Sensitivity to Interest Rate Shocks",
            markers=True,
            line_shape='spline'
        )
        fig_sim.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Policy Cutoff (35%)")
        st.plotly_chart(fig_sim, use_container_width=True)

# ----------------- TAB 3: PORTFOLIO INTELLIGENCE -----------------
with tabs[2]:
    st.markdown("#### **Loan Portfolio Distribution & Basel Analytics**")
    processed_path = os.path.join(base_dir, 'data', 'processed', 'loan_features.parquet')
    
    if os.path.exists(processed_path):
        df_port = pd.read_parquet(processed_path)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Loans Monitored", f"{len(df_port):,}")
        col_m2.metric("Portfolio Default Rate", f"{df_port['loan_status'].mean()*100:.2f}%")
        col_m3.metric("Average Credit Score", f"{df_port['credit_score'].mean():.0f}")
        col_m4.metric("Avg Debt-to-Income", f"{df_port['debt_to_income_ratio'].mean()*100:.1f}%")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_intent = px.histogram(
                df_port, 
                x="loan_intent", 
                color="loan_status", 
                barmode="group",
                title="Default Rate across Loan Purposes",
                color_discrete_map={0: "#3B82F6", 1: "#EF4444"}
            )
            st.plotly_chart(fig_intent, use_container_width=True)
            
        with col_c2:
            fig_score = px.box(
                df_port, 
                x="loan_grade", 
                y="credit_score", 
                color="loan_status",
                title="Credit Bureau Score Distribution by Loan Grade",
                color_discrete_map={0: "#10B981", 1: "#F87171"}
            )
            st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("Portfolio data will appear once `python src/train_model.py` finishes running.")

# ----------------- TAB 4: COMPLIANCE AUDIT LOG -----------------
with tabs[3]:
    st.markdown("#### **Institutional Underwriting Audit Trail (SQLite)**")
    st.markdown("Regulatory compliance logging for internal audit and Central Bank inspections.")
    
    audit_data = get_recent_evaluations(limit=100)
    if audit_data:
        df_audit = pd.DataFrame(audit_data)
        st.dataframe(
            df_audit[['id', 'timestamp', 'applicant_name', 'loan_amnt', 'credit_score', 'probability_of_default', 'risk_tier', 'decision', 'notes']],
            use_container_width=True
        )
        
        csv_data = df_audit.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit Trail (CSV)",
            data=csv_data,
            file_name="credit_underwriting_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.write("No loan evaluations logged in this session yet. Run an evaluation in Tab 1 to populate the audit log.")
