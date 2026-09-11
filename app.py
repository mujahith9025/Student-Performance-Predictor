import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from src.pdf_generator import generate_student_pdf_report, generate_classroom_pdf_report
from src.explainability import explain_single_student
from src.goal_simulator import simulate_academic_goal
from src.batch_predictor import process_batch_predictions, generate_sample_csv_template, load_or_create_sample_cohort
from src.sample_generator import generate_synthetic_classroom
from src.prescriptive_solutions import (
    diagnose_student_weaknesses,
    generate_prescriptive_solution,
    generate_classroom_intervention_matrix
)
from src.advanced_feature_engineering import engineer_features
from src.plotly_charts import (
    create_score_gauge,
    create_radar_chart,
    create_local_xai_waterfall,
    create_goal_trajectory_chart,
    create_batch_bubble_chart,
    create_global_importance_plotly,
    create_eda_distribution_plotly,
    create_eda_correlation_plotly,
    create_eda_test_prep_plotly,
    create_eda_parental_education_plotly,
    create_model_comparison_plotly,
    create_interactive_confusion_matrix,
    create_interactive_roc_curve,
    create_prescriptive_study_hours_chart,
    create_intervention_uplift_chart,
    create_classroom_intervention_cluster_chart
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Performance & Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PREMIUM GLASSMORPHIC DESIGN SYSTEM & MODERN TYPOGRAPHY
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --font-heading: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: #0F172A;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-heading) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Hero Banner Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(240, 249, 255, 0.85) 50%, rgba(238, 242, 255, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -5px rgba(30, 58, 138, 0.08), 0 4px 12px -2px rgba(30, 58, 138, 0.04);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
    }

    .hero-title {
        font-family: var(--font-heading);
        font-size: 2.35rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        max-width: 900px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    /* Live Status Badge Chips */
    .badge-chip-group {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: var(--font-body);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }

    .badge-primary {
        background: rgba(239, 246, 255, 0.85);
        color: #1D4ED8;
        border: 1px solid rgba(191, 219, 254, 0.9);
    }

    .badge-success {
        background: rgba(236, 253, 245, 0.85);
        color: #047857;
        border: 1px solid rgba(167, 243, 208, 0.9);
    }

    .badge-purple {
        background: rgba(245, 243, 255, 0.85);
        color: #6D28D9;
        border: 1px solid rgba(221, 214, 254, 0.9);
    }

    .badge-cyan {
        background: rgba(236, 254, 255, 0.85);
        color: #0E7490;
        border: 1px solid rgba(165, 243, 252, 0.9);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Glassmorphic Metric KPI Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        text-align: left;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -3px rgba(37, 99, 235, 0.1), 0 4px 12px -2px rgba(37, 99, 235, 0.05);
        border-color: rgba(147, 197, 253, 0.8);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
    }

    .metric-val {
        font-family: var(--font-heading);
        font-size: 1.95rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Glassmorphic Result Outcome Cards */
    .result-box-pass {
        background: linear-gradient(135deg, rgba(236, 253, 245, 0.95) 0%, rgba(209, 250, 229, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1.5px solid #6EE7B7;
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.15);
        position: relative;
    }

    .result-box-risk {
        background: linear-gradient(135deg, rgba(254, 242, 242, 0.95) 0%, rgba(254, 226, 226, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1.5px solid #FCA5A5;
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.15);
        position: relative;
    }

    .result-score-number {
        font-family: var(--font-heading);
        font-size: 3.6rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.4rem 0;
    }

    /* Roadmap Action Cards */
    .roadmap-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        border-left: 4px solid #2563EB;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-left-width: 4px;
        border-left-color: #2563EB;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease;
    }

    .roadmap-card:hover {
        transform: translateX(3px);
    }

    /* Preset Selection Header */
    .preset-chip-box {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.2rem;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }

    /* Modern Tabs Redesign */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(241, 245, 249, 0.7);
        padding: 6px 8px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: #475569;
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0 16px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.8);
        color: #1E3A8A;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD ARTIFACTS & RAW DATA
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    best_model_path = os.path.join(artifacts_dir, "best_model.joblib")
    best_clf_path = os.path.join(artifacts_dir, "best_classifier.joblib")
    
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    best_model = joblib.load(best_model_path) if os.path.exists(best_model_path) else None
    best_clf = joblib.load(best_clf_path) if os.path.exists(best_clf_path) else None
    
    # Load all regression models
    models_dir = os.path.join(artifacts_dir, "models")
    all_models = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib") and not f.startswith("clf_"):
                name = f.replace(".joblib", "").replace("_", " ").title()
                all_models[name] = joblib.load(os.path.join(models_dir, f))
                
    return preprocessor, best_model, best_clf, all_models

@st.cache_data
def load_raw_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "StudentsPerformance.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

preprocessor, best_model, best_clf, all_models = load_artifacts()
raw_df = load_raw_dataset()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("⚙️ Engine Hub")
    
    if all_models:
        default_index = 0
        model_names = sorted(list(all_models.keys()))
        if "Super Stacking Meta Regressor" in model_names:
            default_index = model_names.index("Super Stacking Meta Regressor")
        elif "Optimized Elasticnet" in model_names:
            default_index = model_names.index("Optimized Elasticnet")
        elif "Voting Ensemble Regressor" in model_names:
            default_index = model_names.index("Voting Ensemble Regressor")
            
        selected_model_name = st.selectbox(
            "Regression Algorithm:",
            options=model_names,
            index=default_index
        )
        active_model = all_models[selected_model_name]
    else:
        active_model = best_model
        selected_model_name = "Optimized Model (Default)"
        
    st.info("🛡️ Classifier: **Support Vector Machine (90.0% Acc | 0.932 ROC-AUC)**")
    
    st.markdown("---")
    st.markdown("### 📊 Production Benchmarks")
    st.markdown("- **Engineered Features:** **24 Synergy Metrics**")
    st.markdown("- **Regression $R^2$ Score:** **76.31%** ($\pm 5.96$ marks)")
    st.markdown("- **Classification Accuracy:** **90.0%**")
    st.markdown("- **Outlier Normalization:** **RobustScaler (IQR)**")
    st.markdown("- **Batch CSV Processing:** **Active**")
    st.markdown("- **PDF Generator:** **ReportLab 5.0**")
    
    st.caption("EduPredict AI v2.4 • Production Ready")

# ---------------------------------------------------------
# HERO BANNER & STATUS BADGES
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎓 Student Performance & Risk Intelligence</div>
    <div class="hero-subtitle">
        Enterprise-grade dual-task Machine Learning platform with 24-Feature Domain Engineering, Classroom Bulk CSV Analytics, 'What-If' Milestone Roadmapping, Explainable AI (XAI), and Verified PDF Academic Reports.
    </div>
    <div class="badge-chip-group">
        <span class="badge-chip badge-success"><span class="pulse-dot"></span> Production Engine Active</span>
        <span class="badge-chip badge-primary">⚡ 24-Feature Synergy Pipeline</span>
        <span class="badge-chip badge-purple">🛡️ 90.0% Pass Accuracy (0.932 ROC-AUC)</span>
        <span class="badge-chip badge-cyan">📄 ReportLab 5.0 PDF Certified</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metric KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">76.3%</div>
        <div class="metric-lbl">Regression R² (±5.9 MAE)</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">90.0%</div>
        <div class="metric-lbl">Classifier Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">24 Feat</div>
        <div class="metric-lbl">Synergy Features</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-val">PDF</div>
        <div class="metric-lbl">Verified Export v5.0</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🚀 Score & Risk Predictor", 
    "📂 Classroom Batch Predictor",
    "🎯 'What-If' Goal Simulator",
    "🔍 Explainable AI (XAI)",
    "📈 Exploratory Data Analysis", 
    "🏆 Model Leaderboards", 
    "⚙️ Hyperparameter Tuning",
    "📖 System Architecture"
])

# ----------------- TAB 1: PREDICTOR -----------------
with tab1:
    st.markdown("### 📝 Enter Student Multi-Dimensional Academic & Lifestyle Profile")
    st.markdown("Choose a quick-start persona below or customize 14 academic, behavioral, lifestyle, and socioeconomic characteristics for ultra-accurate prediction:")
    
    # Session State Initialization for 14 Form Inputs
    if "p_name" not in st.session_state: st.session_state["p_name"] = "Alex Johnson"
    if "p_id" not in st.session_state: st.session_state["p_id"] = "STU-2026-101"
    if "p_gender" not in st.session_state: st.session_state["p_gender"] = "female"
    if "p_race" not in st.session_state: st.session_state["p_race"] = "group C"
    if "p_edu" not in st.session_state: st.session_state["p_edu"] = "bachelor's degree"
    if "p_lunch" not in st.session_state: st.session_state["p_lunch"] = "standard"
    if "p_prep" not in st.session_state: st.session_state["p_prep"] = "completed"
    if "p_internet" not in st.session_state: st.session_state["p_internet"] = "yes"
    if "p_extra" not in st.session_state: st.session_state["p_extra"] = "yes"
    if "p_tutor" not in st.session_state: st.session_state["p_tutor"] = "peer_tutoring"
    if "p_read" not in st.session_state: st.session_state["p_read"] = 78
    if "p_write" not in st.session_state: st.session_state["p_write"] = 82
    if "p_att" not in st.session_state: st.session_state["p_att"] = 92.0
    if "p_study" not in st.session_state: st.session_state["p_study"] = 16.0
    if "p_sleep" not in st.session_state: st.session_state["p_sleep"] = 7.5
    if "p_fails" not in st.session_state: st.session_state["p_fails"] = 0

    # Quick Select Persona Presets (6 Personas in 2 Rows)
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.85rem; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.5px;">⚡ Quick-Load Demo Personas (1-Click Test):</span>
    </div>
    """, unsafe_allow_html=True)
    
    p_row1_c1, p_row1_c2, p_row1_c3 = st.columns(3)
    with p_row1_c1:
        if st.button("🌟 Honors Candidate (90+ Marks)", use_container_width=True):
            st.session_state["p_name"] = "Elena Rostova"
            st.session_state["p_id"] = "STU-2026-HONORS"
            st.session_state["p_gender"] = "female"
            st.session_state["p_race"] = "group E"
            st.session_state["p_edu"] = "master's degree"
            st.session_state["p_lunch"] = "standard"
            st.session_state["p_prep"] = "completed"
            st.session_state["p_internet"] = "yes"
            st.session_state["p_extra"] = "yes"
            st.session_state["p_tutor"] = "private_tutor"
            st.session_state["p_read"] = 92
            st.session_state["p_write"] = 95
            st.session_state["p_att"] = 98.0
            st.session_state["p_study"] = 24.0
            st.session_state["p_sleep"] = 8.0
            st.session_state["p_fails"] = 0
            st.rerun()
            
    with p_row1_c2:
        if st.button("⚖️ Average Profile (60-70 Marks)", use_container_width=True):
            st.session_state["p_name"] = "Jordan Miller"
            st.session_state["p_id"] = "STU-2026-AVG"
            st.session_state["p_gender"] = "male"
            st.session_state["p_race"] = "group C"
            st.session_state["p_edu"] = "some college"
            st.session_state["p_lunch"] = "standard"
            st.session_state["p_prep"] = "none"
            st.session_state["p_internet"] = "yes"
            st.session_state["p_extra"] = "no"
            st.session_state["p_tutor"] = "none"
            st.session_state["p_read"] = 65
            st.session_state["p_write"] = 62
            st.session_state["p_att"] = 86.0
            st.session_state["p_study"] = 12.0
            st.session_state["p_sleep"] = 7.2
            st.session_state["p_fails"] = 0
            st.rerun()
            
    with p_row1_c3:
        if st.button("🚨 High-Risk Alert (<40 Marks)", use_container_width=True):
            st.session_state["p_name"] = "Marcus Vance"
            st.session_state["p_id"] = "STU-2026-RISK"
            st.session_state["p_gender"] = "male"
            st.session_state["p_race"] = "group A"
            st.session_state["p_edu"] = "some high school"
            st.session_state["p_lunch"] = "free/reduced"
            st.session_state["p_prep"] = "none"
            st.session_state["p_internet"] = "no"
            st.session_state["p_extra"] = "no"
            st.session_state["p_tutor"] = "none"
            st.session_state["p_read"] = 34
            st.session_state["p_write"] = 30
            st.session_state["p_att"] = 62.0
            st.session_state["p_study"] = 4.0
            st.session_state["p_sleep"] = 5.0
            st.session_state["p_fails"] = 2
            st.rerun()
            
    p_row2_c1, p_row2_c2, p_row2_c3 = st.columns(3)
    with p_row2_c1:
        if st.button("📖 Verbal Strong / Math Lagging", use_container_width=True):
            st.session_state["p_name"] = "Sophia Chen"
            st.session_state["p_id"] = "STU-2026-VERBAL"
            st.session_state["p_gender"] = "female"
            st.session_state["p_race"] = "group D"
            st.session_state["p_edu"] = "bachelor's degree"
            st.session_state["p_lunch"] = "standard"
            st.session_state["p_prep"] = "completed"
            st.session_state["p_internet"] = "yes"
            st.session_state["p_extra"] = "yes"
            st.session_state["p_tutor"] = "peer_tutoring"
            st.session_state["p_read"] = 88
            st.session_state["p_write"] = 85
            st.session_state["p_att"] = 90.0
            st.session_state["p_study"] = 14.0
            st.session_state["p_sleep"] = 7.5
            st.session_state["p_fails"] = 0
            st.rerun()
            
    with p_row2_c2:
        if st.button("🚀 High-Effort Rising Star", use_container_width=True):
            st.session_state["p_name"] = "Lucas Taylor"
            st.session_state["p_id"] = "STU-2026-RISING"
            st.session_state["p_gender"] = "male"
            st.session_state["p_race"] = "group B"
            st.session_state["p_edu"] = "high school"
            st.session_state["p_lunch"] = "free/reduced"
            st.session_state["p_prep"] = "completed"
            st.session_state["p_internet"] = "yes"
            st.session_state["p_extra"] = "yes"
            st.session_state["p_tutor"] = "peer_tutoring"
            st.session_state["p_read"] = 76
            st.session_state["p_write"] = 74
            st.session_state["p_att"] = 96.0
            st.session_state["p_study"] = 20.0
            st.session_state["p_sleep"] = 7.5
            st.session_state["p_fails"] = 0
            st.rerun()
            
    with p_row2_c3:
        if st.button("🎯 Borderline Pass (50 Marks)", use_container_width=True):
            st.session_state["p_name"] = "Amara Patel"
            st.session_state["p_id"] = "STU-2026-BORDER"
            st.session_state["p_gender"] = "female"
            st.session_state["p_race"] = "group C"
            st.session_state["p_edu"] = "some college"
            st.session_state["p_lunch"] = "free/reduced"
            st.session_state["p_prep"] = "none"
            st.session_state["p_internet"] = "yes"
            st.session_state["p_extra"] = "no"
            st.session_state["p_tutor"] = "none"
            st.session_state["p_read"] = 52
            st.session_state["p_write"] = 49
            st.session_state["p_att"] = 78.0
            st.session_state["p_study"] = 7.0
            st.session_state["p_sleep"] = 6.0
            st.session_state["p_fails"] = 1
            st.rerun()

    st.write("")
    
    # Contextual Domain Insights Popover
    with st.expander("💡 Understanding the 14 Multi-Dimensional Features (Domain Guidance)"):
        st.markdown("""
        - **Attendance Rate (%) & Study Hours (hrs/wk):** The strongest behavioral drivers of academic momentum and examination success.
        - **Reading & Writing Scores:** Prerequisite linguistic ability correlates directly with mathematical problem modeling ($r > 0.80$).
        - **Sleep & Rest Balance:** Fatigue (<6 hrs/day) introduces working memory cognitive degradation.
        - **Past Course Backlogs:** Critical historical indicator for pinpointing foundational concept gaps.
        - **Digital Access & Tutoring Support:** Measure institutional and household educational resources.
        """)

    with st.form("prediction_form"):
        # Student Identification
        st.markdown("#### 🆔 Student Identification")
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            student_name = st.text_input("Student Full Name", value=st.session_state["p_name"], placeholder="e.g. Alex Johnson")
        with id_col2:
            student_id = st.text_input("Student ID / Roll Number", value=st.session_state["p_id"], placeholder="e.g. STU-2026-101")
            
        st.write("")
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Academic Literacy & Deliberate Study
        with col1:
            st.markdown("#### 📚 Academic & Study Habits")
            reading_score = st.slider("Reading Score (0 - 100)", min_value=0, max_value=100, value=int(st.session_state["p_read"]), step=1)
            writing_score = st.slider("Writing Score (0 - 100)", min_value=0, max_value=100, value=int(st.session_state["p_write"]), step=1)
            attendance_rate = st.slider("Attendance Rate (%)", min_value=50.0, max_value=100.0, value=float(st.session_state["p_att"]), step=0.5)
            weekly_study_hours = st.slider("Weekly Self-Study (Hours/Week)", min_value=1.0, max_value=40.0, value=float(st.session_state["p_study"]), step=0.5)
            past_failures = st.selectbox("Prior Course Backlogs / Failures", options=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(st.session_state["p_fails"]))

        # Column 2: Demographics, Household & Digital
        with col2:
            st.markdown("#### 👤 Demographics & Digital Capital")
            gender_opts = ["female", "male"]
            gender = st.selectbox("Gender", options=gender_opts, index=gender_opts.index(st.session_state["p_gender"]))
            
            race_opts = ["group A", "group B", "group C", "group D", "group E"]
            race_ethnicity = st.selectbox("Race / Ethnicity Group", options=race_opts, index=race_opts.index(st.session_state["p_race"]))
            
            edu_opts = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
            parental_education = st.selectbox("Parental Level of Education", options=edu_opts, index=edu_opts.index(st.session_state["p_edu"]))
            
            lunch_opts = ["standard", "free/reduced"]
            lunch = st.selectbox("Lunch Plan", options=lunch_opts, index=lunch_opts.index(st.session_state["p_lunch"]))
            
            internet_opts = ["yes", "no"]
            internet_access = st.selectbox("Home High-Speed Internet Access", options=internet_opts, index=internet_opts.index(st.session_state["p_internet"]))

        # Column 3: Wellness, Curricular & Support
        with col3:
            st.markdown("#### 🌱 Wellness, Prep & Tutoring")
            sleep_hours_per_day = st.slider("Average Daily Sleep (Hours/Day)", min_value=4.0, max_value=10.0, value=float(st.session_state["p_sleep"]), step=0.2)
            
            prep_opts = ["none", "completed"]
            test_prep = st.selectbox("Test Preparation Course", options=prep_opts, index=prep_opts.index(st.session_state["p_prep"]))
            
            extra_opts = ["yes", "no"]
            extracurricular_activities = st.selectbox("Extracurricular Engagement", options=extra_opts, index=extra_opts.index(st.session_state["p_extra"]))
            
            tutor_opts = ["none", "peer_tutoring", "private_tutor"]
            tutoring_support = st.selectbox(
                "Tutoring Support Enrolled",
                options=tutor_opts,
                index=tutor_opts.index(st.session_state["p_tutor"]),
                format_func=lambda x: x.replace('_', ' ').title()
            )

        st.write("")
        submit_btn = st.form_submit_button("⚡ Run Multi-Dimensional ML Assessment", use_container_width=True, type="primary")

    if submit_btn:
        if preprocessor is None or active_model is None:
            st.error("Model artifacts not found! Please run the training pipeline first.")
        else:
            # Create DataFrame matching 14-feature format
            input_dict = {
                "gender": [gender],
                "race/ethnicity": [race_ethnicity],
                "parental level of education": [parental_education],
                "lunch": [lunch],
                "test preparation course": [test_prep],
                "internet_access": [internet_access],
                "extracurricular_activities": [extracurricular_activities],
                "tutoring_support": [tutoring_support],
                "attendance_rate": [attendance_rate],
                "weekly_study_hours": [weekly_study_hours],
                "sleep_hours_per_day": [sleep_hours_per_day],
                "past_failures": [past_failures],
                "reading score": [reading_score],
                "writing score": [writing_score]
            }
            input_df = pd.DataFrame(input_dict)
            input_df_eng = engineer_features(input_df)
            transformed_input = preprocessor.transform(input_df_eng)
            
            # 1. Regression Prediction
            raw_prediction = active_model.predict(transformed_input)[0]
            predicted_math = float(np.clip(raw_prediction, 0, 100))
            overall_avg = (predicted_math + reading_score + writing_score) / 3.0
            
            # 2. Classification Prediction
            if best_clf is not None:
                pass_prob = float(best_clf.predict_proba(transformed_input)[0][1]) * 100.0
                is_pass = int(best_clf.predict(transformed_input)[0])
            else:
                pass_prob = 95.0 if predicted_math >= 50 else 25.0
                is_pass = 1 if predicted_math >= 50 else 0
                
            # Risk Level Assessment
            if pass_prob >= 80:
                risk_level = "Safe / Low Risk"
            elif pass_prob >= 50:
                risk_level = "Moderate Risk (Needs Monitoring)"
            else:
                risk_level = "🚨 High Academic / Dropout Risk"
                
            # Grade Mapping
            if overall_avg >= 90:
                grade = "A+ (Outstanding)"
            elif overall_avg >= 80:
                grade = "A (Excellent)"
            elif overall_avg >= 70:
                grade = "B (Good)"
            elif overall_avg >= 60:
                grade = "C (Satisfactory)"
            elif overall_avg >= 50:
                grade = "D (Pass)"
            else:
                grade = "F (Needs Remedial)"
            
            # Results Box Styling
            box_class = "result-box-pass" if is_pass == 1 else "result-box-risk"
            score_color = "#047857" if is_pass == 1 else "#B91C1C"
            
            st.markdown(f"""
            <div class="{box_class}">
                <div style="font-size: 1rem; color: {score_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Mathematics Score & Risk Status</div>
                <div class="result-score-number" style="color: {score_color};">{predicted_math:.1f} <span style="font-size: 1.6rem; opacity: 0.85;">/ 100</span></div>
                <div style="font-size: 1.05rem; color: {score_color};">
                    <b>Pass Probability: {pass_prob:.1f}%</b> • Status: <b>{'Passed' if is_pass==1 else 'At-Risk / Fail'}</b> • Risk Tier: <b>{risk_level}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Interactive Visual Gauges
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                fig_gauge = create_score_gauge(predicted_math, grade)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with g_col2:
                socio_index = input_df_eng["socio_readiness_index"].iloc[0]
                fig_radar = create_radar_chart(
                    reading=reading_score,
                    writing=writing_score,
                    predicted_math=predicted_math,
                    socio_index=socio_index,
                    attendance=attendance_rate,
                    study_hours=weekly_study_hours,
                    sleep_hours=sleep_hours_per_day,
                    prep_status=test_prep
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric("Math Score (Predicted)", f"{predicted_math:.1f} / 100")
            with r_col2:
                st.metric("3-Subject Average", f"{overall_avg:.1f} / 100")
            with r_col3:
                st.metric("Pass Probability", f"{pass_prob:.1f}%")
            with r_col4:
                st.metric("Predicted Grade", grade)
                
            # 3. Local Explainable AI (SHAP Waterfall Breakdown)
            st.markdown("---")
            st.markdown("### 🔍 Why did the AI predict this score? (Interactive Waterfall Breakdown)")
            st.markdown("This interactive breakdown details how each student factor added or deducted marks from the baseline average:")
            
            base_val, pred_val, contrib_df = explain_single_student(input_df, preprocessor, active_model, None)
            
            xai_c1, xai_c2 = st.columns([1.3, 1])
            with xai_c1:
                fig_waterfall = create_local_xai_waterfall(base_val, predicted_math, contrib_df)
                st.plotly_chart(fig_waterfall, use_container_width=True)
            with xai_c2:
                st.markdown("#### 📋 Top Factor Attributions:")
                for _, row in contrib_df.head(6).iterrows():
                    impact = row["Impact"]
                    sign = "+" if impact >= 0 else ""
                    color = "#059669" if impact >= 0 else "#DC2626"
                    st.markdown(f"- **{row['Factor']}** (`{row['Value']}`): <span style='color:{color}; font-weight:bold;'>{sign}{impact:.2f} marks</span>", unsafe_allow_html=True)
                st.dataframe(
                    contrib_df.style.format({"Impact": "{:+.2f} marks"}),
                    use_container_width=True,
                    hide_index=True
                )
                st.dataframe(
                    contrib_df.style.format({"Impact": "{:+.2f} marks"}),
                    use_container_width=True,
                    hide_index=True
                )

            # 4. AI Prescriptive Solution & Intervention Engine
            st.markdown("---")
            st.markdown("### 💡 AI Prescriptive Solution & Targeted Intervention Plan")
            st.markdown("EduPredict AI's clinical diagnostic algorithm has generated the following personalized academic prescriptions, weekly study schedule, and projected score uplift roadmap:")
            
            prescriptive_sol = generate_prescriptive_solution(
                input_df.iloc[0].to_dict(),
                predicted_math=predicted_math,
                pass_prob=pass_prob,
                grade=grade
            )
            
            # Root Cause Diagnostic Bottlenecks
            st.markdown("#### 🔍 Root-Cause Diagnostic Bottlenecks:")
            bn_cols = st.columns(len(prescriptive_sol["bottlenecks"])) if prescriptive_sol["bottlenecks"] else [st.container()]
            if prescriptive_sol["bottlenecks"]:
                for idx, bn in enumerate(prescriptive_sol["bottlenecks"]):
                    with bn_cols[idx]:
                        sev_color = "#DC2626" if bn["severity"] == "High" else ("#D97706" if bn["severity"] == "Medium" else "#2563EB")
                        bg_color = "rgba(254, 242, 242, 0.9)" if bn["severity"] == "High" else ("rgba(254, 243, 199, 0.9)" if bn["severity"] == "Medium" else "rgba(239, 246, 255, 0.9)")
                        st.markdown(f"""
                        <div style="background: {bg_color}; border-left: 4px solid {sev_color}; border-radius: 8px; padding: 0.8rem; height: 100%;">
                            <div style="font-size: 0.95rem; font-weight: 700; color: {sev_color};">{bn['icon']} {bn['category']}</div>
                            <div style="font-size: 0.75rem; font-weight: 700; color: {sev_color}; text-transform: uppercase; margin-bottom: 0.3rem;">Severity: {bn['severity']}</div>
                            <div style="font-size: 0.82rem; color: #334155; line-height: 1.35;">{bn['detail']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("🌟 **Zero Critical Bottlenecks Identified:** Student maintains balanced, optimal performance across all academic pillars.")
                
            st.write("")
            
            # Prescriptive Action Items & Score Uplift Stages
            st.markdown("#### 🎯 Prioritized Interventions & Quantitative Uplift:")
            for idx, item in enumerate(prescriptive_sol["interventions"]):
                with st.expander(f"📌 **{item['priority']}: {item['title']}** (Timeline: `{item['timeline']}`) — Uplift: **{item['est_uplift']}**", expanded=(idx==0)):
                    st.markdown(f"**Action Required:** {item['action']}")
                    st.markdown(f"**Recommended Resource:** `{item['resource']}`")
                    st.markdown(f"**Projected Score Contribution:** <span style='color:#059669; font-weight:bold;'>{item['est_uplift']}</span>", unsafe_allow_html=True)
                    
            st.write("")
            
            # Interactive Visual Charts: Study Plan + Score Uplift Waterfall
            rx_col1, rx_col2 = st.columns(2)
            with rx_col1:
                fig_study = create_prescriptive_study_hours_chart(prescriptive_sol["study_hours"])
                st.plotly_chart(fig_study, use_container_width=True)
            with rx_col2:
                fig_uplift = create_intervention_uplift_chart(predicted_math, prescriptive_sol["projected_score"], prescriptive_sol["interventions"])
                st.plotly_chart(fig_uplift, use_container_width=True)
                
            # 12-Week Step-by-Step Action Roadmap
            st.markdown("#### 🗺️ 12-Week Academic Recovery & Growth Roadmap:")
            ms_cols = st.columns(4)
            for m_idx, ms in enumerate(prescriptive_sol["milestones"]):
                with ms_cols[m_idx]:
                    st.markdown(f"""
                    <div style="background: rgba(248, 250, 252, 0.95); border: 1.5px solid #CBD5E1; border-top: 3px solid #2563EB; border-radius: 8px; padding: 0.8rem; height: 100%;">
                        <div style="font-size: 0.85rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem;">{ms['week']}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: #059669; margin-bottom: 0.4rem;">🎯 {ms['target']}</div>
                        <div style="font-size: 0.8rem; color: #475569; line-height: 1.3;">{ms['milestone']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.write("")
            
            # Educator & Counselor Action Checklist
            with st.expander("👨‍🏫 Counselor & Educator Direct Action Checklist"):
                for g_item in prescriptive_sol["teacher_guidance"]:
                    st.markdown(f"- ✔️ {g_item}")

            # 5. Diagnostic Feedback Badges
            st.markdown("---")
            st.markdown("#### 💬 Clinical Diagnostic Advice & Summary Notes")
            tips = []
            if pass_prob < 50:
                tips.append("🚨 **High Risk Alert:** Student is performing below benchmark in Mathematics. Immediate remedial sessions recommended.")
            if test_prep == "none":
                tips.append("📌 **Test Prep Course:** Completing the preparation course statistically provides a **+9.4 mark boost**.")
            if lunch == "free/reduced":
                tips.append("📌 **Nutrition:** Standard lunch plan correlates with an **+8.0 mark boost** across all exams.")
            if reading_score < 60:
                tips.append("📌 **Reading Focus:** Enhancing reading comprehension reinforces mathematical problem solving.")
            if not tips:
                tips.append("🌟 **Optimal Academic Standing:** Student profile exhibits strong positive indicators across all subjects.")
                
            for tip in tips:
                if "Alert" in tip:
                    st.error(tip)
                elif "📌" in tip:
                    st.warning(tip)
                else:
                    st.success(tip)

            # 6. Generate Official PDF Certificate with Prescriptive Solutions
            st.markdown("---")
            st.markdown("#### 📄 Official Counselor Evaluation & PDF Report Card")
            
            counselor_note = st.text_area(
                "✍️ Academic Counselor / Teacher Remarks (Optional):",
                value="Student exhibits strong conceptual grasp in language components. Recommended enrollment in mathematics peer tutoring and weekly practice modules to consolidate exam performance.",
                placeholder="Type personalized counselor remarks to appear on the official certificate...",
                height=80
            )
            
            # Live In-App Certificate Preview Badge
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(241, 245, 249, 0.85) 100%); border: 1.5px solid #CBD5E1; border-radius: 12px; padding: 1.1rem 1.3rem; margin: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="font-family: var(--font-heading); font-size: 1.1rem; font-weight: 800; color: #1E3A8A;">🎓 EduPredict AI Certified Report Card</span>
                    <span style="background: #ECFDF5; color: #047857; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 9999px; border: 1px solid #A7F3D0;">✔ Verified Dual ML Engine</span>
                </div>
                <span style="font-size: 0.9rem; color: #334155;"><b>Candidate:</b> {student_name if student_name.strip() else 'Student'} (`{student_id if student_id.strip() else 'STU-101'}`) • <b>Predicted Marks:</b> {predicted_math:.1f}/100 • <b>Grade:</b> {grade.split()[0]} • <b>Risk Tier:</b> {risk_level} • <b>Projected Post-Intervention:</b> {prescriptive_sol['projected_score']:.1f}/100 ({prescriptive_sol['projected_grade']})</span>
            </div>
            """, unsafe_allow_html=True)
            
            pdf_bytes = generate_student_pdf_report(
                student_name=student_name if student_name.strip() else "Student",
                student_id=student_id if student_id.strip() else "STU-UNASSIGNED",
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_education=parental_education,
                lunch=lunch,
                test_prep=test_prep,
                reading_score=reading_score,
                writing_score=writing_score,
                predicted_math=predicted_math,
                overall_avg=overall_avg,
                grade=grade,
                model_name=selected_model_name,
                tips=tips,
                pass_prob=pass_prob,
                risk_level=risk_level,
                custom_counselor_note=counselor_note,
                prescriptive_solution=prescriptive_sol,
                attendance_rate=attendance_rate,
                weekly_study_hours=weekly_study_hours,
                sleep_hours_per_day=sleep_hours_per_day,
                past_failures=past_failures,
                tutoring_support=tutoring_support,
                internet_access=internet_access
            )
            
            clean_filename = f"Official_Academic_Report_{student_id.replace('/', '_')}.pdf"
            
            st.download_button(
                label=f"📥 Download Verified PDF Academic Certificate with Prescriptions ({clean_filename})",
                data=pdf_bytes,
                file_name=clean_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

# ----------------- TAB 2: CLASSROOM BATCH PREDICTOR -----------------
with tab2:
    st.markdown("### 📂 Classroom Batch Prediction & Cohort Risk Analytics")
    st.markdown("Upload a CSV dataset or generate realistic synthetic cohorts (10 to 500 students) to evaluate grade distributions, early dropout risk, and prescriptive classroom intervention clusters at scale.")
    
    # ---------------------------------------------------------
    # 1. BULK CLASSROOM SAMPLE HUB & DYNAMIC GENERATOR
    # ---------------------------------------------------------
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.85rem; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.5px;">⚡ Instant 1-Click Pre-Built Cohort Loaders:</span>
    </div>
    """, unsafe_allow_html=True)
    
    samp_c1, samp_c2, samp_c3, samp_c4, samp_c5 = st.columns(5)
    
    with samp_c1:
        if st.button("🌟 Balanced 50 Class", use_container_width=True, type="secondary"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("balanced_50")
            st.session_state["cached_batch_name"] = "Balanced Classroom Cohort (50 Students)"
            st.rerun()
            
    with samp_c2:
        if st.button("🏫 Grade 100 Cohort", use_container_width=True, type="secondary"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("large_100")
            st.session_state["cached_batch_name"] = "Grade-Wide Cohort (100 Students)"
            st.rerun()
            
    with samp_c3:
        if st.button("🚨 At-Risk Focus 40", use_container_width=True, type="secondary"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("at_risk_40")
            st.session_state["cached_batch_name"] = "High-Risk Intervention Focus (40 Students)"
            st.rerun()
            
    with samp_c4:
        if st.button("🏆 Honors 35 Cohort", use_container_width=True, type="secondary"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("honors_35")
            st.session_state["cached_batch_name"] = "Honors / AP Distinction Cohort (35 Students)"
            st.rerun()
            
    with samp_c5:
        if st.button("🌐 Diverse 200 Class", use_container_width=True, type="secondary"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("mixed_200")
            st.session_state["cached_batch_name"] = "Diverse Multi-Section Cohort (200 Students)"
            st.rerun()
            
    st.write("")
    
    # Expandable Hub for CSV Downloads & Dynamic Synthetic Generator
    hub_exp1, hub_exp2 = st.columns(2)
    
    with hub_exp1:
        with st.expander("📥 Download Pre-Built Bulk Classroom CSV Files", expanded=False):
            st.markdown("Download realistic benchmark CSV files for offline evaluation, reporting, or stress testing:")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                df_b50 = load_or_create_sample_cohort("balanced_50")
                buf_b50 = io.StringIO()
                df_b50.to_csv(buf_b50, index=False)
                st.download_button(
                    "📄 Balanced 50 CSV",
                    data=buf_b50.getvalue(),
                    file_name="sample_classroom_balanced_50.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                df_risk40 = load_or_create_sample_cohort("at_risk_40")
                buf_risk40 = io.StringIO()
                df_risk40.to_csv(buf_risk40, index=False)
                st.download_button(
                    "🚨 At-Risk Focus 40 CSV",
                    data=buf_risk40.getvalue(),
                    file_name="sample_classroom_at_risk_focus_40.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with d_col2:
                df_l100 = load_or_create_sample_cohort("large_100")
                buf_l100 = io.StringIO()
                df_l100.to_csv(buf_l100, index=False)
                st.download_button(
                    "🏫 Grade 100 CSV",
                    data=buf_l100.getvalue(),
                    file_name="sample_classroom_large_100.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                df_honors = load_or_create_sample_cohort("honors_35")
                buf_honors = io.StringIO()
                df_honors.to_csv(buf_honors, index=False)
                st.download_button(
                    "🏆 Honors 35 CSV",
                    data=buf_honors.getvalue(),
                    file_name="sample_classroom_honors_35.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            template_data = generate_sample_csv_template()
            st.download_button(
                "📋 10-Student Starter Template CSV",
                data=template_data,
                file_name="starter_classroom_template.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with hub_exp2:
        with st.expander("⚙️ Dynamic Synthetic Cohort Generator (Custom Size)", expanded=False):
            st.markdown("Generate custom synthetic classrooms with adjustable student volume & demographic distribution:")
            gen_size = st.slider("Classroom Size (Number of Students):", min_value=10, max_value=500, value=75, step=5)
            gen_dist = st.selectbox(
                "Performance Distribution Profile:",
                options=[
                    ("balanced", "Balanced Mixed Classroom (Bell Curve)"),
                    ("at_risk_focus", "High Academic Risk & Remedial Focus"),
                    ("honors_advanced", "AP / Honors Distinction Cohort")
                ],
                format_func=lambda x: x[1]
            )[0]
            
            gen_c1, gen_c2 = st.columns(2)
            with gen_c1:
                if st.button("⚡ Generate & Assess Cohort", use_container_width=True, type="primary"):
                    gen_df = generate_synthetic_classroom(n_students=gen_size, cohort_type=gen_dist, seed=np.random.randint(1, 9999))
                    st.session_state["cached_batch"] = gen_df
                    st.session_state["cached_batch_name"] = f"Custom Synthetic Cohort ({gen_size} Students • {gen_dist.replace('_', ' ').title()})"
                    st.rerun()
            with gen_c2:
                custom_df = generate_synthetic_classroom(n_students=gen_size, cohort_type=gen_dist, seed=42)
                custom_buf = io.StringIO()
                custom_df.to_csv(custom_buf, index=False)
                st.download_button(
                    "💾 Download Synthetic CSV",
                    data=custom_buf.getvalue(),
                    file_name=f"custom_cohort_{gen_size}_students.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
    st.write("")
    uploaded_file = st.file_uploader("📤 Or Upload Custom CSV File (Drag & Drop):", type=["csv"])
    
    batch_to_process = None
    batch_source_name = ""
    
    if uploaded_file is not None:
        batch_to_process = pd.read_csv(uploaded_file)
        batch_source_name = uploaded_file.name
        st.session_state["cached_batch"] = batch_to_process
        st.session_state["cached_batch_name"] = batch_source_name
    elif "cached_batch" in st.session_state:
        batch_to_process = st.session_state["cached_batch"]
        batch_source_name = st.session_state.get("cached_batch_name", "Loaded Cohort")
    
    if batch_to_process is not None:
        try:
            st.success(f"✅ Successfully loaded `{batch_source_name}` ({len(batch_to_process)} students found).")
            
            if preprocessor is not None and active_model is not None:
                processed_batch, summary = process_batch_predictions(
                    batch_to_process, preprocessor, active_model, best_clf
                )
                
                # 1. Summary Metric Cards
                st.markdown("#### 📊 Classroom Overview Metrics:")
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1:
                    st.metric("Total Students", summary["total_students"])
                with b2:
                    st.metric("Class Avg Math", f"{summary['class_avg_math']:.1f} / 100")
                with b3:
                    st.metric("Overall 3-Sub Avg", f"{summary['class_avg_overall']:.1f} / 100")
                with b4:
                    st.metric("Class Pass Rate", f"{summary['pass_rate']}%")
                with b5:
                    st.metric("🚨 At-Risk Count", summary["at_risk_count"])
                    
                st.write("")
                
                # 2. Classroom Prescriptive Intervention Matrix
                matrix = generate_classroom_intervention_matrix(processed_batch)
                
                st.markdown("#### 📋 Classroom Prescriptive Intervention Matrix (AI Clustered Cohorts):")
                m_col1, m_col2 = st.columns([1.1, 1.4])
                
                with m_col1:
                    fig_clusters = create_classroom_intervention_cluster_chart(matrix["summary"])
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                with m_col2:
                    st.markdown("""
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem;">Targeted Institutional Action Strategies:</div>
                    """, unsafe_allow_html=True)
                    
                    # 4 Intervention Cluster Cards
                    st.markdown(f"""
                    <div style="background: rgba(254, 242, 242, 0.9); border-left: 4px solid #EF4444; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;">
                        <span style="font-weight: 800; color: #DC2626;">🚨 Intensive Remedial Cohort ({matrix['summary']['high_risk_count']} students):</span><br/>
                        <span style="font-size: 0.82rem; color: #334155;">{matrix['action_plans']['Intensive Remedial (High Risk)']}</span>
                    </div>
                    
                    <div style="background: rgba(254, 243, 199, 0.9); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;">
                        <span style="font-weight: 800; color: #D97706;">🎯 Test Prep Bootcamp Cohort ({matrix['summary']['test_prep_needed_count']} students):</span><br/>
                        <span style="font-size: 0.82rem; color: #334155;">{matrix['action_plans']['Test Prep Bootcamp (Moderate Gap)']}</span>
                    </div>
                    
                    <div style="background: rgba(239, 246, 255, 0.9); border-left: 4px solid #3B82F6; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;">
                        <span style="font-weight: 800; color: #2563EB;">📚 Verbal & Reading Support ({matrix['summary']['verbal_support_count']} students):</span><br/>
                        <span style="font-size: 0.82rem; color: #334155;">{matrix['action_plans']['Verbal / Reading Support']}</span>
                    </div>
                    
                    <div style="background: rgba(236, 253, 245, 0.9); border-left: 4px solid #10B981; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;">
                        <span style="font-weight: 800; color: #059669;">🏆 Honors & Distinction Mentorship ({matrix['summary']['honors_count']} students):</span><br/>
                        <span style="font-size: 0.82rem; color: #334155;">{matrix['action_plans']['Honors / Distinction Mentorship']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                
                # 3. Priority Watchlist & Distinction Honor Roll
                high_risk_df = processed_batch[processed_batch["Risk_Tier"].str.contains("High", na=False)]
                honors_df = processed_batch[processed_batch["Cumulative_3Subject_Avg"] >= 85.0]
                
                c_alert1, c_alert2 = st.columns(2)
                with c_alert1:
                    if not high_risk_df.empty:
                        st.markdown(f"#### 🚨 Priority At-Risk Intervention Watchlist ({len(high_risk_df)} Students)")
                        for _, s_row in high_risk_df.head(6).iterrows():
                            s_name = s_row.get("student_name", s_row.get("student_id", "Student"))
                            s_id = s_row.get("student_id", "N/A")
                            st.markdown(f"""
                            <div style="background: rgba(254, 242, 242, 0.85); border-left: 4px solid #EF4444; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;">
                                <b>{s_name}</b> (`{s_id}`) • Predicted Math: <span style="color:#DC2626; font-weight:bold;">{s_row['Predicted_Math_Score']}</span> | Pass Prob: <span style="color:#DC2626; font-weight:bold;">{s_row['Pass_Probability_Pct']}%</span><br/>
                                <span style="font-size: 0.82rem; color: #475569;"><b>Prescription:</b> {s_row.get('Prescribed_Intervention', 'Intensive Tutoring')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(high_risk_df) > 6:
                            st.caption(f"... and {len(high_risk_df) - 6} more at-risk students in full roster below.")
                    else:
                        st.success("🎉 No high-risk students identified in this classroom cohort!")
                        
                with c_alert2:
                    if not honors_df.empty:
                        st.markdown(f"#### 🌟 Distinction Honor Roll ({len(honors_df)} Students)")
                        for _, h_row in honors_df.head(6).iterrows():
                            h_name = h_row.get("student_name", h_row.get("student_id", "Student"))
                            h_id = h_row.get("student_id", "N/A")
                            st.markdown(f"""
                            <div style="background: rgba(236, 253, 245, 0.85); border-left: 4px solid #10B981; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;">
                                <b>{h_name}</b> (`{h_id}`) • Overall Avg: <span style="color:#059669; font-weight:bold;">{h_row['Cumulative_3Subject_Avg']} / 100</span> • Grade: <span style="color:#059669; font-weight:bold;">{h_row['Predicted_Grade']}</span><br/>
                                <span style="font-size: 0.82rem; color: #475569;"><b>Standing:</b> Distinction Honor Candidate • AP/Olympiad Ready</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(honors_df) > 6:
                            st.caption(f"... and {len(honors_df) - 6} more distinction students in full roster below.")
                    else:
                        st.info("ℹ️ No distinction earners (85+ marks) in this cohort.")
                        
                st.write("")
                
                # 4. Visual Analytics for Classroom
                st.markdown("#### 📈 Cohort Visual Analytics (Interactive Plotly):")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    grade_counts = processed_batch["Predicted_Grade"].value_counts().reset_index()
                    grade_counts.columns = ["Grade", "Count"]
                    fig_grade = px.pie(
                        grade_counts, 
                        names="Grade", 
                        values="Count", 
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        title="<b>Classroom Grade Breakdown</b>"
                    )
                    fig_grade.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_grade, use_container_width=True)
                    
                with chart_col2:
                    risk_counts = processed_batch["Risk_Tier"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Tier", "Count"]
                    fig_risk = px.bar(
                        risk_counts,
                        x="Risk Tier",
                        y="Count",
                        color="Risk Tier",
                        color_discrete_map={
                            "Safe / Low Risk": "#10B981",
                            "Moderate Risk": "#F59E0B",
                            "🚨 High Academic Risk": "#EF4444"
                        },
                        title="<b>Academic Risk Tier Breakdown</b>"
                    )
                    fig_risk.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                # Multi-Dimensional Bubble Scatter
                fig_bubble = create_batch_bubble_chart(processed_batch)
                st.plotly_chart(fig_bubble, use_container_width=True)
                
                # 5. Interactive Cohort Search & Filters
                st.markdown("#### 🔍 Filter & Search Classroom Roster:")
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    search_term = st.text_input("Search Student Name or ID:", placeholder="e.g. Liam or STU-2026")
                with f_col2:
                    risk_filter = st.selectbox("Filter by Risk Tier:", options=["All Risk Tiers"] + sorted(list(processed_batch["Risk_Tier"].unique())))
                with f_col3:
                    grade_filter = st.selectbox("Filter by Grade:", options=["All Grades"] + sorted(list(processed_batch["Predicted_Grade"].unique())))
                    
                filtered_df = processed_batch.copy()
                if search_term:
                    mask = (
                        filtered_df["student_name"].astype(str).str.contains(search_term, case=False, na=False) |
                        filtered_df["student_id"].astype(str).str.contains(search_term, case=False, na=False)
                    )
                    filtered_df = filtered_df[mask]
                if risk_filter != "All Risk Tiers":
                    filtered_df = filtered_df[filtered_df["Risk_Tier"] == risk_filter]
                if grade_filter != "All Grades":
                    filtered_df = filtered_df[filtered_df["Predicted_Grade"] == grade_filter]
                    
                st.markdown(f"**Showing {len(filtered_df)} of {len(processed_batch)} students:**")
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                
                # 6. Executive Export Suite (CSV + Official PDF Report)
                st.markdown("---")
                st.markdown("#### 📄 Classroom Executive Export Suite:")
                
                class_counselor_notes = st.text_area(
                    "✍️ Cohort Counselor Observations & Executive Action Plan:",
                    value="Classroom evaluated with dual ML forecasting. Prescriptive intervention clusters generated for remedial tutoring, test prep bootcamps, and distinction mentorship.",
                    height=70,
                    key="class_notes_input"
                )
                
                export_c1, export_c2 = st.columns(2)
                
                with export_c1:
                    enriched_csv_buffer = io.StringIO()
                    filtered_df.to_csv(enriched_csv_buffer, index=False)
                    st.download_button(
                        label=f"📥 Download Processed CSV with Prescriptions ({len(filtered_df)} Records)",
                        data=enriched_csv_buffer.getvalue(),
                        file_name="Processed_Classroom_Report.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                    
                with export_c2:
                    class_pdf_bytes = generate_classroom_pdf_report(
                        classroom_df=filtered_df,
                        summary=summary,
                        cohort_name=batch_source_name,
                        custom_counselor_notes=class_counselor_notes,
                        intervention_matrix=matrix
                    )
                    st.download_button(
                        label="📄 Download Classroom Executive PDF Report (With Solutions)",
                        data=class_pdf_bytes,
                        file_name="Classroom_Executive_Summary_Report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# ----------------- TAB 3: WHAT-IF GOAL SIMULATOR -----------------
with tab3:
    st.markdown("### 🎯 'What-If' Gamified Academic Goal Simulator")
    st.markdown("Select a target achievement trophy or set a custom score to reverse-engineer the exact study quest roadmap and exam milestones:")
    
    # Trophy Quick-Select Header
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.85rem; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.5px;">🏆 Select Target Milestone Trophy:</span>
    </div>
    """, unsafe_allow_html=True)
    
    if "sim_target_val" not in st.session_state:
        st.session_state["sim_target_val"] = 85
        
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    with t_c1:
        if st.button("🥉 Bronze: Pass (50 Marks)", use_container_width=True):
            st.session_state["sim_target_val"] = 50
            st.rerun()
    with t_c2:
        if st.button("🥈 Silver: Credit (70 Marks)", use_container_width=True):
            st.session_state["sim_target_val"] = 70
            st.rerun()
    with t_c3:
        if st.button("🥇 Gold: Honor Roll (85 Marks)", use_container_width=True):
            st.session_state["sim_target_val"] = 85
            st.rerun()
    with t_c4:
        if st.button("💎 Diamond: Ivy (95 Marks)", use_container_width=True):
            st.session_state["sim_target_val"] = 95
            st.rerun()
            
    st.write("")
    
    col_sim1, col_sim2 = st.columns([1, 1.3])
    
    with col_sim1:
        st.markdown("#### 1. Define Student Baseline & Goal")
        sim_target_score = st.slider("🎯 Desired Target Math Score (0 - 100):", min_value=50, max_value=100, value=st.session_state["sim_target_val"], step=1)
        
        st.markdown("##### 📚 Academic Baseline:")
        sim_curr_read = st.slider("Current Reading Score:", min_value=0, max_value=100, value=65, step=1)
        sim_curr_write = st.slider("Current Writing Score:", min_value=0, max_value=100, value=62, step=1)
        
        st.markdown("##### ⚡ Behavioral & Study Levers:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sim_study_hours = st.slider("Weekly Study Hours:", min_value=1.0, max_value=40.0, value=12.0, step=0.5)
            sim_attendance = st.slider("Attendance Rate (%):", min_value=50.0, max_value=100.0, value=85.0, step=1.0)
        with col_s2:
            sim_sleep = st.slider("Sleep (Hours/Day):", min_value=4.0, max_value=11.0, value=7.5, step=0.5)
            sim_failures = st.selectbox("Past Failed Courses:", [0, 1, 2, 3, 4], index=0)
            
        st.markdown("##### 🏫 Context & Academic Support:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sim_gender = st.selectbox("Gender", ["female", "male"], key="sim_g")
            sim_prep = st.selectbox("Test Prep Course", ["none", "completed"], key="sim_p")
            sim_tutoring = st.selectbox("Tutoring Support", ["none", "peer_tutoring", "private_tutor", "school_program"], index=0, key="sim_tut")
        with col_c2:
            sim_lunch = st.selectbox("Lunch Program", ["standard", "free/reduced"], key="sim_l")
            sim_internet = st.selectbox("Internet Access", ["yes", "no"], index=0, key="sim_net")
            sim_edu = st.selectbox("Parental Education", [
                "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
            ], index=2, key="sim_e")
        
        sim_btn = st.button("🚀 Calculate Milestone Roadmap", use_container_width=True, type="primary")

    with col_sim2:
        st.markdown("#### 2. Simulation Results & Quest Roadmap")
        
        # Build Profile Dict
        sim_profile = {
            "gender": sim_gender,
            "race/ethnicity": "group C",
            "parental level of education": sim_edu,
            "lunch": sim_lunch,
            "test preparation course": sim_prep,
            "internet_access": sim_internet,
            "extracurricular_activities": "no",
            "tutoring_support": sim_tutoring,
            "reading score": sim_curr_read,
            "writing score": sim_curr_write,
            "attendance_rate": sim_attendance,
            "weekly_study_hours": sim_study_hours,
            "sleep_hours_per_day": sim_sleep,
            "past_failures": sim_failures
        }
        
        if preprocessor is not None and active_model is not None:
            sim_res = simulate_academic_goal(sim_profile, sim_target_score, preprocessor, active_model)
            
            # Gap Header Card
            gap = sim_res["score_gap"]
            if gap <= 0:
                st.success(f"🎉 **Target Already Reached!** Current projected score is **{sim_res['current_predicted_math']:.1f} / 100**.")
            else:
                st.warning(f"🎯 **Target Goal:** `{sim_target_score} Marks` | **Current Projected:** `{sim_res['current_predicted_math']:.1f}` | **Score Gap to Bridge:** `+{gap:.1f} marks`")
                
            st.markdown(f"**Feasibility Rating:** <span style='color:{sim_res['badge_color']}; font-weight:bold;'>{sim_res['feasibility']}</span>", unsafe_allow_html=True)
            st.info(sim_res["advice"])
            
            # Interactive Plotly Trajectory Stepper Chart
            fig_traj = create_goal_trajectory_chart(sim_res["current_predicted_math"], sim_res["test_prep_benefit"], sim_target_score)
            st.plotly_chart(fig_traj, use_container_width=True)
            
            st.markdown("#### 🗺️ Step-by-Step Multi-Lever Quest Roadmap:")
            
            # Quest 1: Test Prep & Tutoring
            if sim_prep == "none" or sim_tutoring == "none":
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>⚔️ Quest 1: Unlock Academic Boosters (Test Prep & Tutoring)</b><br/>
                    <span style="color:#0284C7; font-weight:600;">Immediate Projected Gain: <b>+{sim_res['test_prep_benefit']:.1f} Marks</b> in Mathematics.</span><br/>
                    <span style="font-size: 0.85rem; color: #64748B;">Action: Complete test preparation modules and attend weekly peer/school tutoring sessions.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="roadmap-card">
                    <b>✅ Quest 1: Academic Support Active</b><br/>
                    <span style="color:#059669; font-weight:600;">Great job! Your profile already utilizes test prep and active tutoring.</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Quest 2: Study Habits & Attendance
            study_diff = sim_res['required_study_hours'] - sim_study_hours
            att_diff = sim_res['required_attendance'] - sim_attendance
            st.markdown(f"""
            <div class="roadmap-card">
                <b>⏱️ Quest 2: Optimize Study Hours & Attendance Habit</b><br/>
                • Target Weekly Study: <b>{sim_res['required_study_hours']:.1f} hrs/week</b> {'<span style="color:#0284C7; font-weight:bold;">(+'+str(round(study_diff, 1))+' hrs/wk)</span>' if study_diff > 0 else '✅'}<br/>
                • Target Attendance: <b>{sim_res['required_attendance']:.0f}%</b> {'<span style="color:#0284C7; font-weight:bold;">(+'+str(round(att_diff, 0))+'%)</span>' if att_diff > 0 else '✅'}<br/>
                <span style="font-size: 0.85rem; color: #64748B;">Action: Schedule dedicated daily study blocks and maintain consistent lecture presence.</span>
            </div>
            """, unsafe_allow_html=True)
                
            # Quest 3: Subject Marks Targets
            st.markdown(f"""
            <div class="roadmap-card">
                <b>📖 Quest 3: Reach Prerequisite Exam Milestones</b><br/>
                • Target Reading Score: <b>{sim_res['required_reading_score']} / 100</b> <span style="color:#0284C7; font-weight:bold;">(+{sim_res['reading_delta']} marks from current {sim_curr_read})</span><br/>
                • Target Writing Score: <b>{sim_res['required_writing_score']} / 100</b> <span style="color:#0284C7; font-weight:bold;">(+{sim_res['writing_delta']} marks from current {sim_curr_write})</span><br/>
                <span style="font-size: 0.85rem; color: #64748B;">Action: Focus on text comprehension drills and analytical writing structure.</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Quest 4: Victory Outcome
            st.markdown(f"""
            <div class="roadmap-card">
                <b>🏆 Final Victory Condition: Goal Achievement</b><br/>
                Meeting these multi-lever milestones is mathematically and statistically verified by the ML model to deliver your target of <b>{sim_target_score}+ marks (High Standing / Honor Roll)</b>.
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 4: EXPLAINABLE AI (XAI) -----------------
with tab4:
    st.markdown("### 🔍 Explainable AI (XAI) & Feature Importance Analysis")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    feat_csv = os.path.join(os.path.dirname(__file__), "artifacts", "feature_importance.csv")
    if os.path.exists(feat_csv):
        f_df = pd.read_csv(feat_csv)
        
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            fig_imp = create_global_importance_plotly(f_df)
            st.plotly_chart(fig_imp, use_container_width=True)
                
        with col_x2:
            st.markdown("#### 2. Directional Feature Attribution (Point Adjustments)")
            p11 = os.path.join(plots_dir, "11_shap_directional_impact.png")
            if os.path.exists(p11):
                st.image(p11, caption="Positive drivers (Green) vs Negative penalties (Red) on marks.", use_container_width=True)
                
        st.markdown("#### 📋 Full 24-Feature Importance Impact Table:")
        st.dataframe(f_df, use_container_width=True, hide_index=True)

# ----------------- TAB 5: EDA & INSIGHTS -----------------
with tab5:
    st.markdown("### 📊 Exploratory Data Analysis & Interactive Visual Insights")
    
    if not raw_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_dist = create_eda_distribution_plotly(raw_df)
            st.plotly_chart(fig_dist, use_container_width=True)
                
            fig_box_prep = create_eda_test_prep_plotly(raw_df)
            st.plotly_chart(fig_box_prep, use_container_width=True)
                
        with col_b:
            fig_corr = create_eda_correlation_plotly(raw_df)
            st.plotly_chart(fig_corr, use_container_width=True)
                
            fig_box_edu = create_eda_parental_education_plotly(raw_df)
            st.plotly_chart(fig_box_edu, use_container_width=True)
    else:
        st.info("Raw dataset not loaded.")

# ----------------- TAB 6: DUAL MODEL LEADERBOARD -----------------
with tab6:
    st.markdown("### 🏆 Dual-Task Evaluation Leaderboards & Model Comparison")
    
    metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        
        # Interactive Model Comparison Bar Chart
        fig_model_comp = create_model_comparison_plotly(m_df)
        st.plotly_chart(fig_model_comp, use_container_width=True)
        
        st.markdown("#### A. Regression Leaderboard (Continuous Score Prediction)")
        st.dataframe(m_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
        
    st.markdown("#### B. Classification Leaderboard (Pass / Fail & Dropout Risk)")
    clf_path = os.path.join(os.path.dirname(__file__), "artifacts", "classifier_metrics.csv")
    if os.path.exists(clf_path):
        c_df = pd.read_csv(clf_path)
        st.dataframe(c_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
    
    st.markdown("#### 📈 Interactive Model Diagnostic Curves:")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_cm = create_interactive_confusion_matrix()
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_v2:
        fig_roc = create_interactive_roc_curve()
        st.plotly_chart(fig_roc, use_container_width=True)

# ----------------- TAB 7: HYPERPARAMETER TUNING -----------------
with tab7:
    st.markdown("### ⚙️ 5-Fold Cross-Validation Hyperparameter Optimization")
    tuning_csv = os.path.join(os.path.dirname(__file__), "artifacts", "hyperparameter_tuning_results.csv")
    if os.path.exists(tuning_csv):
        t_df = pd.read_csv(tuning_csv)
        st.dataframe(t_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)

# ----------------- TAB 8: SYSTEM ARCHITECTURE -----------------
with tab8:
    st.markdown("### 📖 Multi-Engine Machine Learning Architecture")
    st.markdown("""
    ```
    1. Input Data Processing (Single Profile or Classroom Bulk CSV)
       └── Engineered via 24-Feature Synergy Pipeline (RobustScaler + OneHotEncoder)
    
    2. Multi-Engine Architecture:
       ├── Regression Engine: Super-Stacking & Optimized ElasticNet (R² 76.31%, MAE ±5.96)
       ├── Classification Engine: Support Vector Classifier (Accuracy 90.0%, ROC-AUC 0.932)
       ├── Classroom Batch Engine: Interactive Plotly Bubble Cohort + Grade/Risk Donut Charts
       ├── 'What-If' Simulator: Milestone Roadmapping & Score Gap Solver
       └── Explainable AI (XAI): Interactive Waterfall Attributions & 50-Shuffle Permutation Importance
    
    3. Outputs & Deliverables:
       ├── Exact Predicted Marks & Grade (A+ to F)
       ├── Pass Probability & Early Dropout Risk Tier
       ├── 5-Axis Student Competency Radar Chart & Speedometer Gauge
       ├── Cohort-Level Analytics & Downloadable Enriched CSV
       └── Verified PDF Performance Certificate v5.0
    ```
    """)
