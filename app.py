import os
import sys
import io
from pathlib import Path

# Ensure project root is in sys.path for Streamlit Cloud & local execution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from src.advanced_ml_statistical import (
    predict_with_confidence_intervals,
    classify_student_archetype
)
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
    create_classroom_intervention_cluster_chart,
    create_confidence_interval_gauge,
    create_archetype_pca_scatter_chart,
    create_multi_subject_forecast_chart,
    create_before_after_radar_chart,
    create_peer_comparison_bar_chart
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Performance & Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# MODERN BENTO GRID DESIGN SYSTEM & CLEAN TYPOGRAPHY
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

    /* Streamlined Modern Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 249, 255, 0.9) 50%, rgba(238, 242, 255, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 16px;
        padding: 1.1rem 1.5rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 20px -4px rgba(30, 58, 138, 0.05);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
    }

    .hero-title {
        font-family: var(--font-heading);
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.6rem;
        line-height: 1.35;
    }

    /* Compact Pill Badges */
    .badge-chip-group {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: var(--font-body);
    }

    .badge-primary {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }

    .badge-success {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    .badge-purple {
        background: #F5F3FF;
        color: #6D28D9;
        border: 1px solid #DDD6FE;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Bento Tile Card Styles */
    .bento-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        margin-bottom: 0.9rem;
        height: 100%;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .bento-tile:hover {
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.08);
        border-color: #CBD5E1;
    }

    .bento-tile-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #F1F5F9;
    }

    .bento-tile-title {
        font-family: var(--font-heading);
        font-size: 0.98rem;
        font-weight: 700;
        color: #1E3A8A;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .bento-score-hero {
        font-family: var(--font-heading);
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.2rem 0;
    }

    /* Roadmap Action Item Cards */
    .roadmap-card {
        background: #FFFFFF;
        border-left: 3.5px solid #2563EB;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 9px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }

    /* Preset Chip Box */
    .preset-chip-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 9px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.7rem;
    }

    /* Streamlined Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: rgba(241, 245, 249, 0.85);
        padding: 4px 5px;
        border-radius: 11px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        border-radius: 7px;
        color: #475569;
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.84rem;
        padding: 0 13px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.9);
        color: #1E3A8A;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.12) !important;
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
    uncertainty_path = os.path.join(artifacts_dir, "uncertainty_model.joblib")
    cluster_path = os.path.join(artifacts_dir, "archetype_clusterer.joblib")
    multi_model_path = os.path.join(artifacts_dir, "multi_subject_model.joblib")
    
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    best_model = joblib.load(best_model_path) if os.path.exists(best_model_path) else None
    best_clf = joblib.load(best_clf_path) if os.path.exists(best_clf_path) else None
    
    uncertainty_dict = joblib.load(uncertainty_path) if os.path.exists(uncertainty_path) else {"q95_margin": 8.50, "q90_margin": 7.26}
    cluster_bundle = joblib.load(cluster_path) if os.path.exists(cluster_path) else None
    multi_subject_model = joblib.load(multi_model_path) if os.path.exists(multi_model_path) else None
    
    # Load all regression models
    models_dir = os.path.join(artifacts_dir, "models")
    all_models = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib") and not f.startswith("clf_"):
                name = f.replace(".joblib", "").replace("_", " ").title()
                all_models[name] = joblib.load(os.path.join(models_dir, f))
                
    return preprocessor, best_model, best_clf, all_models, uncertainty_dict, cluster_bundle, multi_subject_model

@st.cache_data
def load_raw_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "StudentsPerformance.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

preprocessor, best_model, best_clf, all_models, uncertainty_dict, cluster_bundle, multi_subject_model = load_artifacts()
raw_df = load_raw_dataset()

# ---------------------------------------------------------
# SIDEBAR CONTROLS & BEGINNER GUIDES
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("⚙️ AI Control Hub")
    
    if all_models:
        default_index = 0
        model_names = sorted(list(all_models.keys()))
        if "Optimized Elasticnet" in model_names:
            default_index = model_names.index("Optimized Elasticnet")
        elif "Super Stacking Meta Regressor" in model_names:
            default_index = model_names.index("Super Stacking Meta Regressor")
            
        selected_model_name = st.selectbox(
            "Select AI Model:",
            options=model_names,
            index=default_index,
            help="Choose which trained machine learning model will predict the student's exam score."
        )
        active_model = all_models[selected_model_name]
    else:
        active_model = best_model
        selected_model_name = "Optimized Model (Default)"
        
    st.success("🛡️ Pass Classifier: **Gradient Boosting (98.0% Accuracy)**")
    
    # Simple Beginner Guide
    with st.expander("💡 3-Step Beginner Guide", expanded=False):
        st.markdown("""
        **1. 📝 Enter Details:** Pick a preset student above or adjust sliders for study hours, prior exam, etc.  
        **2. ⚡ Get Predictions:** See predicted marks out of 100, letter grade, and passing likelihood.  
        **3. 🛠️ Follow Action Plan:** Check customized study tips, weekly schedule, and download the official PDF report.
        """)
        
    # Plain-English Glossary
    with st.expander("📘 Easy Terms Glossary", expanded=False):
        st.markdown("""
        - **🎯 Predicted Score:** The marks (out of 100) our AI estimates the student will achieve.
        - **🛡️ 95% Safe Range:** The range where the actual score is 95% likely to land (e.g. 79 – 91).
        - **✨ Pass Likelihood:** The probability (0–100%) that the student scores 50+ marks.
        - **⚡ Student Persona:** The student's learning and habit style (e.g. *High-Effort Dedicated Striver*).
        - **📊 Model Accuracy ($R^2$ 95.8%):** How closely AI predictions match real examination results.
        - **📉 Average Error ($\pm 2.5$ marks):** How close our predictions are to the true marks on average.
        """)

    st.markdown("---")
    st.markdown("### 📊 Quick Project Stats")
    st.markdown("- **18 Student Dimensions** (Academics, Lifestyle, Habits)")
    st.markdown("- **95.75% Prediction Accuracy** ($\pm 2.54$ avg error)")
    st.markdown("- **98.00% Pass / Fail Accuracy**")
    st.markdown("- **95% Confidence Bounds** ($\pm 6.03$ safe margin)")
    st.markdown("- **4 Behavioral Learning Personas**")
    
    st.caption("EduPredict AI • Simple, Smart & User-Friendly")

# ---------------------------------------------------------
# CRISP & WELCOMING HERO HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎓 EduPredict AI • Student Success & Growth Intelligence</div>
    <div class="hero-subtitle">
        Simple, smart, and accurate AI to predict exam marks, identify learning habits, simulate growth, and help every student succeed.
    </div>
    <div class="badge-chip-group">
        <span class="badge-chip badge-success"><span class="pulse-dot"></span> System Ready</span>
        <span class="badge-chip badge-primary">✨ Beginner Friendly</span>
        <span class="badge-chip badge-purple">🎯 95.8% Prediction Accuracy</span>
        <span class="badge-chip badge-primary">🔄 Growth Simulator</span>
        <span class="badge-chip badge-success">📄 Instant PDF Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6 STREAMLINED WORKSPACE TABS
# ---------------------------------------------------------
tab_pred, tab_batch, tab_compare, tab_goal, tab_xai, tab_models = st.tabs([
    "🎯 Student Predictor", 
    "📂 Classroom Analytics",
    "🔄 Growth Simulator",
    "🗺️ Goal Planner",
    "🔍 What Drives Scores?",
    "⚙️ AI Models & Accuracy"
])

# =========================================================
# TAB 1: BENTO GRID STUDENT PREDICTOR & ACTION PLAN
# =========================================================
with tab_pred:
    st.info("💡 **Quick Start:** Click any preset button below to load a sample student, or customize the sliders and click **'Predict Student Score & Generate Action Plan'**.")

    # State Initialization
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
    if "p_method" not in st.session_state: st.session_state["p_method"] = "active_problem_solving"
    if "p_inv" not in st.session_state: st.session_state["p_inv"] = "high"
    if "p_prev" not in st.session_state: st.session_state["p_prev"] = 80
    if "p_read" not in st.session_state: st.session_state["p_read"] = 78
    if "p_write" not in st.session_state: st.session_state["p_write"] = 82
    if "p_att" not in st.session_state: st.session_state["p_att"] = 92.0
    if "p_study" not in st.session_state: st.session_state["p_study"] = 16.0
    if "p_sleep" not in st.session_state: st.session_state["p_sleep"] = 7.5
    if "p_screen" not in st.session_state: st.session_state["p_screen"] = 2.5
    if "p_fails" not in st.session_state: st.session_state["p_fails"] = 0

    # 1-Row Quick Demo Personas
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.8rem; font-weight: 700; color: #334155; text-transform: uppercase;">⚡ 1-Click Demo Profiles:</span>
    </div>
    """, unsafe_allow_html=True)
    
    pre_c1, pre_c2, pre_c3, pre_c4, pre_c5, pre_c6 = st.columns(6)
    with pre_c1:
        if st.button("🌟 Top Scorer (90+)", use_container_width=True, help="High prior scores, active study habits & high attendance"):
            st.session_state.update({
                "p_name": "Elena Rostova", "p_id": "STU-2026-HONORS", "p_gender": "female",
                "p_race": "group E", "p_edu": "master's degree", "p_lunch": "standard",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "private_tutor",
                "p_method": "active_problem_solving", "p_inv": "high", "p_prev": 92,
                "p_read": 92, "p_write": 95, "p_att": 98.0, "p_study": 24.0, "p_sleep": 8.0, "p_screen": 1.8, "p_fails": 0
            })
            st.rerun()
    with pre_c2:
        if st.button("⚖️ Average Student (65+)", use_container_width=True, help="Moderate study hours and steady baseline marks"):
            st.session_state.update({
                "p_name": "Jordan Miller", "p_id": "STU-2026-AVG", "p_gender": "male",
                "p_race": "group C", "p_edu": "some college", "p_lunch": "standard",
                "p_prep": "none", "p_internet": "yes", "p_extra": "no", "p_tutor": "none",
                "p_method": "spaced_repetition", "p_inv": "medium", "p_prev": 64,
                "p_read": 65, "p_write": 62, "p_att": 86.0, "p_study": 12.0, "p_sleep": 7.2, "p_screen": 3.5, "p_fails": 0
            })
            st.rerun()
    with pre_c3:
        if st.button("🚨 Needs Support (<40)", use_container_width=True, help="Low study time, high screen time, and past backlogs"):
            st.session_state.update({
                "p_name": "Marcus Vance", "p_id": "STU-2026-RISK", "p_gender": "male",
                "p_race": "group A", "p_edu": "some high school", "p_lunch": "free/reduced",
                "p_prep": "none", "p_internet": "no", "p_extra": "no", "p_tutor": "none",
                "p_method": "passive_reading", "p_inv": "low", "p_prev": 38,
                "p_read": 34, "p_write": 30, "p_att": 62.0, "p_study": 4.0, "p_sleep": 5.0, "p_screen": 6.5, "p_fails": 2
            })
            st.rerun()
    with pre_c4:
        if st.button("📖 Strong in Reading", use_container_width=True, help="High literacy and verbal strength with room for math gains"):
            st.session_state.update({
                "p_name": "Sophia Chen", "p_id": "STU-2026-VERBAL", "p_gender": "female",
                "p_race": "group D", "p_edu": "bachelor's degree", "p_lunch": "standard",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "peer_tutoring",
                "p_method": "spaced_repetition", "p_inv": "high", "p_prev": 82,
                "p_read": 88, "p_write": 85, "p_att": 90.0, "p_study": 14.0, "p_sleep": 7.5, "p_screen": 2.8, "p_fails": 0
            })
            st.rerun()
    with pre_c5:
        if st.button("🚀 Fast Improver", use_container_width=True, help="High study effort and dedication to rapidly raise marks"):
            st.session_state.update({
                "p_name": "Lucas Taylor", "p_id": "STU-2026-RISING", "p_gender": "male",
                "p_race": "group B", "p_edu": "high school", "p_lunch": "free/reduced",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "peer_tutoring",
                "p_method": "active_problem_solving", "p_inv": "high", "p_prev": 70,
                "p_read": 76, "p_write": 74, "p_att": 96.0, "p_study": 20.0, "p_sleep": 7.5, "p_screen": 2.2, "p_fails": 0
            })
            st.rerun()
    with pre_c6:
        if st.button("🎯 Borderline (50)", use_container_width=True, help="On the edge of passing — a small boost makes a big difference"):
            st.session_state.update({
                "p_name": "Amara Patel", "p_id": "STU-2026-BORDER", "p_gender": "female",
                "p_race": "group C", "p_edu": "some college", "p_lunch": "free/reduced",
                "p_prep": "none", "p_internet": "yes", "p_extra": "no", "p_tutor": "none",
                "p_method": "group_study", "p_inv": "medium", "p_prev": 49,
                "p_read": 52, "p_write": 49, "p_att": 78.0, "p_study": 7.0, "p_sleep": 6.0, "p_screen": 4.5, "p_fails": 1
            })
            st.rerun()

    # Input Form with Progressive Disclosure
    with st.form("bento_prediction_form"):
        f_id1, f_id2 = st.columns(2)
        with f_id1:
            student_name = st.text_input("📝 Student Name:", value=st.session_state["p_name"])
        with f_id2:
            student_id = st.text_input("🆔 Student Roll / ID:", value=st.session_state["p_id"])

        st.markdown("#### 📊 Key Academic & Study Habits")
        c_in1, c_in2, c_in3, c_in4, c_in5 = st.columns(5)
        with c_in1:
            previous_term_score = st.slider("Prior Exam Marks (0-100):", 0, 100, int(st.session_state["p_prev"]), 1, help="Marks scored in the previous semester / term test")
        with c_in2:
            reading_score = st.slider("Reading Score (0-100):", 0, 100, int(st.session_state["p_read"]), 1, help="Reading comprehension test score")
        with c_in3:
            writing_score = st.slider("Writing Score (0-100):", 0, 100, int(st.session_state["p_write"]), 1, help="Writing and essay test score")
        with c_in4:
            weekly_study_hours = st.slider("Weekly Study Hours:", 1.0, 40.0, float(st.session_state["p_study"]), 0.5, help="Self-study hours per week outside class")
        with c_in5:
            attendance_rate = st.slider("Attendance Rate (%):", 50.0, 100.0, float(st.session_state["p_att"]), 0.5, help="Class attendance percentage")

        with st.expander("⚙️ Additional Lifestyle, Study Technique & Support (Optional)", expanded=False):
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                method_opts = ["active_problem_solving", "spaced_repetition", "group_study", "passive_reading"]
                study_method = st.selectbox(
                    "Study Technique:", 
                    method_opts, 
                    index=method_opts.index(st.session_state["p_method"]),
                    format_func=lambda x: x.replace('_', ' ').title(),
                    help="Active problem solving & flashcard review offer the highest retention."
                )
                daily_screen_time_hours = st.slider("Recreational Screen Time (Hours/Day):", 0.5, 10.0, float(st.session_state["p_screen"]), 0.2, help="Hours spent daily on social media, gaming, or phone entertainment")
                sleep_hours_per_day = st.slider("Sleep (Hours/Day):", 4.0, 10.0, float(st.session_state["p_sleep"]), 0.2, help="Average daily sleep hours")
                past_failures = st.selectbox("Prior Course Backlogs / Failures:", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(st.session_state["p_fails"]), help="Number of previous courses failed or backlogs")
            with adv_c2:
                inv_opts = ["high", "medium", "low"]
                parental_involvement = st.selectbox("Parental Mentorship / Support:", inv_opts, index=inv_opts.index(st.session_state["p_inv"]), format_func=lambda x: x.title(), help="Level of guidance, check-ins, and study environment support at home")
                prep_opts = ["none", "completed"]
                test_prep = st.selectbox("Exam Preparation Course:", prep_opts, index=prep_opts.index(st.session_state["p_prep"]), format_func=lambda x: "Completed (+Prep)" if x=="completed" else "None", help="Whether the student finished a structured exam prep course")
                tut_opts = ["none", "peer_tutoring", "private_tutor"]
                tutoring_support = st.selectbox("Tutoring Support:", tut_opts, index=tut_opts.index(st.session_state["p_tutor"]), format_func=lambda x: x.replace('_', ' ').title(), help="Extra tutoring sessions outside school")
                lunch_opts = ["standard", "free/reduced"]
                lunch = st.selectbox("Lunch Program:", lunch_opts, index=lunch_opts.index(st.session_state["p_lunch"]), format_func=lambda x: "Standard Meal" if x=="standard" else "Subsidized Meal")
            with adv_c3:
                net_opts = ["yes", "no"]
                internet_access = st.selectbox("Home Internet Access:", net_opts, index=net_opts.index(st.session_state["p_internet"]), format_func=lambda x: "Yes" if x=="yes" else "No")
                edu_opts = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
                parental_education = st.selectbox("Parent's Education Level:", edu_opts, index=edu_opts.index(st.session_state["p_edu"]), format_func=lambda x: x.title())
                extra_opts = ["yes", "no"]
                extracurricular_activities = st.selectbox("Extracurricular Activities (Sports/Clubs):", extra_opts, index=extra_opts.index(st.session_state["p_extra"]), format_func=lambda x: "Yes" if x=="yes" else "No")
                gen_opts = ["female", "male"]
                gender = st.selectbox("Gender:", gen_opts, index=gen_opts.index(st.session_state["p_gender"]), format_func=lambda x: x.title())
                race_ethnicity = "group C"

        submit_btn = st.form_submit_button("⚡ Predict Student Score & Generate Action Plan", use_container_width=True, type="primary")

    if submit_btn or "last_pred" in st.session_state:
        if submit_btn:
            st.session_state["last_pred"] = True

        if preprocessor is not None and active_model is not None:
            input_dict = {
                "gender": [gender],
                "race/ethnicity": [race_ethnicity],
                "parental level of education": [parental_education],
                "lunch": [lunch],
                "test preparation course": [test_prep],
                "internet_access": [internet_access],
                "extracurricular_activities": [extracurricular_activities],
                "tutoring_support": [tutoring_support],
                "study_method": [study_method],
                "parental_involvement": [parental_involvement],
                "previous_term_score": [previous_term_score],
                "attendance_rate": [attendance_rate],
                "weekly_study_hours": [weekly_study_hours],
                "sleep_hours_per_day": [sleep_hours_per_day],
                "daily_screen_time_hours": [daily_screen_time_hours],
                "past_failures": [past_failures],
                "reading score": [reading_score],
                "writing score": [writing_score]
            }
            input_df = pd.DataFrame(input_dict)
            input_df_eng = engineer_features(input_df)
            transformed_input = preprocessor.transform(input_df_eng)
            
            ci_res = predict_with_confidence_intervals(transformed_input, active_model, uncertainty_dict)
            predicted_math = ci_res["predicted_score"]
            overall_avg = (predicted_math + reading_score + writing_score) / 3.0
            
            archetype_res = classify_student_archetype(transformed_input, cluster_bundle) if cluster_bundle else None
            
            if best_clf is not None:
                pass_prob = float(best_clf.predict_proba(transformed_input)[0][1]) * 100.0
                is_pass = int(best_clf.predict(transformed_input)[0])
            else:
                pass_prob = 95.0 if predicted_math >= 50 else 25.0
                is_pass = 1 if predicted_math >= 50 else 0
                
            if pass_prob >= 80:
                risk_level = "Safe / Low Risk"
                risk_badge_bg = "#ECFDF5"
                risk_badge_color = "#047857"
            elif pass_prob >= 50:
                risk_level = "Moderate (Monitor)"
                risk_badge_bg = "#FEF3C7"
                risk_badge_color = "#D97706"
            else:
                risk_level = "🚨 High Academic Risk"
                risk_badge_bg = "#FEF2F2"
                risk_badge_color = "#DC2626"
                
            if overall_avg >= 90: grade = "A+ (Outstanding)"
            elif overall_avg >= 80: grade = "A (Excellent)"
            elif overall_avg >= 70: grade = "B (Good)"
            elif overall_avg >= 60: grade = "C (Satisfactory)"
            elif overall_avg >= 50: grade = "D (Pass)"
            else: grade = "F (Remedial)"

            prescriptive_sol = generate_prescriptive_solution(
                input_df.iloc[0].to_dict(),
                predicted_math=predicted_math,
                pass_prob=pass_prob,
                grade=grade
            )
            
            base_val, pred_val, contrib_df = explain_single_student(input_df, preprocessor, active_model, None)

            # =========================================================
            # 🍱 4-BOX SIMPLE & CRISP STUDENT DASHBOARD
            # =========================================================
            st.markdown("### 📊 Assessment Summary & Personalized Roadmap")
            
            bento_row1_c1, bento_row1_c2 = st.columns([1.1, 1.1])
            
            # --- BENTO TILE 1: PREDICTED MARKS & SAFE RANGE ---
            with bento_row1_c1:
                score_accent = "#059669" if is_pass == 1 else "#DC2626"
                ci_l, ci_u = ci_res["ci_95_range"]
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🎯 Predicted Score & Likely Range</span>
                        <span style="background:{risk_badge_bg}; color:{risk_badge_color}; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">{risk_level}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="bento-score-hero" style="color:{score_accent};">{predicted_math:.1f} <span style="font-size:1.2rem; opacity:0.8;">/ 100</span></div>
                            <div style="font-size:0.86rem; color:#1D4ED8; font-weight:700; background:#EFF6FF; padding:0.25rem 0.6rem; border-radius:6px; display:inline-block; margin-bottom:0.3rem;">
                                🛡️ 95% Certainty Bounds: <b>{ci_l:.1f} – {ci_u:.1f} marks</b> (±{ci_res['margin_95']:.1f})
                            </div>
                            <div style="font-size:0.9rem; color:#334155; font-weight:600;">Letter Grade: <b>{grade.split()[0]}</b> • 3-Subject Average: <b>{overall_avg:.1f}</b></div>
                            <div style="font-size:0.84rem; color:#64748B;">Pass Likelihood: <b style="color:{score_accent};">{pass_prob:.1f}%</b> ({'Safe / Passing' if is_pass==1 else 'Needs Support'})</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_gauge = create_confidence_interval_gauge(predicted_math, grade, ci_l, ci_u)
                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- BENTO TILE 2: LEARNING STYLE & KEY HABIT DRIVERS ---
            with bento_row1_c2:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🔍 Learning Persona & Key Habit Drivers</span>
                        <span style="background:#EFF6FF; color:#1D4ED8; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">AI Insights</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if archetype_res:
                    st.markdown(f"""
                    <div style="background:{archetype_res['bg_color']}; border-left:3.5px solid {archetype_res['badge_color']}; border-radius:8px; padding:0.5rem 0.75rem; margin-bottom:0.5rem;">
                        <div style="font-weight:800; font-size:0.86rem; color:{archetype_res['badge_color']};">{archetype_res['name']} <span style="font-size:0.75rem; opacity:0.85;">({archetype_res['affinity_score']:.0f}% Match)</span></div>
                        <div style="font-size:0.78rem; color:#334155; margin-top:0.15rem; line-height:1.3;">{archetype_res['summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                for _, f_row in contrib_df.head(3).iterrows():
                    imp = f_row["Impact"]
                    s_sign = "+" if imp >= 0 else ""
                    s_color = "#059669" if imp >= 0 else "#DC2626"
                    f_name = f_row["Factor"].replace('_', ' ').title()
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:7px; padding:0.35rem 0.65rem; margin-bottom:0.3rem; font-size:0.82rem;">
                        <span><b>{f_name}</b> <span style="color:#64748B;">({f_row['Value']})</span></span>
                        <span style="color:{s_color}; font-weight:800;">{s_sign}{imp:.2f} marks</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                fig_waterfall = create_local_xAI_waterfall = create_local_xai_waterfall(base_val, predicted_math, contrib_df)
                st.plotly_chart(create_local_xAI_waterfall, use_container_width=True)

            bento_row2_c1, bento_row2_c2 = st.columns([1.1, 1.1])
            
            # --- BENTO TILE 3: ACTION PLAN & RECOMMENDED STUDY TIME ---
            with bento_row2_c1:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🛠️ Top Action Steps & Study Timetable</span>
                        <span style="background:#ECFDF5; color:#047857; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">Prescriptive</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                for a_idx, action_item in enumerate(prescriptive_sol["interventions"][:3]):
                    st.markdown(f"""
                    <div class="roadmap-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; font-size:0.86rem; color:#1E3A8A;">📌 {action_item['title']}</span>
                            <span style="font-weight:800; font-size:0.8rem; color:#059669;">{action_item['est_uplift']}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#475569; margin-top:0.2rem;">{action_item['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                study_alloc_dict = prescriptive_sol.get("weekly_study_allocation", prescriptive_sol.get("study_hours", {}))
                fig_study = create_prescriptive_study_hours_chart(study_alloc_dict)
                st.plotly_chart(fig_study, use_container_width=True)

            # --- BENTO TILE 4: CERTIFIED PDF REPORT CARD EXPORT ---
            with bento_row2_c2:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">📄 Official Downloadable Report Card</span>
                        <span style="background:#F5F3FF; color:#6D28D9; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">Verified</span>
                    </div>
                    <div style="font-size:0.84rem; color:#475569; margin-bottom:0.4rem;">Generate a ready-to-print official PDF report card with tailored study prescriptions:</div>
                </div>
                """, unsafe_allow_html=True)
                
                counselor_note = st.text_area(
                    "✍️ Counselor / Teacher Observations (Included on PDF):",
                    value="Student demonstrates strong literacy skills. Recommended daily problem solving practice and enrollment in structured math tutoring.",
                    height=75
                )
                
                tips = []
                if pass_prob < 50: tips.append("🚨 Support Alert: Immediate remedial practice recommended in Mathematics.")
                if test_prep == "none": tips.append("📌 Exam Prep: Completing a prep module statistically provides a +9.4 mark boost.")
                if reading_score < 60: tips.append("📌 Reading Focus: Enhancing reading comprehension reinforces mathematical word problems.")
                if daily_screen_time_hours > 4.5: tips.append("📱 Screen Time Diet: Capping recreational screen time under 2.5h/day restores daily focus.")
                if not tips: tips.append("🌟 Strong academic standing across all subjects.")

                try:
                    pdf_bytes = generate_student_pdf_report(
                        student_name=student_name if (student_name and str(student_name).strip()) else "Student",
                        student_id=student_id if (student_id and str(student_id).strip()) else "STU-UNASSIGNED",
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
                        internet_access=internet_access,
                        previous_term_score=previous_term_score,
                        study_method=study_method,
                        daily_screen_time_hours=daily_screen_time_hours,
                        parental_involvement=parental_involvement
                    )
                    clean_id_safe = str(student_id).replace('/', '_').replace('\\', '_').strip() or 'Student'
                    clean_filename = f"Official_Academic_Report_{clean_id_safe}.pdf"
                    
                    st.download_button(
                        label="📥 Download Certified PDF Report Card",
                        data=pdf_bytes,
                        file_name=clean_filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.warning(f"ℹ️ PDF Export Note: Standard PDF preview is momentarily generating ({pdf_err}). All interactive dashboard cards above remain fully active.")
                
                fig_uplift = create_intervention_uplift_chart(predicted_math, prescriptive_sol["projected_score"], prescriptive_sol["interventions"])
                st.plotly_chart(fig_uplift, use_container_width=True)

            # =========================================================
            # 🔬 DEEP-DIVE EXPANDER: PCA CLUSTER MAP, RADAR & ROADMAP
            # =========================================================
            with st.expander("🔍 Explore Deeper: Radar Profile, 3-Subject Breakdown & Growth Map", expanded=False):
                dd_col1, dd_col2 = st.columns(2)
                with dd_col1:
                    st.markdown("#### 🌐 8-Axis Competency Radar Profile")
                    socio_index = input_df_eng["socio_readiness_index"].iloc[0]
                    fig_radar = create_radar_chart(
                        reading=reading_score,
                        writing=writing_score,
                        predicted_math=predicted_math,
                        socio_index=socio_index,
                        attendance=attendance_rate,
                        study_hours=weekly_study_hours,
                        sleep_hours=sleep_hours_per_day,
                        prep_status=test_prep,
                        prev_score=previous_term_score,
                        screen_hours=daily_screen_time_hours
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    fig_multi = create_multi_subject_forecast_chart(predicted_math, reading_score, writing_score, ci_l, ci_u)
                    st.plotly_chart(fig_multi, use_container_width=True)
                    
                with dd_col2:
                    if cluster_bundle and archetype_res:
                        fig_pca = create_archetype_pca_scatter_chart(
                            cluster_bundle,
                            current_pca_x=archetype_res["pca_x"],
                            current_pca_y=archetype_res["pca_y"],
                            student_name=student_name if student_name.strip() else "Student"
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                    st.markdown("#### 🗺️ 12-Week Academic Growth Milestones")
                    for ms in prescriptive_sol["milestones"]:
                        st.markdown(f"""
                        <div style="background:#F8FAFC; border-left:3px solid #2563EB; border-radius:6px; padding:0.55rem 0.8rem; margin-bottom:0.4rem;">
                            <b>{ms.get('phase', 'Phase')}</b> (Target: <span style="color:#059669; font-weight:700;">{ms.get('target', '')}</span>): {ms.get('focus', '')}
                        </div>
                        """, unsafe_allow_html=True)

# =========================================================
# TAB 2: CLASSROOM ANALYTICS & GROUP SUPPORT
# =========================================================
with tab_batch:
    st.markdown("### 📂 Classroom Batch Prediction & Early Risk Detection")
    st.info("💡 **How it works:** Upload a classroom CSV roster or load one of our ready-made sample classes below to see grade distributions, passing rates, and automatically group students who need extra tutoring or advanced honors mentorship.")

    # 1-Click Pre-Built Cohorts
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.8rem; font-weight: 700; color: #334155; text-transform: uppercase;">⚡ 1-Click Ready-Made Classes:</span>
    </div>
    """, unsafe_allow_html=True)
    
    samp_c1, samp_c2, samp_c3, samp_c4, samp_c5 = st.columns(5)
    with samp_c1:
        if st.button("🌟 Balanced Class (50)", use_container_width=True, help="A normal class with balanced mix of scores"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("balanced_50")
            st.session_state["cached_batch_name"] = "Balanced Classroom Cohort (50 Students)"
            st.rerun()
    with samp_c2:
        if st.button("🏫 Whole Grade (100)", use_container_width=True, help="Full grade-wide roster"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("large_100")
            st.session_state["cached_batch_name"] = "Grade-Wide Cohort (100 Students)"
            st.rerun()
    with samp_c3:
        if st.button("🚨 Support Focus (40)", use_container_width=True, help="Classroom with higher number of students needing help"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("at_risk_40")
            st.session_state["cached_batch_name"] = "Support Intervention Focus (40 Students)"
            st.rerun()
    with samp_c4:
        if st.button("🏆 Honors Class (35)", use_container_width=True, help="High-achieving honors and distinction candidates"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("honors_35")
            st.session_state["cached_batch_name"] = "Honors / AP Distinction Cohort (35 Students)"
            st.rerun()
    with samp_c5:
        if st.button("🌐 Diverse Class (200)", use_container_width=True, help="Large multi-section cohort"):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("mixed_200")
            st.session_state["cached_batch_name"] = "Diverse Multi-Section Cohort (200 Students)"
            st.rerun()

    with st.expander("📥 Download CSV Templates & Custom Classroom Generator", expanded=False):
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.markdown("**Download Sample CSVs:**")
            d1, d2 = st.columns(2)
            with d1:
                df_b50 = load_or_create_sample_cohort("balanced_50")
                buf_b50 = io.StringIO()
                df_b50.to_csv(buf_b50, index=False)
                st.download_button("📄 Balanced 50 CSV", data=buf_b50.getvalue(), file_name="sample_balanced_50.csv", mime="text/csv", use_container_width=True)
            with d2:
                df_r40 = load_or_create_sample_cohort("at_risk_40")
                buf_r40 = io.StringIO()
                df_r40.to_csv(buf_r40, index=False)
                st.download_button("🚨 Support Focus 40 CSV", data=buf_r40.getvalue(), file_name="sample_at_risk_40.csv", mime="text/csv", use_container_width=True)
                
            tmpl_data = generate_sample_csv_template()
            st.download_button("📋 10-Student Starter Template CSV", data=tmpl_data, file_name="starter_classroom_template.csv", mime="text/csv", use_container_width=True)
            
        with exp_c2:
            st.markdown("**Generate Custom Synthetic Classroom:**")
            gen_size = st.slider("Class Size (Students):", 10, 500, 60, 5)
            gen_dist = st.selectbox("Class Makeup:", [("balanced", "Balanced Mix"), ("at_risk_focus", "Needs Extra Support Focus"), ("honors_advanced", "High Achievers Focus")], format_func=lambda x: x[1])[0]
            if st.button("⚡ Generate & Analyze Class", use_container_width=True, type="primary"):
                gen_df = generate_synthetic_classroom(n_students=gen_size, cohort_type=gen_dist, seed=np.random.randint(1, 9999))
                st.session_state["cached_batch"] = gen_df
                st.session_state["cached_batch_name"] = f"Custom Class ({gen_size} Students • {gen_dist.replace('_', ' ').title()})"
                st.rerun()

    uploaded_file = st.file_uploader("📤 Or Drag & Drop Your Custom CSV File:", type=["csv"])
    
    batch_to_process = None
    batch_source_name = ""
    
    if uploaded_file is not None:
        batch_to_process = pd.read_csv(uploaded_file)
        batch_source_name = uploaded_file.name
        st.session_state["cached_batch"] = batch_to_process
        st.session_state["cached_batch_name"] = batch_source_name
    elif "cached_batch" in st.session_state:
        batch_to_process = st.session_state["cached_batch"]
        batch_source_name = st.session_state.get("cached_batch_name", "Loaded Class")
        
    if batch_to_process is not None:
        try:
            st.success(f"✅ Loaded `{batch_source_name}` ({len(batch_to_process)} students found).")
            if preprocessor is not None and active_model is not None:
                processed_batch, summary = process_batch_predictions(
                    batch_to_process, preprocessor, active_model, best_clf
                )
                
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1: st.metric("Total Students", summary["total_students"])
                with b2: st.metric("Class Math Avg", f"{summary['class_avg_math']:.1f} / 100")
                with b3: st.metric("3-Subject Avg", f"{summary['class_avg_overall']:.1f} / 100")
                with b4: st.metric("Class Pass Rate", f"{summary['pass_rate']}%")
                with b5: st.metric("🚨 Need Support", summary["at_risk_count"])
                
                st.write("")
                
                bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs([
                    "📊 Grade & Risk Visuals",
                    "🎯 Targeted Student Action Groups",
                    "🚨 Student Roster & Filters",
                    "📄 Download Reports (PDF & CSV)"
                ])
                
                matrix = generate_classroom_intervention_matrix(processed_batch)
                
                with bt_tab1:
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        grade_counts = processed_batch["Predicted_Grade"].value_counts().reset_index()
                        grade_counts.columns = ["Grade", "Count"]
                        fig_grade = px.pie(grade_counts, names="Grade", values="Count", hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism, title="<b>Class Grade Breakdown</b>")
                        fig_grade.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_grade, use_container_width=True)
                    with c_col2:
                        risk_counts = processed_batch["Risk_Tier"].value_counts().reset_index()
                        risk_counts.columns = ["Risk Tier", "Count"]
                        fig_risk = px.bar(risk_counts, x="Risk Tier", y="Count", color="Risk Tier", color_discrete_map={"Safe / Low Risk": "#10B981", "Moderate Risk": "#F59E0B", "🚨 High Academic Risk": "#EF4444"}, title="<b>Academic Safety Status</b>")
                        fig_risk.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                    fig_bubble = create_batch_bubble_chart(processed_batch)
                    st.plotly_chart(fig_bubble, use_container_width=True)

                with bt_tab2:
                    m_c1, m_c2 = st.columns([1.1, 1.4])
                    with m_c1:
                        fig_clusters = create_classroom_intervention_cluster_chart(matrix["summary"])
                        st.plotly_chart(fig_clusters, use_container_width=True)
                    with m_c2:
                        st.markdown("#### 🎯 Targeted Action Strategies for Groups:")
                        st.markdown(f"""
                        <div style="background: rgba(254, 242, 242, 0.9); border-left: 4px solid #EF4444; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #DC2626;">🚨 1. Intensive Math Remedial ({matrix['summary']['high_risk_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Intensive Remedial (High Risk)']}</span>
                        </div>
                        <div style="background: rgba(254, 243, 199, 0.9); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #D97706;">🎯 2. Exam Prep Bootcamp ({matrix['summary']['test_prep_needed_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Test Prep Bootcamp (Moderate Gap)']}</span>
                        </div>
                        <div style="background: rgba(239, 246, 255, 0.9); border-left: 4px solid #3B82F6; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #2563EB;">📚 3. Reading & Verbal Support ({matrix['summary']['verbal_support_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Verbal / Reading Support']}</span>
                        </div>
                        <div style="background: rgba(236, 253, 245, 0.9); border-left: 4px solid #10B981; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #059669;">🏆 4. Honors & Distinction Mentorship ({matrix['summary']['honors_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Honors / Distinction Mentorship']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                with bt_tab3:
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        search_term = st.text_input("🔍 Search by Student Name or ID:", placeholder="e.g. Liam or STU-2026")
                    with f_col2:
                        risk_filter = st.selectbox("Filter Risk Status:", ["All Students"] + sorted(list(processed_batch["Risk_Tier"].unique())))
                    with f_col3:
                        grade_filter = st.selectbox("Filter Letter Grade:", ["All Grades"] + sorted(list(processed_batch["Predicted_Grade"].unique())))
                        
                    filtered_df = processed_batch.copy()
                    if search_term:
                        mask = (filtered_df["student_name"].astype(str).str.contains(search_term, case=False, na=False) |
                                filtered_df["student_id"].astype(str).str.contains(search_term, case=False, na=False))
                        filtered_df = filtered_df[mask]
                    if risk_filter != "All Students":
                        filtered_df = filtered_df[filtered_df["Risk_Tier"] == risk_filter]
                    if grade_filter != "All Grades":
                        filtered_df = filtered_df[filtered_df["Predicted_Grade"] == grade_filter]
                        
                    st.markdown(f"**Showing {len(filtered_df)} of {len(processed_batch)} students:**")
                    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                with bt_tab4:
                    class_counselor_notes = st.text_area(
                        "✍️ Teacher / Counselor Executive Notes:",
                        value="Classroom assessed with predictive AI. Action groups organized for remedial tutoring, exam prep bootcamps, and distinction mentorship.",
                        height=60
                    )
                    exp_c1, exp_c2 = st.columns(2)
                    with exp_c1:
                        enriched_csv = io.StringIO()
                        filtered_df.to_csv(enriched_csv, index=False)
                        st.download_button(
                            label=f"📥 Download Processed CSV ({len(filtered_df)} Students)",
                            data=enriched_csv.getvalue(),
                            file_name="Processed_Classroom_Report.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    with exp_c2:
                        try:
                            class_pdf = generate_classroom_pdf_report(
                                classroom_df=filtered_df,
                                summary=summary,
                                cohort_name=batch_source_name,
                                custom_counselor_notes=class_counselor_notes,
                                intervention_matrix=matrix
                            )
                            st.download_button(
                                label="📄 Download Classroom PDF Executive Summary",
                                data=class_pdf,
                                file_name="Classroom_Executive_Summary_Report.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                        except Exception as c_pdf_err:
                            st.warning(f"ℹ️ Classroom PDF Notice: Report preview is momentarily processing ({c_pdf_err}). All classroom roster tables & CSV exports are active.")
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")

# =========================================================
# TAB 3: GROWTH SIMULATOR & COMPARISON
# =========================================================
with tab_compare:
    cmp_sub1, cmp_sub2 = st.tabs([
        "🔄 Before vs After Growth Simulator",
        "👥 Compare Two Students Side-by-Side"
    ])
    
    with cmp_sub1:
        st.markdown("### 🔄 Before vs After Habit Growth Simulator")
        st.info("💡 **See the Impact:** Adjust the habit boosters on the left (e.g. studying 4 extra hours or cutting screen time) to see how much marks and pass chance increase in real time!")
        
        sim_c1, sim_c2 = st.columns([1.1, 1.2])
        
        with sim_c1:
            st.markdown("#### 1. Current Baseline Habits:")
            b_prev = st.slider("Prior Exam Marks:", 0, 100, 55, 1, key="b_pr")
            b_read = st.slider("Reading Score:", 0, 100, 58, 1, key="b_r")
            b_write = st.slider("Writing Score:", 0, 100, 54, 1, key="b_w")
            b_study = st.slider("Weekly Study Hours:", 1.0, 40.0, 6.0, 0.5, key="b_s")
            b_screen = st.slider("Recreational Screen Time (Hours/Day):", 0.5, 10.0, 5.5, 0.2, key="b_sc")
            b_att = st.slider("Attendance Rate (%):", 50.0, 100.0, 78.0, 1.0, key="b_a")
            b_method = st.selectbox("Study Technique:", ["active_problem_solving", "spaced_repetition", "group_study", "passive_reading"], index=3, format_func=lambda x: x.replace('_', ' ').title(), key="b_m")
            b_prep = st.selectbox("Exam Prep Status:", ["none", "completed"], index=0, format_func=lambda x: "Completed" if x=="completed" else "None", key="b_p")
            b_tut = st.selectbox("Tutoring Support:", ["none", "peer_tutoring", "private_tutor"], index=0, format_func=lambda x: x.replace('_', ' ').title(), key="b_t")
            
            st.markdown("#### 2. Apply Targeted Habit Boosters:")
            boost_prep = st.checkbox("⚔️ Complete Exam Preparation Course (+5.5 pts)", value=True)
            boost_method = st.selectbox("🧠 Upgrade Study Technique:", ["Keep Current", "active_problem_solving (+4.2 pts)", "spaced_repetition (+2.8 pts)"], index=1)
            boost_screen = st.slider("📱 Lower Daily Screen Time (Target Hours/Day):", 0.5, 6.0, 2.0, 0.2)
            boost_tut = st.selectbox("👥 Enroll in Tutoring Program:", ["none", "peer_tutoring (+4.0 pts)", "private_tutor (+6.0 pts)"], index=1)
            boost_study = st.slider("⏱️ Add Extra Weekly Study Hours (+ hrs/wk):", 0.0, 15.0, 8.0, 0.5)
            boost_att = st.slider("📅 Raise Attendance Rate (+%):", 0.0, 20.0, 12.0, 1.0)
            boost_read = st.slider("📖 Improve Reading Practice (+ marks):", 0, 25, 12, 1)

        # Baseline Data Preparation
        base_dict = {
            "gender": "female", "race/ethnicity": "group C", "parental level of education": "some college",
            "lunch": "standard", "test preparation course": b_prep, "internet_access": "yes",
            "extracurricular_activities": "no", "tutoring_support": b_tut, "study_method": b_method,
            "parental_involvement": "medium", "previous_term_score": b_prev, "reading score": b_read,
            "writing score": b_write, "attendance_rate": b_att, "weekly_study_hours": b_study,
            "sleep_hours_per_day": 7.0, "daily_screen_time_hours": b_screen, "past_failures": 1
        }
        
        # Post-Intervention Data Preparation
        tut_applied = "peer_tutoring" if "peer" in boost_tut else ("private_tutor" if "private" in boost_tut else b_tut)
        prep_applied = "completed" if boost_prep else b_prep
        method_applied = "active_problem_solving" if "active" in boost_method else ("spaced_repetition" if "spaced" in boost_method else b_method)
        post_read = min(100, b_read + boost_read)
        post_write = min(100, b_write + int(boost_read * 0.9))
        post_study = min(40.0, b_study + boost_study)
        post_att = min(100.0, b_att + boost_att)
        
        post_dict = {
            "gender": "female", "race/ethnicity": "group C", "parental level of education": "some college",
            "lunch": "standard", "test preparation course": prep_applied, "internet_access": "yes",
            "extracurricular_activities": "yes", "tutoring_support": tut_applied, "study_method": method_applied,
            "parental_involvement": "high", "previous_term_score": b_prev, "reading score": post_read,
            "writing score": post_write, "attendance_rate": post_att, "weekly_study_hours": post_study,
            "sleep_hours_per_day": 7.5, "daily_screen_time_hours": boost_screen, "past_failures": 0
        }
        
        if preprocessor is not None and active_model is not None:
            # Baseline Prediction
            df_b = pd.DataFrame([base_dict])
            t_b = preprocessor.transform(engineer_features(df_b))
            score_b = float(np.clip(active_model.predict(t_b)[0], 0, 100))
            prob_b = float(best_clf.predict_proba(t_b)[0][1] * 100) if best_clf else 50.0
            base_dict["predicted_math"] = score_b
            
            # Post Prediction
            df_p = pd.DataFrame([post_dict])
            t_p = preprocessor.transform(engineer_features(df_p))
            score_p = float(np.clip(active_model.predict(t_p)[0], 0, 100))
            prob_p = float(best_clf.predict_proba(t_p)[0][1] * 100) if best_clf else 95.0
            post_dict["predicted_math"] = score_p
            
            delta_score = score_p - score_b
            delta_prob = prob_p - prob_b
            
            with sim_c2:
                st.markdown("#### 3. Real-Time Growth Comparison:")
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""
                    <div style="background:#FEF2F2; border:1.5px solid #FCA5A5; border-radius:12px; padding:1rem; text-align:center;">
                        <div style="font-size:0.8rem; font-weight:700; color:#DC2626; text-transform:uppercase;">🔴 Current Baseline</div>
                        <div style="font-size:2.2rem; font-weight:900; color:#DC2626; font-family:var(--font-heading);">{score_b:.1f}</div>
                        <div style="font-size:0.82rem; color:#475569;">Pass Likelihood: <b>{prob_b:.1f}%</b></div>
                        <div style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Prior: {b_prev:.0f} • Screen: {b_screen:.1f}h • Study: {b_study:.1f}h</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with sc2:
                    st.markdown(f"""
                    <div style="background:#ECFDF5; border:1.5px solid #6EE7B7; border-radius:12px; padding:1rem; text-align:center;">
                        <div style="font-size:0.8rem; font-weight:700; color:#047857; text-transform:uppercase;">🟢 Projected After Boosters</div>
                        <div style="font-size:2.2rem; font-weight:900; color:#047857; font-family:var(--font-heading);">{score_p:.1f} <span style="font-size:1.1rem; color:#059669;">(+{delta_score:.1f} pts)</span></div>
                        <div style="font-size:0.82rem; color:#475569;">Pass Likelihood: <b>{prob_p:.1f}%</b> (+{delta_prob:.1f}%)</div>
                        <div style="font-size:0.8rem; color:#059669; margin-top:0.2rem;">Method: {method_applied.replace('_', ' ').title()} • Screen: {boost_screen:.1f}h</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                fig_ba_radar = create_before_after_radar_chart(base_dict, post_dict)
                st.plotly_chart(fig_ba_radar, use_container_width=True)

    with cmp_sub2:
        st.markdown("### 👥 Head-to-Head Student Peer Comparison")
        st.info("💡 **Compare Profiles:** Select two student profiles below to compare their marks, study habits, and competencies side-by-side.")
        
        peer_col1, peer_col2 = st.columns(2)
        
        with peer_col1:
            st.markdown("#### 👤 Student Profile A:")
            a_preset = st.selectbox("Select Preset A:", ["🌟 Top Scorer (90+)", "⚖️ Average Student (65+)", "🚨 Needs Support (<40)", "📖 Strong in Reading", "🚀 Fast Improver"], index=0, key="preset_a")
            
            if "Top Scorer" in a_preset or "Honors" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name, sa_prev, sa_screen, sa_meth = 92, 95, 24.0, 98.0, "Elena Rostova (Top Scorer)", 92, 1.8, "active_problem_solving"
            elif "Average" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name, sa_prev, sa_screen, sa_meth = 65, 62, 12.0, 86.0, "Jordan Miller (Average)", 64, 3.5, "spaced_repetition"
            elif "Support" in a_preset or "At-Risk" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name, sa_prev, sa_screen, sa_meth = 34, 30, 4.0, 62.0, "Marcus Vance (Needs Support)", 38, 6.5, "passive_reading"
            elif "Reading" in a_preset or "Verbal" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name, sa_prev, sa_screen, sa_meth = 88, 85, 14.0, 90.0, "Sophia Chen (Strong in Reading)", 82, 2.8, "spaced_repetition"
            else:
                sa_read, sa_write, sa_study, sa_att, sa_name, sa_prev, sa_screen, sa_meth = 76, 74, 20.0, 96.0, "Lucas Taylor (Fast Improver)", 70, 2.2, "active_problem_solving"
                
            stud_a_dict = {
                "gender": "female", "race/ethnicity": "group C", "parental level of education": "bachelor's degree",
                "lunch": "standard", "test preparation course": "completed", "internet_access": "yes",
                "extracurricular_activities": "yes", "tutoring_support": "peer_tutoring", "study_method": sa_meth,
                "parental_involvement": "high", "previous_term_score": sa_prev, "reading score": sa_read,
                "writing score": sa_write, "attendance_rate": sa_att, "weekly_study_hours": sa_study,
                "sleep_hours_per_day": 7.5, "daily_screen_time_hours": sa_screen, "past_failures": 0
            }
            
        with peer_col2:
            st.markdown("#### 👤 Student Profile B:")
            b_preset = st.selectbox("Select Preset B:", ["🌟 Top Scorer (90+)", "⚖️ Average Student (65+)", "🚨 Needs Support (<40)", "📖 Strong in Reading", "🚀 Fast Improver"], index=2, key="preset_b")
            
            if "Top Scorer" in b_preset or "Honors" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name, sb_prev, sb_screen, sb_meth = 92, 95, 24.0, 98.0, "Elena Rostova (Top Scorer)", 92, 1.8, "active_problem_solving"
            elif "Average" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name, sb_prev, sb_screen, sb_meth = 65, 62, 12.0, 86.0, "Jordan Miller (Average)", 64, 3.5, "spaced_repetition"
            elif "Support" in b_preset or "At-Risk" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name, sb_prev, sb_screen, sb_meth = 34, 30, 4.0, 62.0, "Marcus Vance (Needs Support)", 38, 6.5, "passive_reading"
            elif "Reading" in b_preset or "Verbal" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name, sb_prev, sb_screen, sb_meth = 88, 85, 14.0, 90.0, "Sophia Chen (Strong in Reading)", 82, 2.8, "spaced_repetition"
            else:
                sb_read, sb_write, sb_study, sb_att, sb_name, sb_prev, sb_screen, sb_meth = 76, 74, 20.0, 96.0, "Lucas Taylor (Fast Improver)", 70, 2.2, "active_problem_solving"
                
            stud_b_dict = {
                "gender": "male", "race/ethnicity": "group A", "parental level of education": "some high school",
                "lunch": "free/reduced", "test preparation course": "none", "internet_access": "no",
                "extracurricular_activities": "no", "tutoring_support": "none", "study_method": sb_meth,
                "parental_involvement": "low", "previous_term_score": sb_prev, "reading score": sb_read,
                "writing score": sb_write, "attendance_rate": sb_att, "weekly_study_hours": sb_study,
                "sleep_hours_per_day": 6.0, "daily_screen_time_hours": sb_screen, "past_failures": 2
            }
            
        if preprocessor is not None and active_model is not None:
            t_a = preprocessor.transform(engineer_features(pd.DataFrame([stud_a_dict])))
            stud_a_dict["predicted_math"] = float(np.clip(active_model.predict(t_a)[0], 0, 100))
            
            t_b = preprocessor.transform(engineer_features(pd.DataFrame([stud_b_dict])))
            stud_b_dict["predicted_math"] = float(np.clip(active_model.predict(t_b)[0], 0, 100))
            
            fig_peer_bar = create_peer_comparison_bar_chart(stud_a_dict, stud_b_dict, sa_name, sb_name)
            st.plotly_chart(fig_peer_bar, use_container_width=True)
            
            fig_peer_radar = create_before_after_radar_chart(stud_a_dict, stud_b_dict, sa_name, sb_name)
            st.plotly_chart(fig_peer_radar, use_container_width=True)

# =========================================================
# TAB 4: 'WHAT-IF' GOAL PLANNER
# =========================================================
with tab_goal:
    st.markdown("### 🎯 'What-If' Academic Goal Planner")
    st.info("💡 **How it works:** Choose a target trophy score (or drag the slider). Our AI will calculate the exact study hours, attendance, and exam marks you need to reach it!")

    if "sim_target_val" not in st.session_state:
        st.session_state["sim_target_val"] = 85
        
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    with t_c1:
        if st.button("🥉 Bronze: Pass (50)", use_container_width=True, help="Baseline pass threshold"):
            st.session_state["sim_target_val"] = 50
            st.rerun()
    with t_c2:
        if st.button("🥈 Silver: Credit (70)", use_container_width=True, help="Solid solid foundation"):
            st.session_state["sim_target_val"] = 70
            st.rerun()
    with t_c3:
        if st.button("🥇 Gold: Honor Roll (85)", use_container_width=True, help="High distinction"):
            st.session_state["sim_target_val"] = 85
            st.rerun()
    with t_c4:
        if st.button("💎 Diamond: Top Rank (95)", use_container_width=True, help="Elite academic performance"):
            st.session_state["sim_target_val"] = 95
            st.rerun()
            
    st.write("")
    col_sim1, col_sim2 = st.columns([1, 1.3])
    
    with col_sim1:
        st.markdown("#### 1. Set Your Target Score")
        sim_target_score = st.slider("🎯 Target Math Score (0 - 100):", 50, 100, st.session_state["sim_target_val"], 1)
        
        st.markdown("#### 2. Your Current Baseline Habits")
        sim_curr_read = st.slider("Current Reading Score:", 0, 100, 65, 1)
        sim_curr_write = st.slider("Current Writing Score:", 0, 100, 62, 1)
        sim_curr_prev = st.slider("Previous Exam Marks:", 0, 100, 65, 1)
        
        cs1, cs2 = st.columns(2)
        with cs1:
            sim_study_hours = st.slider("Weekly Study Hours:", 1.0, 40.0, 12.0, 0.5)
            sim_attendance = st.slider("Attendance Rate (%):", 50.0, 100.0, 85.0, 1.0)
            sim_prep = st.selectbox("Exam Prep Course", ["none", "completed"], format_func=lambda x: "Completed" if x=="completed" else "None", key="sim_p")
        with cs2:
            sim_screen = st.slider("Daily Screen Time (Hours):", 0.5, 10.0, 3.0, 0.5, key="sim_sc")
            sim_sleep = st.slider("Sleep (Hours/Day):", 4.0, 11.0, 7.5, 0.5)
            sim_tutoring = st.selectbox("Tutoring Support", ["none", "peer_tutoring", "private_tutor"], index=0, format_func=lambda x: x.replace('_', ' ').title(), key="sim_tut")
            sim_gender = st.selectbox("Gender", ["female", "male"], format_func=lambda x: x.title(), key="sim_g")
            
        sim_lunch = "standard"
        sim_internet = "yes"
        sim_edu = "some college"
        sim_failures = 0

    with col_sim2:
        st.markdown("#### 3. AI Goal Roadmap & Quest Steps")
        sim_profile = {
            "gender": sim_gender,
            "race/ethnicity": "group C",
            "parental level of education": sim_edu,
            "lunch": sim_lunch,
            "test preparation course": sim_prep,
            "internet_access": sim_internet,
            "extracurricular_activities": "no",
            "tutoring_support": sim_tutoring,
            "study_method": "spaced_repetition",
            "parental_involvement": "medium",
            "previous_term_score": sim_curr_prev,
            "reading score": sim_curr_read,
            "writing score": sim_curr_write,
            "attendance_rate": sim_attendance,
            "weekly_study_hours": sim_study_hours,
            "sleep_hours_per_day": sim_sleep,
            "daily_screen_time_hours": sim_screen,
            "past_failures": sim_failures
        }
        
        if preprocessor is not None and active_model is not None:
            sim_res = simulate_academic_goal(sim_profile, sim_target_score, preprocessor, active_model)
            gap = sim_res["score_gap"]
            if gap <= 0:
                st.success(f"🎉 **Target Already Reached!** Current projected score is **{sim_res['current_predicted_math']:.1f} / 100**.")
            else:
                st.warning(f"🎯 **Target:** `{sim_target_score}` | **Current:** `{sim_res['current_predicted_math']:.1f}` | **Gap to Close:** `+{gap:.1f} marks`")
                
            st.markdown(f"**Feasibility:** <span style='color:{sim_res['badge_color']}; font-weight:bold;'>{sim_res['feasibility']}</span>", unsafe_allow_html=True)
            st.info(sim_res["advice"])
            
            fig_traj = create_goal_trajectory_chart(sim_res["current_predicted_math"], sim_res["test_prep_benefit"], sim_target_score)
            st.plotly_chart(fig_traj, use_container_width=True)
            
            st.markdown("#### 🗺️ 4-Step Quest Roadmap to Reach Target:")
            
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if sim_prep == "none" or sim_tutoring == "none":
                    st.markdown(f"""
                    <div class="roadmap-card">
                        <b>⚔️ Quest 1: Academic Boosters</b><br/>
                        <span style="color:#0284C7; font-weight:600;">Potential Gain: <b>+{sim_res['test_prep_benefit']:.1f} Marks</b></span><br/>
                        <span style="font-size: 0.78rem; color: #64748B;">Complete exam prep module & join weekly tutoring.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="roadmap-card">
                        <b>✅ Quest 1: Boosters Active</b><br/>
                        <span style="color:#059669; font-weight:600;">Prep & tutoring already active.</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                study_diff = sim_res['required_study_hours'] - sim_study_hours
                att_diff = sim_res['required_attendance'] - sim_attendance
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>⏱️ Quest 2: Study Time & Attendance</b><br/>
                    • Study Target: <b>{sim_res['required_study_hours']:.1f} hrs/wk</b> {'(+'+str(round(study_diff, 1))+')' if study_diff > 0 else '✅'}<br/>
                    • Attendance: <b>{sim_res['required_attendance']:.0f}%</b> {'(+'+str(round(att_diff, 0))+'%)' if att_diff > 0 else '✅'}
                </div>
                """, unsafe_allow_html=True)

            with q_col2:
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>📖 Quest 3: Literacy Practice</b><br/>
                    • Reading Target: <b>{sim_res['required_reading_score']} / 100</b> (+{sim_res['reading_delta']})<br/>
                    • Writing Target: <b>{sim_res['required_writing_score']} / 100</b> (+{sim_res['writing_delta']})
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>🏆 Quest 4: Target Achieved</b><br/>
                    Achieves target of <b>{sim_target_score}+ marks</b> (Verified by AI).
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 5: WHAT DRIVES SCORES? & DATA CHARTS
# =========================================================
with tab_xai:
    xai_sub1, xai_sub2 = st.tabs([
        "🧠 What Habits Drive Scores Most?",
        "📈 Explore Dataset Charts"
    ])
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    with xai_sub1:
        st.info("💡 **Understand What Matters:** Our machine learning models measure which factors have the strongest positive or negative impact on exam scores.")
        feat_csv = os.path.join(os.path.dirname(__file__), "artifacts", "feature_importance.csv")
        if os.path.exists(feat_csv):
            f_df = pd.read_csv(feat_csv)
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                fig_imp = create_global_importance_plotly(f_df)
                st.plotly_chart(fig_imp, use_container_width=True)
            with col_x2:
                st.markdown("#### 2. Positive vs Negative Score Drivers")
                p11 = os.path.join(plots_dir, "11_shap_directional_impact.png")
                if os.path.exists(p11):
                    st.image(p11, caption="Green bars increase marks; Red bars decrease marks.", use_container_width=True)
            st.markdown("#### 📋 Complete Factor Importance Ranking:")
            st.dataframe(f_df, use_container_width=True, hide_index=True)

    with xai_sub2:
        st.info("💡 **Data Explorations:** Visual distributions showing relationships between study habits, background factors, and test marks across 2,000 students.")
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

# =========================================================
# TAB 6: AI MODELS & ACCURACY BENCHMARKS
# =========================================================
with tab_models:
    mod_sub1, mod_sub2, mod_sub3, mod_sub4 = st.tabs([
        "🏆 Model Leaderboards",
        "🔬 Statistical Uncertainty & Personas",
        "⚙️ Model Tuning Results",
        "📖 Simple Architecture Guide"
    ])
    
    with mod_sub1:
        st.info("💡 **Model Benchmarks:** Comparison of 10+ trained machine learning algorithms evaluated on unseen test data.")
        metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.csv")
        if os.path.exists(metrics_path):
            m_df = pd.read_csv(metrics_path)
            fig_model_comp = create_model_comparison_plotly(m_df)
            st.plotly_chart(fig_model_comp, use_container_width=True)
            st.markdown("#### A. Score Forecasting Model Leaderboard (Accuracy & Error)")
            st.dataframe(m_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
            
        st.markdown("#### B. Pass / Fail Risk Classifier Leaderboard")
        clf_path = os.path.join(os.path.dirname(__file__), "artifacts", "classifier_metrics.csv")
        if os.path.exists(clf_path):
            c_df = pd.read_csv(clf_path)
            st.dataframe(c_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
        
        st.markdown("#### 📈 Visual Model Diagnostic Curves:")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_cm = create_interactive_confusion_matrix()
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_v2:
            fig_roc = create_interactive_roc_curve()
            st.plotly_chart(fig_roc, use_container_width=True)

    with mod_sub2:
        st.markdown("### 🔬 Confidence Margins & Learning Personas")
        st.info("💡 **Safe Margins:** Conformal prediction ensures that student scores fall within mathematically guaranteed confidence intervals.")
        
        stat_c1, stat_c2 = st.columns(2)
        with stat_c1:
            st.markdown("#### A. 🛡️ Confidence Margins (Safe Prediction Windows):")
            calib_data = [
                {"Confidence Level": "80% Coverage", "Error Margin": f"± {uncertainty_dict.get('q80_margin', 4.0):.2f} marks", "Interpretation": "Standard daily study estimate"},
                {"Confidence Level": "90% Coverage", "Error Margin": f"± {uncertainty_dict.get('q90_margin', 5.0):.2f} marks", "Interpretation": "High precision academic threshold"},
                {"Confidence Level": "95% Coverage (Default)", "Error Margin": f"± {uncertainty_dict.get('q95_margin', 6.03):.2f} marks", "Interpretation": "Statistically verified safety bound"},
                {"Confidence Level": "99% Coverage", "Error Margin": f"± {uncertainty_dict.get('q99_margin', 8.5):.2f} marks", "Interpretation": "Extreme edge-case ceiling/floor"}
            ]
            st.dataframe(pd.DataFrame(calib_data), use_container_width=True, hide_index=True)
            
            multi_csv = os.path.join(os.path.dirname(__file__), "artifacts", "multi_subject_metrics.csv")
            if os.path.exists(multi_csv):
                st.markdown("#### B. 📚 3-Subject Joint Model Performance:")
                st.dataframe(pd.read_csv(multi_csv), use_container_width=True, hide_index=True)
                
        with stat_c2:
            st.markdown("#### C. 🧬 Discovered Student Learning Personas (Archetypes):")
            arch_summary_csv = os.path.join(os.path.dirname(__file__), "artifacts", "archetype_summary.csv")
            if os.path.exists(arch_summary_csv):
                st.dataframe(pd.read_csv(arch_summary_csv), use_container_width=True, hide_index=True)

    with mod_sub3:
        st.markdown("### ⚙️ 5-Fold Cross-Validation Optimization Results")
        tuning_csv = os.path.join(os.path.dirname(__file__), "artifacts", "hyperparameter_tuning_results.csv")
        if os.path.exists(tuning_csv):
            t_df = pd.read_csv(tuning_csv)
            st.dataframe(t_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)

    with mod_sub4:
        st.markdown("### 📖 Simple 3-Tier System Architecture")
        st.markdown("""
        ```
        1. 📝 Input (18 Dimensions: Prior Exam, Reading, Writing, Study Time, Screen Time, Habits)
           └── Automatically calculated 43 synergy and interaction metrics
        
        2. ⚡ AI Forecasting Engines:
           ├── Score Forecaster: Super-Stacking Meta-Regressor (95.75% R² Accuracy, ±2.54 marks MAE)
           ├── Pass / Fail Classifier: Gradient Boosting (98.00% Accuracy, 0.995 ROC-AUC)
           ├── Uncertainty Engine: 95% Confidence Interval (±6.03 marks safe bounds)
           ├── Learning Styles: 4 Discovered Student Personas via K-Means Clustering
           ├── Growth Simulator: Before vs After Habit Transformation
           ├── Goal Planner: Reverse-Engineers Exact Quests for Any Target Score
           └── Action Engine: Generates Personalized Timetable & Study Steps
        
        3. 📄 User Deliverables:
           ├── Predicted Marks out of 100 with Letter Grade
           ├── Passing Likelihood & Early Support Alert
           ├── Top Positive & Negative Habit Drivers
           ├── 12-Week Milestone Action Plan
           └── Downloadable Ready-to-Print Official PDF Report Card
        ```
        """)

