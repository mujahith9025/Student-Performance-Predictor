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
# SIDEBAR CONTROLS & BENCHMARKS
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
            "AI Forecasting Algorithm:",
            options=model_names,
            index=default_index,
            help="Select the machine learning algorithm used for score prediction"
        )
        active_model = all_models[selected_model_name]
    else:
        active_model = best_model
        selected_model_name = "Optimized Model (Default)"
        
    st.info("🛡️ Pass Classifier: **Support Vector Machine (97.0% Accuracy)**")
    
    st.markdown("---")
    st.markdown("### 📊 Statistical Capabilities")
    st.markdown("- **14 Student Input Dimensions**")
    st.markdown("- **38 Synergy Interaction Metrics**")
    st.markdown("- **Regression $R^2$:** **90.07%** ($\pm 3.47$ marks)")
    st.markdown("- **95% Conformal Prediction Intervals:** **Active**")
    st.markdown("- **Side-by-Side Growth Simulator:** **Active**")
    st.markdown("- **Unsupervised Archetypes:** **4 Discovered**")
    
    st.caption("EduPredict AI v3.8 • Side-by-Side Suite")

# ---------------------------------------------------------
# STREAMLINED HERO HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎓 EduPredict AI • Student Intelligence Hub</div>
    <div class="hero-subtitle">
        Intelligent multi-dimensional academic forecasting, 95% conformal prediction intervals, side-by-side growth simulator, classroom batch analytics, and verified PDF certificates.
    </div>
    <div class="badge-chip-group">
        <span class="badge-chip badge-success"><span class="pulse-dot"></span> Statistical Suite Online</span>
        <span class="badge-chip badge-primary">⚡ 14-Dimension Profile</span>
        <span class="badge-chip badge-purple">🎯 90.1% R² Precision</span>
        <span class="badge-chip badge-primary">🔄 Before/After Simulator</span>
        <span class="badge-chip badge-success">📄 Certified PDF Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6 STREAMLINED WORKSPACE TABS
# ---------------------------------------------------------
tab_pred, tab_batch, tab_compare, tab_goal, tab_xai, tab_models = st.tabs([
    "🎯 Student Predictor", 
    "📂 Classroom Analytics",
    "🔄 Side-by-Side Simulator",
    "🗺️ 'What-If' Simulator",
    "🔍 AI Insights & Data",
    "⚙️ Models & Architecture"
])

# =========================================================
# TAB 1: BENTO GRID STUDENT PREDICTOR & ACTION PLAN
# =========================================================
with tab_pred:
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
    if "p_read" not in st.session_state: st.session_state["p_read"] = 78
    if "p_write" not in st.session_state: st.session_state["p_write"] = 82
    if "p_att" not in st.session_state: st.session_state["p_att"] = 92.0
    if "p_study" not in st.session_state: st.session_state["p_study"] = 16.0
    if "p_sleep" not in st.session_state: st.session_state["p_sleep"] = 7.5
    if "p_fails" not in st.session_state: st.session_state["p_fails"] = 0

    # 1-Row Quick Demo Personas
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.8rem; font-weight: 700; color: #334155; text-transform: uppercase;">⚡ Quick-Load Demo Student Presets:</span>
    </div>
    """, unsafe_allow_html=True)
    
    pre_c1, pre_c2, pre_c3, pre_c4, pre_c5, pre_c6 = st.columns(6)
    with pre_c1:
        if st.button("🌟 Honors (90+)", use_container_width=True):
            st.session_state.update({
                "p_name": "Elena Rostova", "p_id": "STU-2026-HONORS", "p_gender": "female",
                "p_race": "group E", "p_edu": "master's degree", "p_lunch": "standard",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "private_tutor",
                "p_read": 92, "p_write": 95, "p_att": 98.0, "p_study": 24.0, "p_sleep": 8.0, "p_fails": 0
            })
            st.rerun()
    with pre_c2:
        if st.button("⚖️ Average (65+)", use_container_width=True):
            st.session_state.update({
                "p_name": "Jordan Miller", "p_id": "STU-2026-AVG", "p_gender": "male",
                "p_race": "group C", "p_edu": "some college", "p_lunch": "standard",
                "p_prep": "none", "p_internet": "yes", "p_extra": "no", "p_tutor": "none",
                "p_read": 65, "p_write": 62, "p_att": 86.0, "p_study": 12.0, "p_sleep": 7.2, "p_fails": 0
            })
            st.rerun()
    with pre_c3:
        if st.button("🚨 High-Risk (<40)", use_container_width=True):
            st.session_state.update({
                "p_name": "Marcus Vance", "p_id": "STU-2026-RISK", "p_gender": "male",
                "p_race": "group A", "p_edu": "some high school", "p_lunch": "free/reduced",
                "p_prep": "none", "p_internet": "no", "p_extra": "no", "p_tutor": "none",
                "p_read": 34, "p_write": 30, "p_att": 62.0, "p_study": 4.0, "p_sleep": 5.0, "p_fails": 2
            })
            st.rerun()
    with pre_c4:
        if st.button("📖 Verbal Focus", use_container_width=True):
            st.session_state.update({
                "p_name": "Sophia Chen", "p_id": "STU-2026-VERBAL", "p_gender": "female",
                "p_race": "group D", "p_edu": "bachelor's degree", "p_lunch": "standard",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "peer_tutoring",
                "p_read": 88, "p_write": 85, "p_att": 90.0, "p_study": 14.0, "p_sleep": 7.5, "p_fails": 0
            })
            st.rerun()
    with pre_c5:
        if st.button("🚀 Rising Star", use_container_width=True):
            st.session_state.update({
                "p_name": "Lucas Taylor", "p_id": "STU-2026-RISING", "p_gender": "male",
                "p_race": "group B", "p_edu": "high school", "p_lunch": "free/reduced",
                "p_prep": "completed", "p_internet": "yes", "p_extra": "yes", "p_tutor": "peer_tutoring",
                "p_read": 76, "p_write": 74, "p_att": 96.0, "p_study": 20.0, "p_sleep": 7.5, "p_fails": 0
            })
            st.rerun()
    with pre_c6:
        if st.button("🎯 Borderline (50)", use_container_width=True):
            st.session_state.update({
                "p_name": "Amara Patel", "p_id": "STU-2026-BORDER", "p_gender": "female",
                "p_race": "group C", "p_edu": "some college", "p_lunch": "free/reduced",
                "p_prep": "none", "p_internet": "yes", "p_extra": "no", "p_tutor": "none",
                "p_read": 52, "p_write": 49, "p_att": 78.0, "p_study": 7.0, "p_sleep": 6.0, "p_fails": 1
            })
            st.rerun()

    # Input Form with Progressive Disclosure
    with st.form("bento_prediction_form"):
        f_id1, f_id2 = st.columns(2)
        with f_id1:
            student_name = st.text_input("Student Name:", value=st.session_state["p_name"])
        with f_id2:
            student_id = st.text_input("Student Roll / ID:", value=st.session_state["p_id"])

        st.markdown("#### ⚡ Essential Academic & Habit Levers")
        c_in1, c_in2, c_in3, c_in4 = st.columns(4)
        with c_in1:
            reading_score = st.slider("Reading Score (0-100):", 0, 100, int(st.session_state["p_read"]), 1)
        with c_in2:
            writing_score = st.slider("Writing Score (0-100):", 0, 100, int(st.session_state["p_write"]), 1)
        with c_in3:
            weekly_study_hours = st.slider("Study Hours / Week:", 1.0, 40.0, float(st.session_state["p_study"]), 0.5)
        with c_in4:
            attendance_rate = st.slider("Attendance Rate (%):", 50.0, 100.0, float(st.session_state["p_att"]), 0.5)

        with st.expander("⚙️ Additional Lifestyle & Support Levers (Optional Details)", expanded=False):
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                sleep_hours_per_day = st.slider("Sleep (Hours/Day):", 4.0, 10.0, float(st.session_state["p_sleep"]), 0.2)
                past_failures = st.selectbox("Prior Course Failures:", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(st.session_state["p_fails"]))
                tut_opts = ["none", "peer_tutoring", "private_tutor"]
                tutoring_support = st.selectbox("Tutoring Support:", tut_opts, index=tut_opts.index(st.session_state["p_tutor"]), format_func=lambda x: x.replace('_', ' ').title())
            with adv_c2:
                prep_opts = ["none", "completed"]
                test_prep = st.selectbox("Test Prep Course:", prep_opts, index=prep_opts.index(st.session_state["p_prep"]))
                lunch_opts = ["standard", "free/reduced"]
                lunch = st.selectbox("Lunch Program:", lunch_opts, index=lunch_opts.index(st.session_state["p_lunch"]))
                net_opts = ["yes", "no"]
                internet_access = st.selectbox("Home Internet Access:", net_opts, index=net_opts.index(st.session_state["p_internet"]))
            with adv_c3:
                edu_opts = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
                parental_education = st.selectbox("Parental Education:", edu_opts, index=edu_opts.index(st.session_state["p_edu"]))
                extra_opts = ["yes", "no"]
                extracurricular_activities = st.selectbox("Extracurricular Engagement:", extra_opts, index=extra_opts.index(st.session_state["p_extra"]))
                gen_opts = ["female", "male"]
                gender = st.selectbox("Gender:", gen_opts, index=gen_opts.index(st.session_state["p_gender"]))
                race_ethnicity = "group C"

        submit_btn = st.form_submit_button("⚡ Calculate Student Assessment & Statistical Intervals", use_container_width=True, type="primary")

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
            # 🍱 4-BOX MODERN BENTO GRID DASHBOARD
            # =========================================================
            st.markdown("### 🍱 Student Academic Intelligence Dashboard")
            
            bento_row1_c1, bento_row1_c2 = st.columns([1.1, 1.1])
            
            # --- BENTO TILE 1: ACADEMIC STANDING & 95% CONFIDENCE INTERVAL ---
            with bento_row1_c1:
                score_accent = "#059669" if is_pass == 1 else "#DC2626"
                ci_l, ci_u = ci_res["ci_95_range"]
                st.markdown(f"""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🎯 Predicted Marks & 95% Confidence Interval</span>
                        <span style="background:{risk_badge_bg}; color:{risk_badge_color}; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">{risk_level}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="bento-score-hero" style="color:{score_accent};">{predicted_math:.1f} <span style="font-size:1.2rem; opacity:0.8;">/ 100</span></div>
                            <div style="font-size:0.86rem; color:#1D4ED8; font-weight:700; background:#EFF6FF; padding:0.25rem 0.6rem; border-radius:6px; display:inline-block; margin-bottom:0.3rem;">
                                🛡️ 95% Confidence Bounds: <b>{ci_l:.1f} – {ci_u:.1f} marks</b> (±{ci_res['margin_95']:.1f})
                            </div>
                            <div style="font-size:0.9rem; color:#334155; font-weight:600;">Grade: <b>{grade.split()[0]}</b> • 3-Subject Avg: <b>{overall_avg:.1f}</b></div>
                            <div style="font-size:0.84rem; color:#64748B;">Pass Probability: <b style="color:{score_accent};">{pass_prob:.1f}%</b> ({'Passed' if is_pass==1 else 'At-Risk'})</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_gauge = create_confidence_interval_gauge(predicted_math, grade, ci_l, ci_u)
                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- BENTO TILE 2: WHY DID AI PREDICT THIS SCORE & ARCHETYPE ---
            with bento_row1_c2:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🔍 Behavioral Archetype & Point Drivers</span>
                        <span style="background:#EFF6FF; color:#1D4ED8; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">AI Insights</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if archetype_res:
                    st.markdown(f"""
                    <div style="background:{archetype_res['bg_color']}; border-left:3.5px solid {archetype_res['badge_color']}; border-radius:8px; padding:0.5rem 0.75rem; margin-bottom:0.5rem;">
                        <div style="font-weight:800; font-size:0.86rem; color:{archetype_res['badge_color']};">{archetype_res['name']} <span style="font-size:0.75rem; opacity:0.85;">({archetype_res['affinity_score']:.0f}% Profile Affinity)</span></div>
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
                    
                fig_waterfall = create_local_xai_waterfall(base_val, predicted_math, contrib_df)
                st.plotly_chart(fig_waterfall, use_container_width=True)

            bento_row2_c1, bento_row2_c2 = st.columns([1.1, 1.1])
            
            # --- BENTO TILE 3: ACTION PLAN & RECOMMENDED STUDY TIME ---
            with bento_row2_c1:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">🛠️ Top Action Plan & Study Schedule</span>
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
                    
                fig_study = create_prescriptive_study_hours_chart(prescriptive_sol["study_hours"])
                st.plotly_chart(fig_study, use_container_width=True)

            # --- BENTO TILE 4: CERTIFIED PDF REPORT CARD EXPORT ---
            with bento_row2_c2:
                st.markdown("""
                <div class="bento-tile">
                    <div class="bento-tile-header">
                        <span class="bento-tile-title">📄 Certified Counselor Report Card</span>
                        <span style="background:#F5F3FF; color:#6D28D9; font-size:0.75rem; font-weight:700; padding:0.2rem 0.6rem; border-radius:9999px;">Verified</span>
                    </div>
                    <div style="font-size:0.84rem; color:#475569; margin-bottom:0.4rem;">Official downloadable PDF report with clinical AI prescriptions:</div>
                </div>
                """, unsafe_allow_html=True)
                
                counselor_note = st.text_area(
                    "✍️ Counselor Observations (Appears on Certificate):",
                    value="Student exhibits strong conceptual grasp in language components. Recommended enrollment in mathematics peer tutoring and weekly practice modules.",
                    height=75
                )
                
                tips = []
                if pass_prob < 50: tips.append("🚨 High Risk Alert: Immediate remedial sessions recommended in Mathematics.")
                if test_prep == "none": tips.append("📌 Test Prep Course: Statistically provides a +9.4 mark boost.")
                if reading_score < 60: tips.append("📌 Reading Focus: Enhancing reading comprehension reinforces mathematical problem solving.")
                if not tips: tips.append("🌟 Optimal Academic Standing across all subjects.")

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
                    label=f"📥 Download Certified PDF Report Card",
                    data=pdf_bytes,
                    file_name=clean_filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
                
                fig_uplift = create_intervention_uplift_chart(predicted_math, prescriptive_sol["projected_score"], prescriptive_sol["interventions"])
                st.plotly_chart(fig_uplift, use_container_width=True)

            # =========================================================
            # 🔬 DEEP-DIVE EXPANDER: PCA CLUSTER MAP, RADAR & ROADMAP
            # =========================================================
            with st.expander("🔬 Deep-Dive AI Diagnostics, 2D PCA Cluster Map & Tri-Axis Radar", expanded=False):
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
                        prep_status=test_prep
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
                            <b>{ms['week']}</b> (Target: <span style="color:#059669; font-weight:700;">{ms['target']}</span>): {ms['milestone']}
                        </div>
                        """, unsafe_allow_html=True)

# =========================================================
# TAB 2: CLASSROOM BATCH ANALYTICS & INTERVENTIONS
# =========================================================
with tab_batch:
    st.markdown("### 📂 Classroom Batch Prediction & Cohort Risk Analytics")
    st.markdown("Upload a classroom CSV or load pre-built benchmark datasets to evaluate risk distributions and generate cohort intervention clusters.")

    # 1-Click Pre-Built Cohorts
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.8rem; font-weight: 700; color: #334155; text-transform: uppercase;">⚡ Quick-Load Benchmark Classrooms:</span>
    </div>
    """, unsafe_allow_html=True)
    
    samp_c1, samp_c2, samp_c3, samp_c4, samp_c5 = st.columns(5)
    with samp_c1:
        if st.button("🌟 Balanced 50 Class", use_container_width=True):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("balanced_50")
            st.session_state["cached_batch_name"] = "Balanced Classroom Cohort (50 Students)"
            st.rerun()
    with samp_c2:
        if st.button("🏫 Grade 100 Cohort", use_container_width=True):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("large_100")
            st.session_state["cached_batch_name"] = "Grade-Wide Cohort (100 Students)"
            st.rerun()
    with samp_c3:
        if st.button("🚨 At-Risk Focus 40", use_container_width=True):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("at_risk_40")
            st.session_state["cached_batch_name"] = "High-Risk Intervention Focus (40 Students)"
            st.rerun()
    with samp_c4:
        if st.button("🏆 Honors 35 Cohort", use_container_width=True):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("honors_35")
            st.session_state["cached_batch_name"] = "Honors / AP Distinction Cohort (35 Students)"
            st.rerun()
    with samp_c5:
        if st.button("🌐 Diverse 200 Class", use_container_width=True):
            st.session_state["cached_batch"] = load_or_create_sample_cohort("mixed_200")
            st.session_state["cached_batch_name"] = "Diverse Multi-Section Cohort (200 Students)"
            st.rerun()

    with st.expander("📥 Download CSV Templates & Custom Synthetic Generator", expanded=False):
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
                st.download_button("🚨 At-Risk 40 CSV", data=buf_r40.getvalue(), file_name="sample_at_risk_40.csv", mime="text/csv", use_container_width=True)
                
            tmpl_data = generate_sample_csv_template()
            st.download_button("📋 10-Student Starter Template CSV", data=tmpl_data, file_name="starter_classroom_template.csv", mime="text/csv", use_container_width=True)
            
        with exp_c2:
            st.markdown("**Generate Custom Synthetic Classroom:**")
            gen_size = st.slider("Classroom Volume (Students):", 10, 500, 60, 5)
            gen_dist = st.selectbox("Distribution Profile:", [("balanced", "Balanced Bell Curve"), ("at_risk_focus", "At-Risk Focus"), ("honors_advanced", "Honors / AP Focus")], format_func=lambda x: x[1])[0]
            if st.button("⚡ Generate & Assess Cohort", use_container_width=True, type="primary"):
                gen_df = generate_synthetic_classroom(n_students=gen_size, cohort_type=gen_dist, seed=np.random.randint(1, 9999))
                st.session_state["cached_batch"] = gen_df
                st.session_state["cached_batch_name"] = f"Custom Synthetic Cohort ({gen_size} Students • {gen_dist.replace('_', ' ').title()})"
                st.rerun()

    uploaded_file = st.file_uploader("📤 Or Drag & Drop Custom CSV Roster:", type=["csv"])
    
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
            st.success(f"✅ Loaded `{batch_source_name}` ({len(batch_to_process)} students found).")
            if preprocessor is not None and active_model is not None:
                processed_batch, summary = process_batch_predictions(
                    batch_to_process, preprocessor, active_model, best_clf
                )
                
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1: st.metric("Total Students", summary["total_students"])
                with b2: st.metric("Class Avg Math", f"{summary['class_avg_math']:.1f} / 100")
                with b3: st.metric("Overall 3-Sub Avg", f"{summary['class_avg_overall']:.1f} / 100")
                with b4: st.metric("Class Pass Rate", f"{summary['pass_rate']}%")
                with b5: st.metric("🚨 At-Risk Count", summary["at_risk_count"])
                
                st.write("")
                
                bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs([
                    "📊 Grade & Risk Visuals",
                    "📋 Targeted Student Action Groups",
                    "🚨 Student Roster & Filters",
                    "📄 Executive PDF Export"
                ])
                
                matrix = generate_classroom_intervention_matrix(processed_batch)
                
                with bt_tab1:
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        grade_counts = processed_batch["Predicted_Grade"].value_counts().reset_index()
                        grade_counts.columns = ["Grade", "Count"]
                        fig_grade = px.pie(grade_counts, names="Grade", values="Count", hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism, title="<b>Grade Breakdown</b>")
                        fig_grade.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_grade, use_container_width=True)
                    with c_col2:
                        risk_counts = processed_batch["Risk_Tier"].value_counts().reset_index()
                        risk_counts.columns = ["Risk Tier", "Count"]
                        fig_risk = px.bar(risk_counts, x="Risk Tier", y="Count", color="Risk Tier", color_discrete_map={"Safe / Low Risk": "#10B981", "Moderate Risk": "#F59E0B", "🚨 High Academic Risk": "#EF4444"}, title="<b>Risk Tier Breakdown</b>")
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
                        st.markdown("#### 🎯 Targeted Institutional Action Strategies:")
                        st.markdown(f"""
                        <div style="background: rgba(254, 242, 242, 0.9); border-left: 4px solid #EF4444; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #DC2626;">🚨 Intensive Remedial ({matrix['summary']['high_risk_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Intensive Remedial (High Risk)']}</span>
                        </div>
                        <div style="background: rgba(254, 243, 199, 0.9); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #D97706;">🎯 Test Prep Bootcamp ({matrix['summary']['test_prep_needed_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Test Prep Bootcamp (Moderate Gap)']}</span>
                        </div>
                        <div style="background: rgba(239, 246, 255, 0.9); border-left: 4px solid #3B82F6; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #2563EB;">📚 Verbal / Reading Support ({matrix['summary']['verbal_support_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Verbal / Reading Support']}</span>
                        </div>
                        <div style="background: rgba(236, 253, 245, 0.9); border-left: 4px solid #10B981; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                            <span style="font-weight: 800; color: #059669;">🏆 Honors & Distinction Mentorship ({matrix['summary']['honors_count']} students):</span><br/>
                            <span style="font-size: 0.8rem; color: #334155;">{matrix['action_plans']['Honors / Distinction Mentorship']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                with bt_tab3:
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        search_term = st.text_input("Search Student Name / ID:", placeholder="e.g. Liam or STU-2026")
                    with f_col2:
                        risk_filter = st.selectbox("Filter Risk Tier:", ["All Risk Tiers"] + sorted(list(processed_batch["Risk_Tier"].unique())))
                    with f_col3:
                        grade_filter = st.selectbox("Filter Grade:", ["All Grades"] + sorted(list(processed_batch["Predicted_Grade"].unique())))
                        
                    filtered_df = processed_batch.copy()
                    if search_term:
                        mask = (filtered_df["student_name"].astype(str).str.contains(search_term, case=False, na=False) |
                                filtered_df["student_id"].astype(str).str.contains(search_term, case=False, na=False))
                        filtered_df = filtered_df[mask]
                    if risk_filter != "All Risk Tiers":
                        filtered_df = filtered_df[filtered_df["Risk_Tier"] == risk_filter]
                    if grade_filter != "All Grades":
                        filtered_df = filtered_df[filtered_df["Predicted_Grade"] == grade_filter]
                        
                    st.markdown(f"**Showing {len(filtered_df)} of {len(processed_batch)} students:**")
                    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                with bt_tab4:
                    class_counselor_notes = st.text_area(
                        "✍️ Cohort Counselor Observations:",
                        value="Classroom evaluated with dual ML forecasting. Prescriptive intervention clusters generated for remedial tutoring, test prep bootcamps, and distinction mentorship.",
                        height=60
                    )
                    exp_c1, exp_c2 = st.columns(2)
                    with exp_c1:
                        enriched_csv = io.StringIO()
                        filtered_df.to_csv(enriched_csv, index=False)
                        st.download_button(
                            label=f"📥 Download Processed CSV ({len(filtered_df)} Records)",
                            data=enriched_csv.getvalue(),
                            file_name="Processed_Classroom_Report.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    with exp_c2:
                        class_pdf = generate_classroom_pdf_report(
                            classroom_df=filtered_df,
                            summary=summary,
                            cohort_name=batch_source_name,
                            custom_counselor_notes=class_counselor_notes,
                            intervention_matrix=matrix
                        )
                        st.download_button(
                            label="📄 Download Classroom Executive PDF Report",
                            data=class_pdf,
                            file_name="Classroom_Executive_Summary_Report.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")

# =========================================================
# TAB 3: SIDE-BY-SIDE COMPARISON & BEFORE/AFTER SIMULATOR
# =========================================================
with tab_compare:
    cmp_sub1, cmp_sub2 = st.tabs([
        "🔄 Before vs After Intervention Growth",
        "👥 Student-to-Student Peer Benchmark"
    ])
    
    with cmp_sub1:
        st.markdown("### 🔄 Before vs After Intervention Simulator")
        st.markdown("Simulate how targeted academic and lifestyle boosters transform a student's baseline performance, grade, and pass probability:")
        
        sim_c1, sim_c2 = st.columns([1.1, 1.2])
        
        with sim_c1:
            st.markdown("#### 1. Configure Current Baseline Profile:")
            b_read = st.slider("Reading Score:", 0, 100, 58, 1, key="b_r")
            b_write = st.slider("Writing Score:", 0, 100, 54, 1, key="b_w")
            b_study = st.slider("Study Hours / Week:", 1.0, 40.0, 6.0, 0.5, key="b_s")
            b_att = st.slider("Attendance Rate (%):", 50.0, 100.0, 78.0, 1.0, key="b_a")
            b_prep = st.selectbox("Test Prep Status:", ["none", "completed"], index=0, key="b_p")
            b_tut = st.selectbox("Tutoring Support:", ["none", "peer_tutoring", "private_tutor"], index=0, key="b_t")
            
            st.markdown("#### 2. Apply Targeted Intervention Boosters:")
            boost_prep = st.checkbox("⚔️ Complete Exam Preparation Course (+9.4 pts)", value=True)
            boost_tut = st.selectbox("👥 Enroll in Tutoring Program:", ["none", "peer_tutoring (+6.5 pts)", "private_tutor (+8.2 pts)"], index=1)
            boost_study = st.slider("⏱️ Add Weekly Study Hours (+ hrs/wk):", 0.0, 15.0, 8.0, 0.5)
            boost_att = st.slider("📅 Improve Attendance Rate (+%):", 0.0, 20.0, 12.0, 1.0)
            boost_read = st.slider("📖 Improve Reading Comprehension (+ marks):", 0, 25, 12, 1)

        # Baseline Data Preparation
        base_dict = {
            "gender": "female", "race/ethnicity": "group C", "parental level of education": "some college",
            "lunch": "standard", "test preparation course": b_prep, "internet_access": "yes",
            "extracurricular_activities": "no", "tutoring_support": b_tut, "reading score": b_read,
            "writing score": b_write, "attendance_rate": b_att, "weekly_study_hours": b_study,
            "sleep_hours_per_day": 7.0, "past_failures": 1
        }
        
        # Post-Intervention Data Preparation
        tut_applied = "peer_tutoring" if "peer" in boost_tut else ("private_tutor" if "private" in boost_tut else b_tut)
        prep_applied = "completed" if boost_prep else b_prep
        post_read = min(100, b_read + boost_read)
        post_write = min(100, b_write + int(boost_read * 0.9))
        post_study = min(40.0, b_study + boost_study)
        post_att = min(100.0, b_att + boost_att)
        
        post_dict = {
            "gender": "female", "race/ethnicity": "group C", "parental level of education": "some college",
            "lunch": "standard", "test preparation course": prep_applied, "internet_access": "yes",
            "extracurricular_activities": "yes", "tutoring_support": tut_applied, "reading score": post_read,
            "writing score": post_write, "attendance_rate": post_att, "weekly_study_hours": post_study,
            "sleep_hours_per_day": 7.5, "past_failures": 0
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
                st.markdown("#### 3. Real-Time Side-by-Side Impact Analysis:")
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""
                    <div style="background:#FEF2F2; border:1.5px solid #FCA5A5; border-radius:12px; padding:1rem; text-align:center;">
                        <div style="font-size:0.8rem; font-weight:700; color:#DC2626; text-transform:uppercase;">🔴 Current Baseline</div>
                        <div style="font-size:2.2rem; font-weight:900; color:#DC2626; font-family:var(--font-heading);">{score_b:.1f}</div>
                        <div style="font-size:0.82rem; color:#475569;">Pass Prob: <b>{prob_b:.1f}%</b></div>
                        <div style="font-size:0.8rem; color:#64748B; margin-top:0.2rem;">Study: {b_study:.1f}h • Att: {b_att:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with sc2:
                    st.markdown(f"""
                    <div style="background:#ECFDF5; border:1.5px solid #6EE7B7; border-radius:12px; padding:1rem; text-align:center;">
                        <div style="font-size:0.8rem; font-weight:700; color:#047857; text-transform:uppercase;">🟢 Projected Post-Boost</div>
                        <div style="font-size:2.2rem; font-weight:900; color:#047857; font-family:var(--font-heading);">{score_p:.1f} <span style="font-size:1.1rem; color:#059669;">(+{delta_score:.1f})</span></div>
                        <div style="font-size:0.82rem; color:#475569;">Pass Prob: <b>{prob_p:.1f}%</b> (+{delta_prob:.1f}%)</div>
                        <div style="font-size:0.8rem; color:#059669; margin-top:0.2rem;">Study: {post_study:.1f}h • Att: {post_att:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                # Growth Radar Chart
                fig_ba_radar = create_before_after_radar_chart(base_dict, post_dict)
                st.plotly_chart(fig_ba_radar, use_container_width=True)

    with cmp_sub2:
        st.markdown("### 👥 Student-to-Student Peer Benchmark Comparison")
        st.markdown("Compare any two students or archetype profiles head-to-head across all competencies:")
        
        peer_col1, peer_col2 = st.columns(2)
        
        with peer_col1:
            st.markdown("#### 👤 Student Profile A:")
            a_preset = st.selectbox("Select Preset A:", ["🌟 Honors Candidate (90+)", "⚖️ Average Profile (65+)", "🚨 At-Risk Alert (<40)", "📖 Verbal Focus", "🚀 Rising Star"], index=0, key="preset_a")
            
            if "Honors" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name = 92, 95, 24.0, 98.0, "Elena Rostova (Honors)"
            elif "Average" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name = 65, 62, 12.0, 86.0, "Jordan Miller (Average)"
            elif "At-Risk" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name = 34, 30, 4.0, 62.0, "Marcus Vance (At-Risk)"
            elif "Verbal" in a_preset:
                sa_read, sa_write, sa_study, sa_att, sa_name = 88, 85, 14.0, 90.0, "Sophia Chen (Verbal Focus)"
            else:
                sa_read, sa_write, sa_study, sa_att, sa_name = 76, 74, 20.0, 96.0, "Lucas Taylor (Rising Star)"
                
            stud_a_dict = {
                "gender": "female", "race/ethnicity": "group C", "parental level of education": "bachelor's degree",
                "lunch": "standard", "test preparation course": "completed", "internet_access": "yes",
                "extracurricular_activities": "yes", "tutoring_support": "peer_tutoring", "reading score": sa_read,
                "writing score": sa_write, "attendance_rate": sa_att, "weekly_study_hours": sa_study,
                "sleep_hours_per_day": 7.5, "past_failures": 0
            }
            
        with peer_col2:
            st.markdown("#### 👤 Student Profile B:")
            b_preset = st.selectbox("Select Preset B:", ["🌟 Honors Candidate (90+)", "⚖️ Average Profile (65+)", "🚨 At-Risk Alert (<40)", "📖 Verbal Focus", "🚀 Rising Star"], index=2, key="preset_b")
            
            if "Honors" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name = 92, 95, 24.0, 98.0, "Elena Rostova (Honors)"
            elif "Average" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name = 65, 62, 12.0, 86.0, "Jordan Miller (Average)"
            elif "At-Risk" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name = 34, 30, 4.0, 62.0, "Marcus Vance (At-Risk)"
            elif "Verbal" in b_preset:
                sb_read, sb_write, sb_study, sb_att, sb_name = 88, 85, 14.0, 90.0, "Sophia Chen (Verbal Focus)"
            else:
                sb_read, sb_write, sb_study, sb_att, sb_name = 76, 74, 20.0, 96.0, "Lucas Taylor (Rising Star)"
                
            stud_b_dict = {
                "gender": "male", "race/ethnicity": "group A", "parental level of education": "some high school",
                "lunch": "free/reduced", "test preparation course": "none", "internet_access": "no",
                "extracurricular_activities": "no", "tutoring_support": "none", "reading score": sb_read,
                "writing score": sb_write, "attendance_rate": sb_att, "weekly_study_hours": sb_study,
                "sleep_hours_per_day": 6.0, "past_failures": 2
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
# TAB 4: 'WHAT-IF' ACADEMIC GOAL SIMULATOR
# =========================================================
with tab_goal:
    st.markdown("### 🎯 'What-If' Gamified Academic Goal Simulator")
    st.markdown("Select a target achievement trophy or set a custom score to reverse-engineer the exact study quest roadmap:")

    if "sim_target_val" not in st.session_state:
        st.session_state["sim_target_val"] = 85
        
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    with t_c1:
        if st.button("🥉 Bronze: Pass (50)", use_container_width=True):
            st.session_state["sim_target_val"] = 50
            st.rerun()
    with t_c2:
        if st.button("🥈 Silver: Credit (70)", use_container_width=True):
            st.session_state["sim_target_val"] = 70
            st.rerun()
    with t_c3:
        if st.button("🥇 Gold: Honor Roll (85)", use_container_width=True):
            st.session_state["sim_target_val"] = 85
            st.rerun()
    with t_c4:
        if st.button("💎 Diamond: Ivy (95)", use_container_width=True):
            st.session_state["sim_target_val"] = 95
            st.rerun()
            
    st.write("")
    col_sim1, col_sim2 = st.columns([1, 1.3])
    
    with col_sim1:
        st.markdown("#### 1. Define Baseline & Target")
        sim_target_score = st.slider("🎯 Target Math Score (0 - 100):", 50, 100, st.session_state["sim_target_val"], 1)
        
        sim_curr_read = st.slider("Current Reading Score:", 0, 100, 65, 1)
        sim_curr_write = st.slider("Current Writing Score:", 0, 100, 62, 1)
        
        cs1, cs2 = st.columns(2)
        with cs1:
            sim_study_hours = st.slider("Weekly Study Hours:", 1.0, 40.0, 12.0, 0.5)
            sim_attendance = st.slider("Attendance Rate (%):", 50.0, 100.0, 85.0, 1.0)
            sim_prep = st.selectbox("Test Prep Course", ["none", "completed"], key="sim_p")
        with cs2:
            sim_sleep = st.slider("Sleep (Hours/Day):", 4.0, 11.0, 7.5, 0.5)
            sim_tutoring = st.selectbox("Tutoring Support", ["none", "peer_tutoring", "private_tutor", "school_program"], index=0, key="sim_tut")
            sim_gender = st.selectbox("Gender", ["female", "male"], key="sim_g")
            
        sim_lunch = "standard"
        sim_internet = "yes"
        sim_edu = "some college"
        sim_failures = 0

    with col_sim2:
        st.markdown("#### 2. Simulation Results & Quest Roadmap")
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
            gap = sim_res["score_gap"]
            if gap <= 0:
                st.success(f"🎉 **Target Already Reached!** Current projected score is **{sim_res['current_predicted_math']:.1f} / 100**.")
            else:
                st.warning(f"🎯 **Target:** `{sim_target_score}` | **Current:** `{sim_res['current_predicted_math']:.1f}` | **Gap:** `+{gap:.1f} marks`")
                
            st.markdown(f"**Feasibility:** <span style='color:{sim_res['badge_color']}; font-weight:bold;'>{sim_res['feasibility']}</span>", unsafe_allow_html=True)
            st.info(sim_res["advice"])
            
            fig_traj = create_goal_trajectory_chart(sim_res["current_predicted_math"], sim_res["test_prep_benefit"], sim_target_score)
            st.plotly_chart(fig_traj, use_container_width=True)
            
            st.markdown("#### 🗺️ Multi-Lever Quest Roadmap:")
            
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if sim_prep == "none" or sim_tutoring == "none":
                    st.markdown(f"""
                    <div class="roadmap-card">
                        <b>⚔️ Quest 1: Academic Boosters</b><br/>
                        <span style="color:#0284C7; font-weight:600;">Gain: <b>+{sim_res['test_prep_benefit']:.1f} Marks</b></span><br/>
                        <span style="font-size: 0.78rem; color: #64748B;">Complete test prep & attend weekly tutoring.</span>
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
                    <b>⏱️ Quest 2: Study Habits & Attendance</b><br/>
                    • Study: <b>{sim_res['required_study_hours']:.1f} hrs/wk</b> {'(+'+str(round(study_diff, 1))+')' if study_diff > 0 else '✅'}<br/>
                    • Attendance: <b>{sim_res['required_attendance']:.0f}%</b> {'(+'+str(round(att_diff, 0))+'%)' if att_diff > 0 else '✅'}
                </div>
                """, unsafe_allow_html=True)

            with q_col2:
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>📖 Quest 3: Prerequisite Exams</b><br/>
                    • Reading: <b>{sim_res['required_reading_score']} / 100</b> (+{sim_res['reading_delta']})<br/>
                    • Writing: <b>{sim_res['required_writing_score']} / 100</b> (+{sim_res['writing_delta']})
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>🏆 Quest 4: Final Victory</b><br/>
                    Delivers projected target of <b>{sim_target_score}+ marks</b> (Verified by ML).
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 5: AI INSIGHTS & EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_xai:
    xai_sub1, xai_sub2 = st.tabs([
        "🧠 AI Decision Factor Breakdown",
        "📈 Exploratory Data Analysis (EDA)"
    ])
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    with xai_sub1:
        feat_csv = os.path.join(os.path.dirname(__file__), "artifacts", "feature_importance.csv")
        if os.path.exists(feat_csv):
            f_df = pd.read_csv(feat_csv)
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                fig_imp = create_global_importance_plotly(f_df)
                st.plotly_chart(fig_imp, use_container_width=True)
            with col_x2:
                st.markdown("#### 2. Directional Factor Attribution")
                p11 = os.path.join(plots_dir, "11_shap_directional_impact.png")
                if os.path.exists(p11):
                    st.image(p11, caption="Positive drivers (Green) vs Penalties (Red)", use_container_width=True)
            st.markdown("#### 📋 Feature Importance Table:")
            st.dataframe(f_df, use_container_width=True, hide_index=True)

    with xai_sub2:
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
# TAB 6: MODELS, STATISTICAL BENCHMARKS & ARCHITECTURE
# =========================================================
with tab_models:
    mod_sub1, mod_sub2, mod_sub3, mod_sub4 = st.tabs([
        "🏆 Model Leaderboards & ROC",
        "🔬 Statistical Uncertainty & Clustering",
        "⚙️ Model Tuning Benchmarks",
        "📖 System Architecture"
    ])
    
    with mod_sub1:
        metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.csv")
        if os.path.exists(metrics_path):
            m_df = pd.read_csv(metrics_path)
            fig_model_comp = create_model_comparison_plotly(m_df)
            st.plotly_chart(fig_model_comp, use_container_width=True)
            st.markdown("#### A. Regression Leaderboard (Continuous Score Prediction)")
            st.dataframe(m_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
            
        st.markdown("#### B. Classification Leaderboard (Pass / Fail Risk)")
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

    with mod_sub2:
        st.markdown("### 🔬 Conformal Prediction Intervals & Unsupervised Archetypes")
        st.markdown("Mathematical verification of non-parametric uncertainty margins, joint multi-subject forecasting, and behavioral cluster profiling:")
        
        stat_c1, stat_c2 = st.columns(2)
        with stat_c1:
            st.markdown("#### A. 🛡️ Conformal Prediction Residual Margins (α-Coverage):")
            calib_data = [
                {"Confidence Level": "80% Coverage", "Error Margin": f"± {uncertainty_dict.get('q80_margin', 5.8):.2f} marks", "Interpretation": "Standard confidence window"},
                {"Confidence Level": "90% Coverage", "Error Margin": f"± {uncertainty_dict.get('q90_margin', 7.3):.2f} marks", "Interpretation": "High precision academic threshold"},
                {"Confidence Level": "95% Coverage (Default)", "Error Margin": f"± {uncertainty_dict.get('q95_margin', 8.5):.2f} marks", "Interpretation": "Statistically verified certainty bound"},
                {"Confidence Level": "99% Coverage", "Error Margin": f"± {uncertainty_dict.get('q99_margin', 11.0):.2f} marks", "Interpretation": "Extreme anomaly ceiling/floor"}
            ]
            st.dataframe(pd.DataFrame(calib_data), use_container_width=True, hide_index=True)
            
            multi_csv = os.path.join(os.path.dirname(__file__), "artifacts", "multi_subject_metrics.csv")
            if os.path.exists(multi_csv):
                st.markdown("#### B. 📚 Multi-Subject Joint Forecast Model Performance:")
                st.dataframe(pd.read_csv(multi_csv), use_container_width=True, hide_index=True)
                
        with stat_c2:
            st.markdown("#### C. 🧬 Discovered Behavioral Student Archetypes:")
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
        st.markdown("### 📖 Multi-Engine Machine Learning Architecture")
        st.markdown("""
        ```
        1. 14-Feature Input (Single Student Profile or Classroom Bulk CSV)
           └── Preprocessed & Engineered via 38-Feature Synergy Pipeline (RobustScaler + OneHotEncoder)
        
        2. Production ML & Statistical Engines:
           ├── Regression Champion: Optimized ElasticNet (R² 90.07%, MAE ±3.47 marks)
           ├── Classification Champion: Support Vector Classifier (Accuracy 97.00%, ROC-AUC 0.9853)
           ├── Uncertainty Quantifier: Conformal Prediction 95% Confidence Bounds (±8.50 marks)
           ├── Behavioral Archetype Clusterer: K-Means (k=4) + 2D PCA Decomposition
           ├── Tri-Axis Multi-Subject Engine: Joint Math, Reading & Writing Regressor
           ├── Side-by-Side Simulator: Before vs After Growth & Head-to-Head Peer Comparison
           ├── Prescriptive Diagnostic Engine: 6-dimensional clinical weakness detection
           └── Explainable AI (XAI): Permutation Importance & SHAP Waterfall Attributions
        
        3. Deliverables:
           ├── Exact Point Score with 95% Confidence Interval [Lower – Upper]
           ├── Before vs After Competency Growth Radar & Head-to-Head Peer Charts
           ├── Behavioral Archetype Profile & 2D PCA Cohort Position Map
           ├── Pass Probability & Early Risk Tier
           ├── 12-Week Growth Milestones & Prescriptive Study Schedule
           └── Verified PDF Performance Certificate & Classroom Executive Report
        ```
        """)
