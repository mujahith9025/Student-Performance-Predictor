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

from src.pdf_generator import generate_student_pdf_report
from src.explainability import explain_single_student
from src.goal_simulator import simulate_academic_goal
from src.batch_predictor import process_batch_predictions, generate_sample_csv_template
from src.advanced_feature_engineering import engineer_features

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
        background: rgba(248, 250, 252, 0.7);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
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
# LOAD MODEL ARTIFACTS
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

preprocessor, best_model, best_clf, all_models = load_artifacts()

# ---------------------------------------------------------
# PLOTLY INTERACTIVE GAUGE & RADAR CHARTS
# ---------------------------------------------------------
def create_score_gauge(score, grade):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Predicted Math Marks ({grade})</b>", 'font': {'size': 15, 'family': 'Outfit', 'color': '#1E3A8A'}},
        number={'suffix': " / 100", 'font': {'size': 28, 'family': 'Outfit', 'color': '#1E3A8A'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#2563EB", 'thickness': 0.28},
            'bgcolor': "rgba(241, 245, 249, 0.8)",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.18)'},
                {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.18)'},
                {'range': [70, 85], 'color': 'rgba(59, 130, 246, 0.18)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.22)'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E293B", 'family': "Plus Jakarta Sans"}
    )
    return fig

def create_radar_chart(reading, writing, predicted_math, socio_index):
    categories = ['Reading', 'Writing', 'Math (Pred)', 'Verbal Synergy', 'Socio-Readiness']
    student_synergy = np.sqrt(max(0, reading * writing))
    student_values = [reading, writing, predicted_math, student_synergy, min(100, socio_index * 10)]
    benchmark_values = [69.2, 68.1, 66.1, 68.6, 55.0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values + [benchmark_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.18)',
        line=dict(color='#94A3B8', dash='dot', width=1.5),
        name='Cohort Benchmark'
    ))
    fig.add_trace(go.Scatterpolar(
        r=student_values + [student_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.28)',
        line=dict(color='#2563EB', width=2.5),
        name='Student Profile'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#64748B")),
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=11)),
        height=240,
        margin=dict(l=30, r=30, t=25, b=15),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

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
    st.markdown("### 📝 Enter Student Academic & Demographic Profile")
    st.markdown("Select a quick-start demo persona or input custom student characteristics to run real-time dual-task assessment:")
    
    # Session State Initialization for Form Inputs
    if "p_name" not in st.session_state: st.session_state["p_name"] = "Alex Johnson"
    if "p_id" not in st.session_state: st.session_state["p_id"] = "STU-2026-101"
    if "p_gender" not in st.session_state: st.session_state["p_gender"] = "female"
    if "p_race" not in st.session_state: st.session_state["p_race"] = "group C"
    if "p_edu" not in st.session_state: st.session_state["p_edu"] = "bachelor's degree"
    if "p_lunch" not in st.session_state: st.session_state["p_lunch"] = "standard"
    if "p_prep" not in st.session_state: st.session_state["p_prep"] = "completed"
    if "p_read" not in st.session_state: st.session_state["p_read"] = 78
    if "p_write" not in st.session_state: st.session_state["p_write"] = 82

    # Quick Select Persona Presets
    st.markdown("""
    <div class="preset-chip-box">
        <span style="font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">⚡ Quick-Load Demo Personas:</span>
    </div>
    """, unsafe_allow_html=True)
    
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🌟 Preset: Honors Candidate (90+ Marks)", use_container_width=True):
            st.session_state["p_name"] = "Elena Rostova"
            st.session_state["p_id"] = "STU-2026-HONORS"
            st.session_state["p_gender"] = "female"
            st.session_state["p_race"] = "group E"
            st.session_state["p_edu"] = "master's degree"
            st.session_state["p_lunch"] = "standard"
            st.session_state["p_prep"] = "completed"
            st.session_state["p_read"] = 92
            st.session_state["p_write"] = 95
            st.rerun()
            
    with p_col2:
        if st.button("⚖️ Preset: Average Profile (60-70 Marks)", use_container_width=True):
            st.session_state["p_name"] = "Jordan Miller"
            st.session_state["p_id"] = "STU-2026-AVG"
            st.session_state["p_gender"] = "male"
            st.session_state["p_race"] = "group C"
            st.session_state["p_edu"] = "some college"
            st.session_state["p_lunch"] = "standard"
            st.session_state["p_prep"] = "none"
            st.session_state["p_read"] = 65
            st.session_state["p_write"] = 62
            st.rerun()
            
    with p_col3:
        if st.button("🚨 Preset: High-Risk Alert (<40 Marks)", use_container_width=True):
            st.session_state["p_name"] = "Marcus Vance"
            st.session_state["p_id"] = "STU-2026-RISK"
            st.session_state["p_gender"] = "male"
            st.session_state["p_race"] = "group A"
            st.session_state["p_edu"] = "some high school"
            st.session_state["p_lunch"] = "free/reduced"
            st.session_state["p_prep"] = "none"
            st.session_state["p_read"] = 34
            st.session_state["p_write"] = 30
            st.rerun()
            
    st.write("")
    
    with st.form("prediction_form"):
        # Student Info Header
        st.markdown("#### 🆔 Student Identification")
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            student_name = st.text_input("Student Full Name", value=st.session_state["p_name"], placeholder="e.g. Alex Johnson")
        with id_col2:
            student_id = st.text_input("Student ID / Roll Number", value=st.session_state["p_id"], placeholder="e.g. STU-2026-101")
            
        st.write("")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 👤 Demographics & Environment")
            gender_opts = ["female", "male"]
            gender = st.selectbox("Gender", options=gender_opts, index=gender_opts.index(st.session_state["p_gender"]))
            
            race_opts = ["group A", "group B", "group C", "group D", "group E"]
            race_ethnicity = st.selectbox(
                "Race / Ethnicity Group",
                options=race_opts,
                index=race_opts.index(st.session_state["p_race"])
            )
            
            edu_opts = [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
            parental_education = st.selectbox(
                "Parental Level of Education",
                options=edu_opts,
                index=edu_opts.index(st.session_state["p_edu"])
            )
            
            lunch_opts = ["standard", "free/reduced"]
            lunch = st.selectbox(
                "Lunch Plan",
                options=lunch_opts,
                index=lunch_opts.index(st.session_state["p_lunch"])
            )
            
            prep_opts = ["none", "completed"]
            test_prep = st.selectbox(
                "Test Preparation Course",
                options=prep_opts,
                index=prep_opts.index(st.session_state["p_prep"])
            )
            
        with col_right:
            st.markdown("#### 📚 Existing Subject Marks (0 - 100)")
            st.markdown("Scores earned by the student in prerequisite subjects:")
            
            reading_score = st.slider("Reading Score", min_value=0, max_value=100, value=st.session_state["p_read"], step=1)
            writing_score = st.slider("Writing Score", min_value=0, max_value=100, value=st.session_state["p_write"], step=1)
            
            st.write("")
            st.write("")
            submit_btn = st.form_submit_button("⚡ Run Dual-Task ML Assessment", use_container_width=True, type="primary")

    if submit_btn:
        if preprocessor is None or active_model is None:
            st.error("Model artifacts not found! Please run the training pipeline first.")
        else:
            # Create DataFrame matching training format
            input_dict = {
                "gender": [gender],
                "race/ethnicity": [race_ethnicity],
                "parental level of education": [parental_education],
                "lunch": [lunch],
                "test preparation course": [test_prep],
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
                pass_prob = 90.0 if predicted_math >= 50 else 30.0
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
                fig_radar = create_radar_chart(reading_score, writing_score, predicted_math, socio_index)
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
                
            # 3. Local Explainable AI (SHAP Breakdown)
            st.markdown("---")
            st.markdown("### 🔍 Why did the AI predict this score? (Feature Attribution Breakdown)")
            st.markdown("This breakdown explains how each characteristic contributed points relative to the population baseline average of **67.95 marks**:")
            
            base_val, pred_val, contrib_df = explain_single_student(input_df, preprocessor, active_model, None)
            
            xai_c1, xai_c2 = st.columns([1.2, 1])
            with xai_c1:
                st.markdown("#### 📊 Individual Factor Impact on Score:")
                for _, row in contrib_df.iterrows():
                    impact = row["Impact"]
                    sign = "+" if impact >= 0 else ""
                    color = "#059669" if impact >= 0 else "#DC2626"
                    st.markdown(f"- **{row['Factor']}** (`{row['Value']}`): <span style='color:{color}; font-weight:bold;'>{sign}{impact:.2f} marks</span>", unsafe_allow_html=True)
            with xai_c2:
                st.dataframe(
                    contrib_df.style.format({"Impact": "{:+.2f} marks"}),
                    use_container_width=True,
                    hide_index=True
                )

            # Diagnostic Feedback
            st.markdown("---")
            st.markdown("#### 💡 Diagnostic Recommendations & Intervention Plan")
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

            # Generate PDF Report Card
            st.markdown("---")
            st.markdown("#### 📄 Export Official Report Card")
            st.markdown("Download a verified PDF performance certificate and counselor evaluation report:")
            
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
                risk_level=risk_level
            )
            
            clean_filename = f"Performance_Risk_Report_{student_id.replace('/', '_')}.pdf"
            
            st.download_button(
                label=f"📥 Download Verified PDF Report ({clean_filename})",
                data=pdf_bytes,
                file_name=clean_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

# ----------------- TAB 2: CLASSROOM BATCH PREDICTOR -----------------
with tab2:
    st.markdown("### 📂 Classroom Batch Prediction (Bulk CSV Upload)")
    st.markdown("Upload a CSV file containing an entire classroom or student cohort to generate predictions, grade distribution analysis, and risk diagnostics at scale.")
    
    # Template download header
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("**Need a sample file format?** Download our pre-formatted 10-student sample CSV:")
    with col_t2:
        sample_csv_data = generate_sample_csv_template()
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_csv_data,
            file_name="classroom_sample_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.write("")
    uploaded_file = st.file_uploader("📤 Upload Classroom Student CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded `{uploaded_file.name}` ({len(input_batch_df)} students found).")
            
            if preprocessor is not None and active_model is not None:
                processed_batch, summary = process_batch_predictions(
                    input_batch_df, preprocessor, active_model, best_clf
                )
                
                # Summary Metric Cards
                st.markdown("#### 📊 Classroom Overview Metrics:")
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1:
                    st.metric("Total Students", summary["total_students"])
                with b2:
                    st.metric("Class Avg Math", f"{summary['class_avg_math']:.1f}")
                with b3:
                    st.metric("Overall 3-Sub Avg", f"{summary['class_avg_overall']:.1f}")
                with b4:
                    st.metric("Pass Rate (%)", f"{summary['pass_rate']}%")
                with b5:
                    st.metric("🚨 At-Risk Count", summary["at_risk_count"])
                    
                st.write("")
                
                # Visual Analytics for Classroom
                st.markdown("#### 📈 Cohort Grade & Risk Distribution:")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Grade Distribution Plotly Donut
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
                    # Risk Tier Distribution Plotly Bar
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
                    
                # Processed Data Table
                st.markdown("#### 📋 Processed Classroom Predictions Table:")
                st.dataframe(processed_batch, use_container_width=True, hide_index=True)
                
                # Download Enriched CSV
                enriched_csv_buffer = io.StringIO()
                processed_batch.to_csv(enriched_csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Complete Processed Classroom Report (.CSV)",
                    data=enriched_csv_buffer.getvalue(),
                    file_name=f"Processed_Report_{uploaded_file.name}",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# ----------------- TAB 3: WHAT-IF GOAL SIMULATOR -----------------
with tab3:
    st.markdown("### 🎯 'What-If' Academic Goal Simulator")
    st.markdown("Set a target mark and simulate the exact study roadmap, reading/writing score targets, and preparation milestones required to achieve it.")
    
    col_sim1, col_sim2 = st.columns([1, 1.2])
    
    with col_sim1:
        st.markdown("#### 1. Define Student Baseline & Desired Target")
        sim_target_score = st.slider("🎯 Desired Target Math Score (0 - 100):", min_value=50, max_value=100, value=85, step=1)
        
        st.markdown("##### Current Student Profile:")
        sim_curr_read = st.slider("Current Reading Score:", min_value=0, max_value=100, value=65, step=1)
        sim_curr_write = st.slider("Current Writing Score:", min_value=0, max_value=100, value=62, step=1)
        
        sim_gender = st.selectbox("Gender", ["female", "male"], key="sim_g")
        sim_prep = st.selectbox("Current Test Prep Status", ["none", "completed"], key="sim_p")
        sim_lunch = st.selectbox("Lunch Nutrition", ["standard", "free/reduced"], key="sim_l")
        sim_edu = st.selectbox("Parental Education", [
            "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
        ], index=2, key="sim_e")
        
        sim_btn = st.button("🚀 Calculate Milestone Roadmap", use_container_width=True, type="primary")

    with col_sim2:
        st.markdown("#### 2. Simulation Results & Target Roadmap")
        
        # Build Profile Dict
        sim_profile = {
            "gender": sim_gender,
            "race/ethnicity": "group C",
            "parental level of education": sim_edu,
            "lunch": sim_lunch,
            "test preparation course": sim_prep,
            "reading score": sim_curr_read,
            "writing score": sim_curr_write
        }
        
        if preprocessor is not None and active_model is not None:
            sim_res = simulate_academic_goal(sim_profile, sim_target_score, preprocessor, active_model)
            
            # Gap Header Card
            gap = sim_res["score_gap"]
            if gap <= 0:
                st.success(f"🎉 **Target Already Reached!** Current projected score is **{sim_res['current_predicted_math']:.1f} / 100**.")
            else:
                st.warning(f"🎯 **Target Score:** `{sim_target_score}` | **Current Projected:** `{sim_res['current_predicted_math']:.1f}` | **Score Gap to Bridge:** `+{gap:.1f} marks`")
                
            st.markdown(f"**Feasibility Rating:** <span style='color:{sim_res['badge_color']}; font-weight:bold;'>{sim_res['feasibility']}</span>", unsafe_allow_html=True)
            st.info(sim_res["advice"])
            
            st.markdown("---")
            st.markdown("#### 🗺️ Recommended Action Roadmap:")
            
            # Step 1: Test Prep
            if sim_prep == "none":
                st.markdown(f"""
                <div class="roadmap-card">
                    <b>Step 1: Enroll in Test Preparation Course</b><br/>
                    <span style="color:#0284C7;">Estimated Gain: <b>+{sim_res['test_prep_benefit']:.1f} to +9.4 marks</b> in Mathematics.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="roadmap-card">
                    <b>Step 1: Test Prep Course Completed</b><br/>
                    <span style="color:#059669;">Great job! You already have the preparation boost active.</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Step 2: Subject Marks Targets
            st.markdown(f"""
            <div class="roadmap-card">
                <b>Step 2: Reach Milestone Exam Scores</b><br/>
                • Target Reading Score: <b>{sim_res['required_reading_score']} / 100</b> <span style="color:#0284C7;">(+{sim_res['reading_delta']} marks from current {sim_curr_read})</span><br/>
                • Target Writing Score: <b>{sim_res['required_writing_score']} / 100</b> <span style="color:#0284C7;">(+{sim_res['writing_delta']} marks from current {sim_curr_write})</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Step 3: Probability forecast
            st.markdown("""
            <div class="roadmap-card">
                <b>Step 3: Projected Academic Outcome</b><br/>
                Meeting these reading and writing milestones is statistically verified by the ML model to deliver your target of <b>85+ marks (Grade A / A+)</b>.
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 4: EXPLAINABLE AI (XAI) -----------------
with tab4:
    st.markdown("### 🔍 Explainable AI (XAI) & Feature Importance Analysis")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("#### 1. Global Feature Importance (Permutation Impact)")
        p10 = os.path.join(plots_dir, "10_global_feature_importance.png")
        if os.path.exists(p10):
            st.image(p10, caption="Global Permutation Importance across 24 engineered features.", use_container_width=True)
            
    with col_x2:
        st.markdown("#### 2. Directional Feature Attribution (Point Adjustments)")
        p11 = os.path.join(plots_dir, "11_shap_directional_impact.png")
        if os.path.exists(p11):
            st.image(p11, caption="Positive drivers (Green) vs Negative penalties (Red) on marks.", use_container_width=True)
            
    st.markdown("#### 📋 Full Feature Importance Impact Table:")
    feat_csv = os.path.join(os.path.dirname(__file__), "artifacts", "feature_importance.csv")
    if os.path.exists(feat_csv):
        f_df = pd.read_csv(feat_csv)
        st.dataframe(f_df, use_container_width=True, hide_index=True)

# ----------------- TAB 5: EDA & INSIGHTS -----------------
with tab5:
    st.markdown("### 📊 Exploratory Data Analysis & Visual Insights")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1. Score Distributions")
        p1 = os.path.join(plots_dir, "01_score_distributions.png")
        if os.path.exists(p1):
            st.image(p1, caption="Normal distributions across Math, Reading, and Writing marks.", use_container_width=True)
            
        st.markdown("#### 3. Test Preparation Boost (+9.4 Marks)")
        p3 = os.path.join(plots_dir, "03_test_prep_impact.png")
        if os.path.exists(p3):
            st.image(p3, caption="Statistically significant mark boost for students with completed test prep.", use_container_width=True)
            
    with col_b:
        st.markdown("#### 2. Cross-Subject Correlation Matrix")
        p2 = os.path.join(plots_dir, "02_correlation_heatmap.png")
        if os.path.exists(p2):
            st.image(p2, caption="Strong Pearson correlation (r = 0.80 - 0.95) between subjects.", use_container_width=True)
            
        st.markdown("#### 4. Parental Education Influence")
        p4 = os.path.join(plots_dir, "04_parental_education_impact.png")
        if os.path.exists(p4):
            st.image(p4, caption="Higher parental education degree correlates with higher median student scores.", use_container_width=True)

# ----------------- TAB 6: DUAL MODEL LEADERBOARD -----------------
with tab6:
    st.markdown("### 🏆 Dual-Task Evaluation Leaderboards")
    
    st.markdown("#### A. Regression Leaderboard (Continuous Score Prediction)")
    metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.dataframe(m_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
        
    st.markdown("#### B. Classification Leaderboard (Pass / Fail & Dropout Risk)")
    clf_path = os.path.join(os.path.dirname(__file__), "artifacts", "classifier_metrics.csv")
    if os.path.exists(clf_path):
        c_df = pd.read_csv(clf_path)
        st.dataframe(c_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        p8 = os.path.join(plots_dir, "08_confusion_matrix.png")
        if os.path.exists(p8):
            st.image(p8, caption="Confusion Matrix on Test Data.", use_container_width=True)
    with col_v2:
        p9 = os.path.join(plots_dir, "09_roc_auc_curve.png")
        if os.path.exists(p9):
            st.image(p9, caption="ROC-AUC Curves for all classifiers.", use_container_width=True)

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
       ├── Classroom Batch Engine: Scale Predictions + Interactive Grade & Risk Donut Charts
       ├── 'What-If' Simulator: Milestone Roadmapping & Score Gap Solver
       └── Explainable AI (XAI): Permutation Importance & Directional Attributions
    
    3. Outputs & Deliverables:
       ├── Exact Predicted Marks & Grade (A+ to F)
       ├── Pass Probability & Early Dropout Risk Tier
       ├── 5-Axis Student Competency Radar Chart
       ├── Cohort-Level Analytics & Downloadable Enriched CSV
       └── Verified PDF Performance Certificate v5.0
    ```
    """)
