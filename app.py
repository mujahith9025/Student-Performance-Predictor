import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="Student Performance Predictor | Multi-Subject AI & Habit Radar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Prepend both project root and src directory to sys.path
for path_entry in [SRC_DIR, PROJECT_ROOT]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

try:
    from data_loader import (
        load_subject_data,
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SUBJECT_METADATA
    )
except ImportError:
    from src.data_loader import (
        load_subject_data,
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SUBJECT_METADATA
    )

try:
    from explainability import compute_shap_waterfall
except ImportError:
    from src.explainability import compute_shap_waterfall

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
ALL_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "all_trained_models.pkl")
QUANTILE_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "quantile_models.pkl")
SUBJECT_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "subject_models.pkl")
SUBJECT_METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "subject_metrics.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")
SAMPLE_BATCH_PATH = os.path.join(PROJECT_ROOT, "data", "sample_batch_test.csv")

# Standard Z-scores for Confidence Intervals
Z_SCORES = {
    "80%": 1.282,
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576
}

# Custom CSS for rich, modern aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    
    .subject-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .score-badge {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38BDF8;
    }
    
    .interval-badge {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FBBF24;
    }
    
    .grade-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .grade-A { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .grade-B { background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }
    .grade-C { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .grade-D { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }
    .grade-F { background-color: rgba(220, 38, 38, 0.3); color: #FCA5A5; border: 1px solid #DC2626; }
    
    .eng-badge {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_subject_artifacts():
    """Load cached multi-subject models bundle, quantile models, and evaluation metrics."""
    if os.path.exists(SUBJECT_MODELS_PATH) and os.path.exists(SUBJECT_METRICS_PATH):
        subject_models = joblib.load(SUBJECT_MODELS_PATH)
        with open(SUBJECT_METRICS_PATH, "r", encoding="utf-8") as f:
            subject_metrics = json.load(f)
        return subject_models, subject_metrics
    elif os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        best_model = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        subject_models = {"General Academics": {"Linear Regression": best_model}}
        subject_metrics = {"General Academics": metrics}
        return subject_models, subject_metrics
    return {}, {}

@st.cache_data
def get_cached_subject_data(subject_name: str):
    """Load cached dataset for a specific academic subject."""
    return load_subject_data(subject_name)

def compute_grade_and_status(score: float):
    """Computes letter grade and qualitative standing."""
    if score >= 90:
        return "A+", "grade-A", "🌟 Outstanding Performance (Top Tier)"
    elif score >= 80:
        return "A", "grade-A", "🎯 Excellent (High Standing)"
    elif score >= 70:
        return "B", "grade-B", "👍 Good Performance (Above Average)"
    elif score >= 60:
        return "C", "grade-C", "⚖️ Satisfactory (Average)"
    elif score >= 50:
        return "D", "grade-D", "⚠️ Borderline / Needs Improvement"
    else:
        return "F", "grade-F", "🚨 At Risk (Action Required)"

def get_interval_bounds(pred_score, conf_str, residual_std=2.075):
    """Calculates lower and upper prediction bounds with margin of error."""
    z = Z_SCORES.get(conf_str, 1.960)
    margin = z * residual_std
    lower = max(10.0, float(pred_score - margin))
    upper = min(100.0, float(pred_score + margin))
    return lower, upper, margin

# Load trained multi-subject artifacts
subject_models_bundle, all_subject_metrics = load_all_subject_artifacts()

if not subject_models_bundle:
    st.error("⚠️ Trained models not found! Please run `python src/train.py` first to train and serialize the multi-subject models.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=110)
    st.title("🎓 Student AI Predictor")
    st.caption("Multi-Subject ML, SHAP & Habit Radar")
    
    st.markdown("---")
    
    # 1. Multi-Subject Switcher
    st.markdown("### 📚 Academic Subject Switcher")
    subject_options = list(SUBJECT_METADATA.keys())
    selected_subject = st.selectbox(
        "Choose Academic Subject Discipline:",
        options=subject_options,
        index=0,
        help="Switch between General Academics, Mathematics, Science, and Humanities to load tailored models and datasets."
    )
    
    active_subject_meta = SUBJECT_METADATA.get(selected_subject, SUBJECT_METADATA["General Academics"])
    active_subject_metrics = all_subject_metrics.get(selected_subject, all_subject_metrics.get("General Academics", {}))
    active_subject_models = subject_models_bundle.get(selected_subject, subject_models_bundle.get("General Academics", {}))
    
    st.caption(f"🏷️ **Discipline:** {active_subject_meta['badge']}")
    st.caption(f"💡 **Key Focus:** {active_subject_meta['focus']}")
    
    st.markdown("---")
    
    # 2. Model Selector
    available_model_names = list(active_subject_models.keys())
    default_best = active_subject_metrics.get("best_model_name", "Linear Regression")
    default_idx = available_model_names.index(default_best) if default_best in available_model_names else 0
    
    st.markdown("### 🤖 Select ML Algorithm")
    selected_model_name = st.selectbox(
        "Choose Model for Inference:",
        options=available_model_names,
        index=default_idx,
        help="Select between classic linear models, random forests, or Advanced Gradient Boosters (XGBoost, LightGBM, GBDT, Stacking)"
    )
    
    active_model = active_subject_models.get(selected_model_name)
    
    st.markdown("---")
    
    # 3. Confidence Level Slider
    st.markdown("### 🎯 Prediction Confidence Level")
    confidence_level_str = st.select_slider(
        "Select Confidence Interval Band:",
        options=["80%", "90%", "95%", "99%"],
        value="95%",
        help="95% is the statistical gold standard, meaning 95 out of 100 students with these habits will fall within the estimated range."
    )
    
    residual_std = active_subject_metrics.get("confidence_intervals", {}).get("residual_std", 2.075)
    cur_z = Z_SCORES[confidence_level_str]
    cur_margin = cur_z * residual_std
    st.caption(f"📌 Margin of Error: **±{cur_margin:.2f} points** ($Z={cur_z}$)")
    
    st.markdown("---")
    st.markdown(f"### 🏆 {selected_subject} Stats")
    model_bench = active_subject_metrics.get("all_model_benchmarks", {}).get(selected_model_name, active_subject_metrics.get("best_model_metrics", {}))
    if model_bench:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test R²", f"{model_bench.get('test_r2', 0.988) * 100:.1f}%")
        with col2:
            st.metric("MAE", f"±{model_bench.get('test_mae', 1.65):.2f}")

# Main Header Banner
st.markdown(f"""
<div class="main-header">
    <h1>{active_subject_meta['icon']} Student Academic Performance Predictor</h1>
    <p>Predict outcomes for <b>{selected_subject}</b> using <b>Student Habit Radar Charts</b>, SHAP Explainability, and Confidence Intervals (80%–99%).</p>
    <div class="subject-pill">{active_subject_meta['icon']} Active Subject: {selected_subject} ({active_subject_meta['badge']})</div>
</div>
""", unsafe_allow_html=True)

# Multi-Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Student Predictor",
    "📂 Batch CSV Prediction",
    "📊 Benchmarks & Multi-Subject Matrix",
    "📈 Dataset Explorer (EDA & Habit Footprint)"
])

# -------------------------------------------------------------
# TAB 1: Single Student Predictor
# -------------------------------------------------------------
with tab1:
    st.subheader(f"{active_subject_meta['icon']} {selected_subject} — Student Outcome Prediction")
    st.write(f"Inference Model: **`{selected_model_name}`** | Confidence Level: **`{confidence_level_str}`** (Margin ±{cur_margin:.2f} pts).")
    
    with st.form(key=f"student_prediction_form_{selected_subject}"):
        col_input1, col_input2 = st.columns([1, 1], gap="large")
        
        with col_input1:
            st.markdown("#### 📚 Academic & Study Habits")
            hours_studied = st.slider(
                "⏱️ Daily Hours Studied",
                min_value=1,
                max_value=9,
                value=5,
                help="Average daily study time outside school (1 to 9 hours)"
            )
            
            previous_score = st.slider(
                "📝 Previous Exam Score",
                min_value=40,
                max_value=100,
                value=75,
                help="Previous test score / marks obtained (40 to 100)"
            )
            
            sample_papers = st.slider(
                "📄 Practice Papers Practiced",
                min_value=0,
                max_value=10,
                value=3,
                help="Number of mock exam papers completed"
            )
            
        with col_input2:
            st.markdown("#### 🌿 Health & Extracurriculars")
            sleep_hours = st.slider(
                "😴 Daily Sleep Hours",
                min_value=4,
                max_value=10,
                value=7,
                help="Average daily sleep duration"
            )
            
            extracurricular = st.radio(
                "⚽ Participates in Extracurricular Activities?",
                options=["Yes", "No"],
                horizontal=True,
                help="Sports, clubs, arts, volunteering, etc."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button(f"🚀 Predict {selected_subject} Score & Habit Radar", type="primary", width="stretch")

    # If user clicked calculate, compute and store in session_state
    if predict_btn:
        raw_input_data = pd.DataFrame([{
            "Hours_Studied": hours_studied,
            "Previous_Score": previous_score,
            "Sleep_Hours": sleep_hours,
            "Sample_Question_Papers_Practiced": sample_papers,
            "Extracurricular_Activities": extracurricular
        }])

        engineered_input = engineer_features(raw_input_data)
        pred_score = float(active_model.predict(engineered_input)[0])
        pred_score = max(10.0, min(100.0, pred_score))
        letter_grade, grade_css, standing_desc = compute_grade_and_status(pred_score)

        # Confidence bounds
        lower_bound, upper_bound, margin = get_interval_bounds(pred_score, confidence_level_str, residual_std)
        lower_grade, _, _ = compute_grade_and_status(lower_bound)
        upper_grade, _, _ = compute_grade_and_status(upper_bound)

        # Compute SHAP Waterfall decomposition
        shap_res = compute_shap_waterfall(active_model, raw_input_data)

        # What-if scenario boost calculations
        boost_study = raw_input_data.copy()
        boost_study["Hours_Studied"] = min(9, hours_studied + 2)
        boost_score_study = float(active_model.predict(engineer_features(boost_study))[0])
        delta_study = max(0.0, boost_score_study - pred_score)
        
        boost_papers = raw_input_data.copy()
        boost_papers["Sample_Question_Papers_Practiced"] = min(10, sample_papers + 3)
        boost_score_papers = float(active_model.predict(engineer_features(boost_papers))[0])
        delta_papers = max(0.0, boost_score_papers - pred_score)

        # Normalized Radar Dimensions (0 - 100%)
        dim_study = min(100.0, (hours_studied / 9.0) * 100.0)
        dim_foundation = min(100.0, max(0.0, ((previous_score - 40.0) / 60.0) * 100.0))
        dim_practice = min(100.0, (sample_papers / 10.0) * 100.0)
        dim_sleep = min(100.0, max(0.0, 100.0 - abs(sleep_hours - 7.5) * 20.0))
        dim_effort = min(100.0, (engineered_input["Study_Effort_Score"].iloc[0] / 9.4) * 100.0)
        dim_ec = 85.0 if extracurricular == "Yes" else 35.0
        
        habit_balance_score = round((dim_study + dim_foundation + dim_practice + dim_sleep + dim_effort + dim_ec) / 6.0, 1)

        st.session_state["single_prediction"] = {
            "subject": selected_subject,
            "model_used": selected_model_name,
            "conf_level": confidence_level_str,
            "raw_input": raw_input_data,
            "engineered_input": engineered_input,
            "pred_score": pred_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "margin": margin,
            "letter_grade": letter_grade,
            "lower_grade": lower_grade,
            "upper_grade": upper_grade,
            "grade_css": grade_css,
            "standing_desc": standing_desc,
            "shap_res": shap_res,
            "hours_studied": hours_studied,
            "sleep_hours": sleep_hours,
            "sample_papers": sample_papers,
            "delta_study": delta_study,
            "delta_papers": delta_papers,
            "radar_dims": {
                "Study Time": dim_study,
                "Exam Foundation": dim_foundation,
                "Mock Practice": dim_practice,
                "Sleep Balance": dim_sleep,
                "Study Effort": dim_effort,
                "Extracurriculars": dim_ec
            },
            "habit_balance_score": habit_balance_score
        }

    # Render results only if prediction has been submitted
    if "single_prediction" in st.session_state:
        res = st.session_state["single_prediction"]
        
        # If user changed confidence level or subject, recompute bounds dynamically
        res_lower, res_upper, res_margin = get_interval_bounds(res["pred_score"], confidence_level_str, residual_std)
        
        st.markdown("---")
        st.markdown(f"### 🎯 Prediction Results for {res.get('subject', selected_subject)} (via `{res.get('model_used', selected_model_name)}`)")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns([1, 1, 1, 1])
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 600;">POINT ESTIMATE</span>
                <div class="score-badge">{res['pred_score']:.1f}<span style="font-size: 1.1rem; color: #94A3B8;">/100</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 600;">{confidence_level_str} CONFIDENCE RANGE</span>
                <div class="interval-badge">[{res_lower:.1f} — {res_upper:.1f}]</div>
                <span style="font-size: 0.8rem; color: #94A3B8;">Margin: ±{res_margin:.1f} pts</span>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 600;">EXPECTED GRADE</span><br>
                <span class="grade-pill {res['grade_css']}">{res['letter_grade']}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col4:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 600;">ACADEMIC STANDING</span>
                <div style="font-size: 1.05rem; font-weight: 700; margin-top: 0.8rem; color: #E2E8F0;">
                    {res['standing_desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Visual Confidence Range Chart
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_gauge = go.Figure()
        fig_gauge.add_vrect(x0=10, x1=50, fillcolor="#DC2626", opacity=0.15, layer="below", line_width=0, annotation_text="F (Fail)", annotation_position="top left")
        fig_gauge.add_vrect(x0=50, x1=60, fillcolor="#EF4444", opacity=0.15, layer="below", line_width=0, annotation_text="D", annotation_position="top left")
        fig_gauge.add_vrect(x0=60, x1=70, fillcolor="#F59E0B", opacity=0.15, layer="below", line_width=0, annotation_text="C", annotation_position="top left")
        fig_gauge.add_vrect(x0=70, x1=80, fillcolor="#3B82F6", opacity=0.15, layer="below", line_width=0, annotation_text="B", annotation_position="top left")
        fig_gauge.add_vrect(x0=80, x1=100, fillcolor="#10B981", opacity=0.15, layer="below", line_width=0, annotation_text="A / A+", annotation_position="top left")
        
        fig_gauge.add_trace(go.Scatter(
            x=[res_lower, res_upper],
            y=[1, 1],
            mode="lines",
            line=dict(color="#FBBF24", width=10),
            name=f"{confidence_level_str} Confidence Interval",
            hoverinfo="x+name"
        ))
        
        fig_gauge.add_trace(go.Scatter(
            x=[res["pred_score"]],
            y=[1],
            mode="markers+text",
            marker=dict(color="#38BDF8", size=20, symbol="diamond-dot", line=dict(color="#ffffff", width=2)),
            text=[f"Predicted: {res['pred_score']:.1f}"],
            textposition="top center",
            name="Point Estimate",
            hoverinfo="x+name"
        ))
        
        fig_gauge.update_layout(
            title=f"🎯 {selected_subject} Prediction Range: [{res_lower:.1f} to {res_upper:.1f}] with {confidence_level_str} Confidence",
            xaxis=dict(title="Score Scale (10 to 100)", range=[10, 100], dtick=10),
            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
            template="plotly_dark",
            height=240,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )
        st.plotly_chart(fig_gauge, width="stretch")

        # ------------------------------------------------------------------
        # STUDENT HABIT RADAR CHART SECTION
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 🕸️ Student Habit Radar Chart (vs. Top Performers & Cohort Baseline)")
        st.write(f"Compare this student's 6-dimensional study & lifestyle footprint against **A+ Top Performers** and the **{selected_subject} Cohort Average**.")
        
        radar_col1, radar_col2 = st.columns([1.2, 1], gap="medium")
        
        with radar_col1:
            radar_dict = res.get("radar_dims", {})
            categories = list(radar_dict.keys())
            student_values = list(radar_dict.values())
            
            # Close the polygon loop for Plotly radar
            categories_closed = categories + [categories[0]]
            student_closed = student_values + [student_values[0]]
            top_performer_closed = [85.0, 90.0, 80.0, 95.0, 88.0, 85.0, 85.0]
            cohort_avg_closed = [55.0, 48.0, 45.0, 75.0, 51.0, 50.0, 55.0]
            
            fig_radar = go.Figure()
            
            # 1. Top Performers (A+ Benchmark)
            fig_radar.add_trace(go.Scatterpolar(
                r=top_performer_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.15)',
                line=dict(color='#10B981', width=2),
                name='🌟 Top Performers (A+ Target)'
            ))
            
            # 2. Cohort Average
            fig_radar.add_trace(go.Scatterpolar(
                r=cohort_avg_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(148, 163, 184, 0.10)',
                line=dict(color='#94A3B8', width=1.5, dash='dash'),
                name='👥 Cohort Average'
            ))
            
            # 3. Current Student
            fig_radar.add_trace(go.Scatterpolar(
                r=student_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(56, 189, 248, 0.35)',
                line=dict(color='#38BDF8', width=3),
                name='🎯 Current Student Profile'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[20, 40, 60, 80, 100],
                        ticktext=["20%", "40%", "60%", "80%", "100%"],
                        gridcolor="rgba(255, 255, 255, 0.15)"
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.15)"
                    )
                ),
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_radar, width="stretch")
            
        with radar_col2:
            st.markdown("#### 🧭 Habit Footprint Analysis")
            balance_score = res.get("habit_balance_score", 75.0)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">HABIT BALANCE INDEX</span>
                <div style="font-size: 2.2rem; font-weight: 800; color: {'#34D399' if balance_score >= 80 else '#38BDF8' if balance_score >= 65 else '#FBBF24'};">
                    {balance_score:.1f}<span style="font-size: 1rem; color: #94A3B8;">/100</span>
                </div>
                <span style="font-size: 0.85rem; color: #E2E8F0;">{'🌟 Elite Multi-Dimensional Habit Profile' if balance_score >= 80 else '👍 Well-Rounded Study Routine' if balance_score >= 65 else '⚠️ Asymmetric Study Habits'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Dimension Breakdown
            st.markdown("**Detailed Dimension Breakdown:**")
            dim_data = res.get("radar_dims", {})
            for d_name, d_val in dim_data.items():
                st.write(f"• **{d_name}:** `{d_val:.0f}%` (vs. A+ Target: `85%+`)")

        # ------------------------------------------------------------------
        # SHAP EXPLAINABILITY WATERFALL SECTION
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 🧠 {selected_subject} — SHAP Explainability Breakdown")
        st.write("SHAP (SHapley Additive exPlanations) breaks down the exact positive and negative contribution of each habit relative to the average student baseline.")
        
        shap_data = res.get("shap_res")
        if not shap_data:
            shap_data = compute_shap_waterfall(active_model, res["raw_input"])
            
        base_val = shap_data["base_value"]
        contribs = shap_data["contributions"]
        sorted_contribs = sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=True)
        
        waterfall_x = ["Baseline (Average Student)"] + [k for k, v in sorted_contribs] + ["Final Predicted Score"]
        waterfall_y = [base_val] + [v for k, v in sorted_contribs] + [res["pred_score"]]
        waterfall_measures = ["absolute"] + ["relative"] * len(sorted_contribs) + ["total"]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP Feature Attribution",
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_x,
            textposition="outside",
            text=[f"{v:+.2f}" if i not in [0, len(waterfall_y)-1] else f"{v:.1f}" for i, v in enumerate(waterfall_y)],
            y=waterfall_y,
            connector={"line": {"color": "#64748B"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#38BDF8"}}
        ))
        
        fig_waterfall.update_layout(
            title=f"SHAP Decision Waterfall ({selected_subject}): Base Score ({base_val:.1f}) ➔ Predicted Score ({res['pred_score']:.1f})",
            xaxis_title="Input Variables & Habit Adjustments",
            yaxis_title="Score Impact (Points)",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_waterfall, width="stretch")
        
        # Narrative Explanation
        top_pos = [f"**{k}** (+{v:.1f} pts)" for k, v in sorted_contribs if v > 0][:2]
        top_neg = [f"**{k}** ({v:.1f} pts)" for k, v in sorted_contribs if v < 0][:2]
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if top_pos:
                st.success(f"🟢 **Top Score Drivers for {selected_subject}:** {', '.join(top_pos)} provided the largest positive lift.")
            else:
                st.info("ℹ️ Habits are close to the average cohort baseline.")
        with exp_col2:
            if top_neg:
                st.warning(f"🔴 **Areas Holding Score Back:** {', '.join(top_neg)} reduced the overall predicted score.")
            else:
                st.success("🌟 All attributes contributed positively above cohort average!")

        # Personalized Recommendations Box
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### 💡 Personalized Study Plan for {selected_subject}")

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if "Math" in selected_subject:
                st.info(f"📐 **Math Practice Boost:** Solving **3 more question papers** is estimated to raise predicted math score by **+{res['delta_papers']:.1f} points**.")
            else:
                if res["hours_studied"] < 7:
                    st.info(f"📈 **Study Habit Boost:** Increasing daily study time by **2 hours** (to {res['hours_studied'] + 2} hrs) is predicted to increase score by **+{res['delta_study']:.1f} points**.")
                else:
                    st.success("🌟 **Study Discipline:** Current study hours are already strong! Maintain consistency.")
                
        with rec_col2:
            if "Science" in selected_subject and res["sleep_hours"] < 7:
                st.warning(f"😴 **STEM Sleep Hygiene:** Sleeping only {res['sleep_hours']} hrs. In science/physics, **7-8 hours** of sleep is critical for memory consolidation.")
            elif "Humanities" in selected_subject:
                st.success("📖 **Humanities Focus:** Active reading and extracurricular debates strongly support language performance!")
            else:
                st.info(f"📄 **Mock Practice Boost:** Practicing **3 more question papers** is estimated to raise predicted score by **+{res['delta_papers']:.1f} points**.")
    else:
        st.markdown("---")
        st.info(f"👈 Set the student's study inputs above and click **'🚀 Predict {selected_subject} Score & Habit Radar'** to generate the prediction.")

# -------------------------------------------------------------
# TAB 2: Batch CSV Prediction
# -------------------------------------------------------------
with tab2:
    st.subheader(f"📂 {selected_subject} — Batch Prediction with Confidence Intervals")
    st.write(f"Inference Engine: **`{selected_model_name}`** | Discipline: **`{selected_subject}`** | Confidence Level: **`{confidence_level_str}`** (Margin ±{cur_margin:.2f} pts).")
    
    # Sample download or demo load
    demo_col1, demo_col2 = st.columns([1, 2])
    with demo_col1:
        if os.path.exists(SAMPLE_BATCH_PATH):
            sample_df = pd.read_csv(SAMPLE_BATCH_PATH)
            csv_sample_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Template / Sample CSV",
                data=csv_sample_bytes,
                file_name="student_sample_template.csv",
                mime="text/csv",
                width="stretch"
            )
            
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: Hours_Studied, Previous_Score, Sleep_Hours, Sample_Question_Papers_Practiced, Extracurricular_Activities",
        type=["csv"]
    )
    
    df_to_predict = None
    if uploaded_file is not None:
        try:
            df_to_predict = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded `{uploaded_file.name}` with {len(df_to_predict)} records.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    else:
        if st.checkbox("🧪 Use Demo Cohort (20 students from test dataset)", value=True):
            if os.path.exists(SAMPLE_BATCH_PATH):
                df_to_predict = pd.read_csv(SAMPLE_BATCH_PATH)
            
    if df_to_predict is not None:
        df_to_predict.columns = [c.strip().replace(" ", "_") for c in df_to_predict.columns]
        required_cols = ["Hours_Studied", "Previous_Score", "Sleep_Hours", "Sample_Question_Papers_Practiced", "Extracurricular_Activities"]
        missing_cols = [c for c in required_cols if c not in df_to_predict.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            df_engineered = engineer_features(df_to_predict)
            preds = active_model.predict(df_engineered)
            preds = np.clip(preds, 10.0, 100.0)
            
            result_df = df_engineered.copy()
            result_df["Subject"] = selected_subject
            result_df["Predicted_Score"] = np.round(preds, 1)
            
            z = Z_SCORES.get(confidence_level_str, 1.960)
            batch_margin = z * residual_std
            result_df[f"Lower_Bound_{confidence_level_str}"] = np.round(np.clip(result_df["Predicted_Score"] - batch_margin, 10.0, 100.0), 1)
            result_df[f"Upper_Bound_{confidence_level_str}"] = np.round(np.clip(result_df["Predicted_Score"] + batch_margin, 10.0, 100.0), 1)
            result_df["Grade"] = [compute_grade_and_status(s)[0] for s in result_df["Predicted_Score"]]
            result_df["Status"] = [compute_grade_and_status(s)[2] for s in result_df["Predicted_Score"]]
            
            # Batch Summary Metrics
            st.markdown(f"#### 📊 {selected_subject} Cohort Summary")
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Students", len(result_df))
            b_col2.metric("Cohort Average Score", f"{result_df['Predicted_Score'].mean():.1f} / 100")
            b_col3.metric("Margin of Error", f"±{batch_margin:.2f} pts")
            b_col4.metric("Pass Rate (Score ≥ 50)", f"{(result_df['Predicted_Score'] >= 50).mean() * 100:.1f}%")
            
            fig_batch = px.histogram(
                result_df,
                x="Predicted_Score",
                color="Grade",
                nbins=15,
                title=f"Predicted {selected_subject} Score Distribution",
                color_discrete_map={"A+": "#10B981", "A": "#34D399", "B": "#3B82F6", "C": "#F59E0B", "D": "#EF4444", "F": "#DC2626"},
                template="plotly_dark"
            )
            fig_batch.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
            st.plotly_chart(fig_batch, width="stretch")
            
            st.markdown("#### 📋 Detailed Predictions Table (with Confidence Bounds)")
            st.dataframe(result_df, width="stretch", height=320)
            
            csv_output = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {selected_subject} Predictions CSV",
                data=csv_output,
                file_name=f"{selected_subject.lower().replace(' ', '_')}_predictions.csv",
                mime="text/csv",
                type="primary"
            )

# -------------------------------------------------------------
# TAB 3: Model Benchmarks & Multi-Subject Comparison Matrix
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Multi-Subject Benchmarks & Cross-Discipline Comparison")
    st.write(f"Compare performance benchmarks and feature sensitivities across all **4 academic disciplines** and **9 ML algorithms**.")
    
    # 1. Multi-Subject Comparison Summary Card
    st.markdown("#### 🌐 Cross-Subject Performance Overview")
    overview_rows = []
    for sub_name, sub_data in all_subject_metrics.items():
        best_m = sub_data.get("best_model_metrics", {})
        meta = SUBJECT_METADATA.get(sub_name, {})
        overview_rows.append({
            "Subject": f"{meta.get('icon', '📚')} {sub_name}",
            "Discipline": meta.get("badge", ""),
            "Champion Model": sub_data.get("best_model_name", "Linear Regression"),
            "Test R² Score": f"{best_m.get('test_r2', 0.988)*100:.2f}%",
            "MAE (Avg Error)": f"±{best_m.get('test_mae', 1.65):.2f} pts",
            "RMSE": f"{best_m.get('test_rmse', 2.08):.2f}",
            "Key Focus": meta.get("focus", "")
        })
    st.dataframe(pd.DataFrame(overview_rows), width="stretch")
    
    # 2. Leaderboard for Current Subject
    st.markdown(f"#### 🏆 Model Leaderboard for `{selected_subject}`")
    benchmarks = active_subject_metrics.get("all_model_benchmarks", {})
    bench_df = pd.DataFrame(benchmarks).T.reset_index()
    bench_df.columns = ["Model", "Train R²", "Test R²", "MAE (Test)", "RMSE (Test)", "MSE (Test)"]
    
    def get_category(name):
        if "XGBoost" in name or "LightGBM" in name or "Gradient" in name or "Hist" in name:
            return "⚡ Advanced Gradient Booster"
        elif "Stacking" in name or "Forest" in name:
            return "🌲 Ensemble Method"
        elif "Tree" in name:
            return "🌿 Decision Tree"
        else:
            return "📐 Linear Model"
            
    bench_df["Category"] = bench_df["Model"].apply(get_category)
    bench_df = bench_df[["Model", "Category", "Test R²", "MAE (Test)", "RMSE (Test)", "Train R²"]]
    bench_df = bench_df.sort_values(by="Test R²", ascending=False).reset_index(drop=True)
    
    st.dataframe(
        bench_df.style.highlight_max(subset=["Test R²"], color="#1E3A8A")
                      .highlight_min(subset=["MAE (Test)", "RMSE (Test)"], color="#065F46")
                      .format({
                          "Train R²": "{:.4f}",
                          "Test R²": "{:.4f}",
                          "MAE (Test)": "{:.4f}",
                          "RMSE (Test)": "{:.4f}"
                      }),
        width="stretch"
    )
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"#### 🧠 Global SHAP Feature Importance ({selected_subject})")
        feat_dict = active_subject_metrics.get("feature_importances", {})
        if feat_dict:
            feat_df = pd.DataFrame({
                "Feature": [f.replace("_", " ") for f in feat_dict.keys()],
                "Mean |SHAP Value| (Impact)": [abs(v) for v in feat_dict.values()]
            }).sort_values(by="Mean |SHAP Value| (Impact)", ascending=True)
            
            fig_shap_global = px.bar(
                feat_df,
                x="Mean |SHAP Value| (Impact)",
                y="Feature",
                orientation="h",
                title=f"Feature Sensitivity: {selected_subject}",
                color="Mean |SHAP Value| (Impact)",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig_shap_global.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_shap_global, width="stretch")
            
    with m_col2:
        st.markdown(f"#### 🎯 Actual vs. Predicted ({selected_subject})")
        plot_data = active_subject_metrics.get("sample_test_plot_data", {})
        if plot_data:
            actual = plot_data.get("actual", [])
            predicted = plot_data.get("predicted", [])
            lower_95 = plot_data.get("lower_95", [])
            upper_95 = plot_data.get("upper_95", [])
            
            scatter_df = pd.DataFrame({
                "Actual Score": actual,
                "Predicted Score": predicted,
                "Lower 95%": lower_95,
                "Upper 95%": upper_95
            }).sort_values(by="Actual Score").reset_index(drop=True)
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=scatter_df["Actual Score"], y=scatter_df["Upper 95%"], mode="lines", line=dict(width=0), showlegend=False))
            fig_scatter.add_trace(go.Scatter(x=scatter_df["Actual Score"], y=scatter_df["Lower 95%"], mode="lines", line=dict(width=0), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.15)', name="95% Confidence Band"))
            fig_scatter.add_trace(go.Scatter(x=scatter_df["Actual Score"], y=scatter_df["Predicted Score"], mode="markers", name="Student Predictions", marker=dict(color="#38BDF8", size=7, opacity=0.85)))
            
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Ideal (Perfect)", line=dict(color="#10B981", dash="dash", width=2)))
            
            fig_scatter.update_layout(
                title=f"Test Scatter Tunnel: {selected_subject}",
                xaxis_title="Actual Score",
                yaxis_title="Predicted Score",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, width="stretch")

# -------------------------------------------------------------
# TAB 4: Dataset Explorer (EDA & Habit Footprint)
# -------------------------------------------------------------
with tab4:
    st.subheader(f"📈 {active_subject_meta['icon']} {selected_subject} — Dataset & Habit Footprint Analysis")
    st.write(f"Explore patterns, correlations, and cross-grade habit footprints across **10,000 student records** for **{selected_subject}**.")
    
    df_raw = get_cached_subject_data(selected_subject)
    if df_raw is not None:
        # Cross-Grade Habit Radar Analysis
        st.markdown("#### 🕸️ Cohort Habit Radar Footprint by Letter Grade (A+ vs. A vs. C vs. F)")
        
        # Calculate grade column for dataset
        df_eda = df_raw.copy()
        df_eda["Grade_Category"] = pd.cut(
            df_eda["Performance_Index"],
            bins=[0, 50, 60, 70, 80, 90, 100],
            labels=["F (<50)", "D (50-60)", "C (60-70)", "B (70-80)", "A (80-90)", "A+ (90-100)"],
            right=False
        )
        
        # Group by grade
        grade_stats = df_eda.groupby("Grade_Category", observed=False).agg({
            "Hours_Studied": "mean",
            "Previous_Score": "mean",
            "Sample_Question_Papers_Practiced": "mean",
            "Sleep_Hours": "mean",
            "Study_Effort_Score": "mean"
        }).reset_index()
        
        radar_eda_cats = ["Study Time", "Foundation Score", "Mock Papers", "Sleep Quality", "Study Effort"]
        radar_eda_cats_closed = radar_eda_cats + [radar_eda_cats[0]]
        
        fig_grade_radar = go.Figure()
        
        # Plot A+ tier
        if "A+ (90-100)" in grade_stats["Grade_Category"].values:
            row_ap = grade_stats[grade_stats["Grade_Category"] == "A+ (90-100)"].iloc[0]
            vals_ap = [
                min(100, (row_ap["Hours_Studied"] / 9) * 100),
                min(100, max(0, ((row_ap["Previous_Score"] - 40) / 60) * 100)),
                min(100, (row_ap["Sample_Question_Papers_Practiced"] / 10) * 100),
                min(100, max(0, 100 - abs(row_ap["Sleep_Hours"] - 7.5) * 20)),
                min(100, (row_ap["Study_Effort_Score"] / 9.4) * 100)
            ]
            fig_grade_radar.add_trace(go.Scatterpolar(
                r=vals_ap + [vals_ap[0]],
                theta=radar_eda_cats_closed,
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='#10B981', width=2.5),
                name='🌟 Top Tier (Grade A+)'
            ))
            
        # Plot B tier
        if "B (70-80)" in grade_stats["Grade_Category"].values:
            row_b = grade_stats[grade_stats["Grade_Category"] == "B (70-80)"].iloc[0]
            vals_b = [
                min(100, (row_b["Hours_Studied"] / 9) * 100),
                min(100, max(0, ((row_b["Previous_Score"] - 40) / 60) * 100)),
                min(100, (row_b["Sample_Question_Papers_Practiced"] / 10) * 100),
                min(100, max(0, 100 - abs(row_b["Sleep_Hours"] - 7.5) * 20)),
                min(100, (row_b["Study_Effort_Score"] / 9.4) * 100)
            ]
            fig_grade_radar.add_trace(go.Scatterpolar(
                r=vals_b + [vals_b[0]],
                theta=radar_eda_cats_closed,
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.15)',
                line=dict(color='#3B82F6', width=2),
                name='👍 Above Average (Grade B)'
            ))
            
        # Plot F tier
        if "F (<50)" in grade_stats["Grade_Category"].values:
            row_f = grade_stats[grade_stats["Grade_Category"] == "F (<50)"].iloc[0]
            vals_f = [
                min(100, (row_f["Hours_Studied"] / 9) * 100),
                min(100, max(0, ((row_f["Previous_Score"] - 40) / 60) * 100)),
                min(100, (row_f["Sample_Question_Papers_Practiced"] / 10) * 100),
                min(100, max(0, 100 - abs(row_f["Sleep_Hours"] - 7.5) * 20)),
                min(100, (row_f["Study_Effort_Score"] / 9.4) * 100)
            ]
            fig_grade_radar.add_trace(go.Scatterpolar(
                r=vals_f + [vals_f[0]],
                theta=radar_eda_cats_closed,
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.15)',
                line=dict(color='#EF4444', width=2, dash='dot'),
                name='🚨 At Risk (Grade F)'
            ))
            
        fig_grade_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255, 255, 255, 0.15)"),
                angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.15)")
            ),
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_grade_radar, width="stretch")
        
        st.markdown("---")
        
        eda_col1, eda_col2 = st.columns([1, 1])
        with eda_col1:
            st.markdown(f"#### 📐 Statistical Summary ({selected_subject})")
            st.dataframe(df_raw.describe().T, width="stretch")
            
        with eda_col2:
            st.markdown("#### 🔥 Correlation Heatmap")
            numeric_df = df_raw.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="Viridis",
                title=f"{selected_subject} Correlation Matrix",
                template="plotly_dark"
            )
            fig_corr.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, width="stretch")
            
        st.markdown("---")
        st.markdown("#### 🔍 Interactive Relationship Inspector")
        
        all_numeric_cols = [c for c in df_raw.select_dtypes(include=['number']).columns if c != "Performance_Index"]
        if not all_numeric_cols:
            all_numeric_cols = NUMERICAL_FEATURES
        
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            default_index = all_numeric_cols.index("Study_Effort_Score") if "Study_Effort_Score" in all_numeric_cols else 0
            x_axis = st.selectbox(
                "Select X-Axis Feature (Base or Engineered):",
                options=all_numeric_cols,
                index=default_index
            )
        with sc_col2:
            color_var = st.selectbox(
                "Color Points By:",
                options=["Extracurricular_Activities", "Sleep_Hours"],
                index=0
            )
            
        sample_eda = df_raw.sample(n=min(1000, len(df_raw)), random_state=42)
        fig_custom = px.scatter(
            sample_eda,
            x=x_axis,
            y="Performance_Index",
            color=color_var,
            trendline="ols",
            title=f"{x_axis.replace('_', ' ')} vs. {selected_subject} Performance",
            template="plotly_dark",
            color_discrete_sequence=["#38BDF8", "#F472B6"]
        )
        fig_custom.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_custom, width="stretch")
    else:
        st.warning("Subject dataset file not found.")
