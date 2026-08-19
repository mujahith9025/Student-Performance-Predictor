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
    page_title="Student Performance Predictor | ML & Confidence Intervals",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
ALL_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "all_trained_models.pkl")
QUANTILE_MODELS_PATH = os.path.join(PROJECT_ROOT, "models", "quantile_models.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "Student_Performance.csv")
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
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
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
def load_model_artifacts():
    """Load cached models bundle, quantile models, and evaluation metrics."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        return None, {}, {}, None
        
    best_model = joblib.load(MODEL_PATH)
    
    all_models = {}
    if os.path.exists(ALL_MODELS_PATH):
        try:
            all_models = joblib.load(ALL_MODELS_PATH)
        except Exception:
            all_models = {"Default Model": best_model}
    else:
        all_models = {"Default Model": best_model}
        
    quantile_models = {}
    if os.path.exists(QUANTILE_MODELS_PATH):
        try:
            quantile_models = joblib.load(QUANTILE_MODELS_PATH)
        except Exception:
            quantile_models = {}
            
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    return best_model, all_models, quantile_models, metrics

@st.cache_data
def load_dataset():
    """Load cached dataset and apply feature engineering for EDA."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df = engineer_features(df)
        return df
    return None

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

# Load trained artifacts
best_model, all_models, quantile_models, metrics = load_model_artifacts()

if best_model is None:
    st.error("⚠️ Trained model not found! Please run `python src/train.py` first to train and serialize the models.")
    st.stop()

# Extract residual std from metrics
residual_std = metrics.get("confidence_intervals", {}).get("residual_std", 2.075)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=110)
    st.title("🎓 Student AI Predictor")
    st.caption("Powered by Machine Learning & Prediction Intervals")
    
    st.markdown("---")
    
    available_model_names = list(all_models.keys()) if all_models else [metrics.get("best_model_name", "Linear Regression")]
    default_idx = available_model_names.index(metrics["best_model_name"]) if metrics and metrics.get("best_model_name") in available_model_names else 0
    
    st.markdown("### 🤖 Select Inference Model")
    selected_model_name = st.selectbox(
        "Choose Algorithm for Prediction:",
        options=available_model_names,
        index=default_idx,
        help="Select between classic linear models, random forests, or Advanced Gradient Boosters (XGBoost, LightGBM, GBDT, Stacking)"
    )
    
    active_model = all_models.get(selected_model_name, best_model)
    
    st.markdown("---")
    st.markdown("### 🎯 Prediction Confidence Level")
    confidence_level_str = st.select_slider(
        "Select Confidence Interval Band:",
        options=["80%", "90%", "95%", "99%"],
        value="95%",
        help="95% is the statistical gold standard, meaning 95 out of 100 students with these habits will fall within the estimated range."
    )
    
    cur_z = Z_SCORES[confidence_level_str]
    cur_margin = cur_z * residual_std
    st.caption(f"📌 Margin of Error: **±{cur_margin:.2f} points** ($Z={cur_z}$)")
    
    if metrics:
        st.markdown("---")
        st.markdown("### 🏆 Model Stats")
        model_bench = metrics.get("all_model_benchmarks", {}).get(selected_model_name, metrics.get("best_model_metrics", {}))
        if model_bench:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test R²", f"{model_bench.get('test_r2', 0.988) * 100:.1f}%")
            with col2:
                st.metric("MAE", f"±{model_bench.get('test_mae', 1.65):.2f}")
            
    st.markdown("---")
    st.markdown("### ⚡ Advanced Gradient Boosters")
    st.markdown("""
    - **XGBoost Regressor:** Extreme gradient boosting
    - **LightGBM Regressor:** High-speed tree booster
    - **GBDT / HistGBDT:** Scikit-Learn gradient boosting
    - **Stacking Ensemble:** Meta-learner combiner
    """)

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Academic Performance Predictor</h1>
    <p>Predict exam outcomes with <b>Prediction Confidence Intervals (80%–99%)</b>, Advanced Gradient Boosters (XGBoost/LightGBM), and Custom Feature Engineering.</p>
</div>
""", unsafe_allow_html=True)

# Multi-Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Student Predictor",
    "📂 Batch CSV Prediction",
    "📊 Benchmarks & Confidence Calibration",
    "📈 Dataset Explorer (EDA)"
])

# -------------------------------------------------------------
# TAB 1: Single Student Predictor
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Student Outcome Prediction with Confidence Bounds")
    st.write(f"Inference Model: **`{selected_model_name}`** | Confidence Level: **`{confidence_level_str}`** (Margin ±{cur_margin:.2f} pts).")
    
    with st.form(key="student_prediction_form"):
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
            predict_btn = st.form_submit_button("🚀 Calculate Predicted Score & Confidence Interval", type="primary", width="stretch")

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

        # What-if scenario boost calculations
        boost_study = raw_input_data.copy()
        boost_study["Hours_Studied"] = min(9, hours_studied + 2)
        boost_score_study = float(active_model.predict(engineer_features(boost_study))[0])
        delta_study = max(0.0, boost_score_study - pred_score)
        
        boost_papers = raw_input_data.copy()
        boost_papers["Sample_Question_Papers_Practiced"] = min(10, sample_papers + 3)
        boost_score_papers = float(active_model.predict(engineer_features(boost_papers))[0])
        delta_papers = max(0.0, boost_score_papers - pred_score)

        st.session_state["single_prediction"] = {
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
            "hours_studied": hours_studied,
            "sleep_hours": sleep_hours,
            "sample_papers": sample_papers,
            "delta_study": delta_study,
            "delta_papers": delta_papers
        }

    # Render results only if prediction has been submitted
    if "single_prediction" in st.session_state:
        res = st.session_state["single_prediction"]
        
        # If user changed confidence level in sidebar, recompute bounds dynamically
        res_lower, res_upper, res_margin = get_interval_bounds(res["pred_score"], confidence_level_str, residual_std)
        
        st.markdown("---")
        st.markdown(f"### 🎯 Prediction Results (via `{res.get('model_used', selected_model_name)}`)")
        
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
        
        # Add background grade zones
        fig_gauge.add_vrect(x0=10, x1=50, fillcolor="#DC2626", opacity=0.15, layer="below", line_width=0, annotation_text="F (Fail)", annotation_position="top left")
        fig_gauge.add_vrect(x0=50, x1=60, fillcolor="#EF4444", opacity=0.15, layer="below", line_width=0, annotation_text="D", annotation_position="top left")
        fig_gauge.add_vrect(x0=60, x1=70, fillcolor="#F59E0B", opacity=0.15, layer="below", line_width=0, annotation_text="C", annotation_position="top left")
        fig_gauge.add_vrect(x0=70, x1=80, fillcolor="#3B82F6", opacity=0.15, layer="below", line_width=0, annotation_text="B", annotation_position="top left")
        fig_gauge.add_vrect(x0=80, x1=100, fillcolor="#10B981", opacity=0.15, layer="below", line_width=0, annotation_text="A / A+", annotation_position="top left")
        
        # Confidence interval band (horizontal bar)
        fig_gauge.add_trace(go.Scatter(
            x=[res_lower, res_upper],
            y=[1, 1],
            mode="lines",
            line=dict(color="#FBBF24", width=10),
            name=f"{confidence_level_str} Confidence Interval",
            hoverinfo="x+name"
        ))
        
        # Point Estimate Marker
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
            title=f"🎯 Prediction Range: [{res_lower:.1f} to {res_upper:.1f}] with {confidence_level_str} Confidence",
            xaxis=dict(title="Score Scale (10 to 100)", range=[10, 100], dtick=10),
            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
            template="plotly_dark",
            height=260,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )
        st.plotly_chart(fig_gauge, width="stretch")

        st.info(f"📊 **Statistical Interpretation:** There is a **{confidence_level_str} probability** that a student with these exact study habits will score between **{res_lower:.1f}** and **{res_upper:.1f}** on the final exam.")

        # Live Engineered Features Card
        st.markdown("#### 🔬 Custom Engineered Metrics")
        eng_col1, eng_col2, eng_col3 = st.columns(3)
        
        with eng_col1:
            ratio_val = res["engineered_input"]["Study_to_Sleep_Ratio"].iloc[0]
            st.markdown(f"""
            <div class="eng-badge">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">⚖️ STUDY-TO-SLEEP RATIO</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8;">{ratio_val:.2f}</div>
                <span style="font-size: 0.8rem; color: #CBD5E1;">{'Balanced ratio' if 0.5 <= ratio_val <= 1.0 else 'High intensity / low sleep' if ratio_val > 1.0 else 'Light study load'}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with eng_col2:
            dens_val = res["engineered_input"]["Practice_Density"].iloc[0]
            st.markdown(f"""
            <div class="eng-badge">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">📄 PRACTICE DENSITY</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8;">{dens_val:.2f}</div>
                <span style="font-size: 0.8rem; color: #CBD5E1;">Mock tests / study hour ratio</span>
            </div>
            """, unsafe_allow_html=True)
            
        with eng_col3:
            eff_val = res["engineered_input"]["Study_Effort_Score"].iloc[0]
            st.markdown(f"""
            <div class="eng-badge">
                <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">⚡ COMPOSITE EFFORT SCORE</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8;">{eff_val:.2f} <span style="font-size: 0.9rem; color: #94A3B8;">/ 9.4</span></div>
                <span style="font-size: 0.8rem; color: #CBD5E1;">Combined study + testing intensity</span>
            </div>
            """, unsafe_allow_html=True)

        # Personalized Recommendations Box
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💡 Personalized Study Plan & Score Boost Insights")

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if res["hours_studied"] < 7:
                st.info(f"📈 **Study Habit Boost:** Increasing daily study time by **2 hours** (to {res['hours_studied'] + 2} hrs) is predicted to increase performance by **+{res['delta_study']:.1f} points**.")
            else:
                st.success("🌟 **Study Discipline:** Current study hours are already strong! Maintain consistency without burnout.")
                
        with rec_col2:
            if res["sleep_hours"] < 7:
                st.warning(f"😴 **Sleep Hygiene:** The student is sleeping {res['sleep_hours']} hrs. Prioritizing **7-8 hours** of sleep enhances memory consolidation and prevents exam fatigue.")
            else:
                st.info(f"📄 **Mock Practice Boost:** Practicing **3 more question papers** is estimated to raise predicted score by **+{res['delta_papers']:.1f} points**.")
    else:
        st.markdown("---")
        st.info("👈 Set the student's study inputs above and click **'🚀 Calculate Predicted Score & Confidence Interval'** to generate the prediction.")

# -------------------------------------------------------------
# TAB 2: Batch CSV Prediction
# -------------------------------------------------------------
with tab2:
    st.subheader("📂 Batch Prediction with Confidence Intervals")
    st.write(f"Inference Engine: **`{selected_model_name}`** | Confidence Level: **`{confidence_level_str}`** (Margin ±{cur_margin:.2f} pts).")
    
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
        # Standardize columns
        df_to_predict.columns = [c.strip().replace(" ", "_") for c in df_to_predict.columns]
        
        required_cols = ["Hours_Studied", "Previous_Score", "Sleep_Hours", "Sample_Question_Papers_Practiced", "Extracurricular_Activities"]
        missing_cols = [c for c in required_cols if c not in df_to_predict.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            # Apply feature engineering automatically
            df_engineered = engineer_features(df_to_predict)
            
            # Predict with selected model
            preds = active_model.predict(df_engineered)
            preds = np.clip(preds, 10.0, 100.0)
            
            result_df = df_engineered.copy()
            result_df["Predicted_Score"] = np.round(preds, 1)
            
            # Add Confidence Interval Bounds
            z = Z_SCORES.get(confidence_level_str, 1.960)
            batch_margin = z * residual_std
            result_df[f"Lower_Bound_{confidence_level_str}"] = np.round(np.clip(result_df["Predicted_Score"] - batch_margin, 10.0, 100.0), 1)
            result_df[f"Upper_Bound_{confidence_level_str}"] = np.round(np.clip(result_df["Predicted_Score"] + batch_margin, 10.0, 100.0), 1)
            result_df["Grade"] = [compute_grade_and_status(s)[0] for s in result_df["Predicted_Score"]]
            result_df["Status"] = [compute_grade_and_status(s)[2] for s in result_df["Predicted_Score"]]
            
            # Batch Summary Metrics
            st.markdown("#### 📊 Cohort Summary")
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Students", len(result_df))
            b_col2.metric("Cohort Average Score", f"{result_df['Predicted_Score'].mean():.1f} / 100")
            b_col3.metric("Margin of Error", f"±{batch_margin:.2f} pts")
            b_col4.metric("Pass Rate (Score ≥ 50)", f"{(result_df['Predicted_Score'] >= 50).mean() * 100:.1f}%")
            
            # Distribution plot of batch
            fig_batch = px.histogram(
                result_df,
                x="Predicted_Score",
                color="Grade",
                nbins=15,
                title="Predicted Score Distribution of Cohort",
                color_discrete_map={"A+": "#10B981", "A": "#34D399", "B": "#3B82F6", "C": "#F59E0B", "D": "#EF4444", "F": "#DC2626"},
                template="plotly_dark"
            )
            fig_batch.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
            st.plotly_chart(fig_batch, width="stretch")
            
            # Data table
            st.markdown("#### 📋 Detailed Predictions Table (with Confidence Bounds)")
            st.dataframe(result_df, width="stretch", height=320)
            
            # Download predicted CSV
            csv_output = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Complete Predictions CSV (with {confidence_level_str} Intervals)",
                data=csv_output,
                file_name="student_predictions_with_intervals.csv",
                mime="text/csv",
                type="primary"
            )

# -------------------------------------------------------------
# TAB 3: Model Benchmarks & Confidence Calibration
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Model Benchmarks & Confidence Interval Calibration")
    st.write("Examine algorithm leaderboards, empirical coverage calibration, and feature weights evaluated on the test set of 1,975 real students.")
    
    if metrics:
        # Confidence Diagnostics Section
        st.markdown("#### 🎯 Prediction Confidence Interval Calibration (Test Set Diagnostics)")
        cov_data = metrics.get("confidence_intervals", {}).get("coverage_diagnostics", {})
        if cov_data:
            cov_rows = []
            for conf_key, diag in cov_data.items():
                cov_rows.append({
                    "Confidence Level (Nominal)": conf_key,
                    "Target Coverage": f"{float(diag['target_coverage'])*100:.0f}%",
                    "Empirical Coverage (Actual)": f"{float(diag['empirical_coverage'])*100:.2f}%",
                    "Z-Multiplier": f"{diag['z_multiplier']:.3f}",
                    "Margin of Error": f"±{diag['margin_of_error']:.2f} pts",
                    "Calibration Status": "✅ Well Calibrated" if abs(diag['empirical_coverage'] - diag['target_coverage']) <= 0.02 else "⚠️ Slight Bias"
                })
            cov_df = pd.DataFrame(cov_rows)
            st.dataframe(cov_df, width="stretch")
            
        # Comparison Table
        st.markdown("#### 🏆 Comprehensive Multi-Model Leaderboard")
        benchmarks = metrics.get("all_model_benchmarks", {})
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
            st.markdown("#### 🔍 Feature Weights / Importances")
            feat_dict = metrics.get("feature_importances", {})
            if feat_dict:
                feat_df = pd.DataFrame({
                    "Feature": [f.replace("_", " ") for f in feat_dict.keys()],
                    "Importance / Weight": list(feat_dict.values())
                }).sort_values(by="Importance / Weight", ascending=True)
                
                fig_feat = px.bar(
                    feat_df,
                    x="Importance / Weight",
                    y="Feature",
                    orientation="h",
                    title="Feature Importances & Regression Weights",
                    color="Importance / Weight",
                    color_continuous_scale="Blues",
                    template="plotly_dark"
                )
                fig_feat.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_feat, width="stretch")
                
        with m_col2:
            st.markdown("#### 🎯 Actual vs. Predicted with 95% Confidence Band")
            plot_data = metrics.get("sample_test_plot_data", {})
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
                
                # Confidence band shaded region
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_df["Actual Score"],
                    y=scatter_df["Upper 95%"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_df["Actual Score"],
                    y=scatter_df["Lower 95%"],
                    mode="lines",
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(56, 189, 248, 0.15)',
                    name="95% Confidence Band"
                ))
                
                # Predictions
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_df["Actual Score"],
                    y=scatter_df["Predicted Score"],
                    mode="markers",
                    name="Student Predictions",
                    marker=dict(color="#38BDF8", size=7, opacity=0.85)
                ))
                
                # Ideal reference line y = x
                min_val = min(min(actual), min(predicted))
                max_val = max(max(actual), max(predicted))
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Ideal (Perfect Prediction)",
                    line=dict(color="#10B981", dash="dash", width=2)
                ))
                
                fig_scatter.update_layout(
                    title=f"Actual vs. Predicted with 95% Prediction Tunnel (R² = {metrics['best_model_metrics']['test_r2']:.4f})",
                    xaxis_title="Actual Score",
                    yaxis_title="Predicted Score",
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_scatter, width="stretch")

# -------------------------------------------------------------
# TAB 4: Dataset Explorer (EDA)
# -------------------------------------------------------------
with tab4:
    st.subheader("📈 Kaggle Dataset Exploration & Visual Analytics")
    st.write("Explore patterns, correlations, and distributions across the **10,000 real student records** from Kaggle, including all newly engineered features.")
    
    df_raw = load_dataset()
    if df_raw is not None:
        eda_col1, eda_col2 = st.columns([1, 1])
        
        with eda_col1:
            st.markdown("#### 📐 Statistical Summary (with Engineered Features)")
            st.dataframe(df_raw.describe().T, width="stretch")
            
        with eda_col2:
            st.markdown("#### 🔥 Correlation Heatmap")
            numeric_df = df_raw.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="Viridis",
                title="Full Feature Correlation Matrix",
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
            title=f"{x_axis.replace('_', ' ')} vs. Performance Index",
            template="plotly_dark",
            color_discrete_sequence=["#38BDF8", "#F472B6"]
        )
        fig_custom.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_custom, width="stretch")
    else:
        st.warning("Dataset file not found at `data/Student_Performance.csv`.")
