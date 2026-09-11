import os
import streamlit as pd_st
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #6EE7B7;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-score {
        font-size: 3.5rem;
        font-weight: 900;
        color: #047857;
        line-height: 1;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load artifacts safely
@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    best_model_path = os.path.join(artifacts_dir, "best_model.joblib")
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(best_model_path):
        return None, None, None
        
    preprocessor = joblib.load(preprocessor_path)
    best_model = joblib.load(best_model_path)
    
    # Load all trained models for selection
    models_dir = os.path.join(artifacts_dir, "models")
    all_models = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib"):
                name = f.replace(".joblib", "").replace("_", " ").title()
                all_models[name] = joblib.load(os.path.join(models_dir, f))
                
    return preprocessor, best_model, all_models

preprocessor, best_model, all_models = load_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("⚙️ Model Settings")
    
    if all_models:
        selected_model_name = st.selectbox(
            "Select ML Algorithm:",
            options=list(all_models.keys()),
            index=list(all_models.keys()).index("Ridge Regression") if "Ridge Regression" in all_models else 0
        )
        active_model = all_models[selected_model_name]
    else:
        active_model = best_model
        selected_model_name = "Ridge Regression (Default)"
        
    st.info(f"Active Model: **{selected_model_name}**")
    
    st.markdown("---")
    st.markdown("### 📊 Benchmark Stats")
    st.markdown("- **Accuracy ($R^2$):** ~76.1%")
    st.markdown("- **Avg Error (MAE):** $\pm 6.0$ marks")
    st.markdown("- **Training Dataset:** 1,000 students")
    
    st.markdown("---")
    st.caption("Machine Learning Project • Beginner Level")

# Main Header
st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An end-to-end machine learning system to predict student examination marks and identify key academic success factors.</div>', unsafe_allow_html=True)

# Overview Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-val">76.1%</div><div class="metric-lbl">Model R² Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-val">±6.0</div><div class="metric-lbl">Avg Prediction Error</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-val">6</div><div class="metric-lbl">Algorithms Evaluated</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-val">Ridge</div><div class="metric-lbl">Champion Model</div></div>', unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Score Predictor", "📈 Exploratory Data Analysis (EDA)", "🏆 Model Leaderboard", "📖 Project Guide"])

# ----------------- TAB 1: PREDICTOR -----------------
with tab1:
    st.markdown("### 📝 Enter Student Academic & Demographic Details")
    st.markdown("Fill in the student details below to generate a real-time predicted **Math Score**.")
    
    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 👤 Student Profile & Demographics")
            gender = st.selectbox("Gender", options=["female", "male"], help="Select student gender")
            race_ethnicity = st.selectbox(
                "Race / Ethnicity Group",
                options=["group A", "group B", "group C", "group D", "group E"],
                index=2
            )
            parental_education = st.selectbox(
                "Parental Level of Education",
                options=[
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree"
                ],
                index=2
            )
            lunch = st.selectbox(
                "Lunch Type",
                options=["standard", "free/reduced"],
                help="Nutritional lunch plan status"
            )
            test_prep = st.selectbox(
                "Test Preparation Course",
                options=["none", "completed"],
                help="Whether the student completed the pre-exam prep course"
            )
            
        with col_right:
            st.markdown("#### 📚 Existing Examination Marks")
            st.markdown("Scores earned by the student in other subjects (out of 100):")
            
            reading_score = st.slider("Reading Score (0 - 100)", min_value=0, max_value=100, value=70, step=1)
            writing_score = st.slider("Writing Score (0 - 100)", min_value=0, max_value=100, value=68, step=1)
            
            st.write("")
            st.write("")
            submit_btn = st.form_submit_button("⚡ Predict Student Math Score", use_container_width=True, type="primary")

    if submit_btn:
        if preprocessor is None or active_model is None:
            st.error("Model artifacts not found! Please run Phase 3 & 4 first.")
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
            
            # Preprocess and Predict
            transformed_input = preprocessor.transform(input_df)
            raw_prediction = active_model.predict(transformed_input)[0]
            predicted_math = float(np.clip(raw_prediction, 0, 100))
            
            # Calculate overall metrics
            overall_avg = (predicted_math + reading_score + writing_score) / 3.0
            
            # Grade Mapping
            if overall_avg >= 90:
                grade, badge_color = "A+ (Outstanding)", "#059669"
            elif overall_avg >= 80:
                grade, badge_color = "A (Excellent)", "#10B981"
            elif overall_avg >= 70:
                grade, badge_color = "B (Good)", "#3B82F6"
            elif overall_avg >= 60:
                grade, badge_color = "C (Satisfactory)", "#F59E0B"
            elif overall_avg >= 50:
                grade, badge_color = "D (Pass)", "#EF4444"
            else:
                grade, badge_color = "F (Needs Support)", "#991B1B"
            
            # Display Prediction Card
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size: 1.1rem; color: #065F46; font-weight: 600;">Predicted Math Examination Score</div>
                <div class="result-score">{predicted_math:.1f} <span style="font-size: 1.5rem; color: #047857;">/ 100</span></div>
                <div style="font-size: 0.95rem; color: #047857;">Algorithm: <b>{selected_model_name}</b> | Expected Range: <b>{max(0, predicted_math - 6.0):.1f} – {min(100, predicted_math + 6.0):.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.metric("Math Score (Predicted)", f"{predicted_math:.1f} / 100")
            with r_col2:
                st.metric("3-Subject Average", f"{overall_avg:.1f} / 100")
            with r_col3:
                st.metric("Predicted Grade", grade)
                
            # Personalized AI Insights
            st.markdown("#### 💡 Diagnostic Feedback for Student")
            tips = []
            if test_prep == "none":
                tips.append("📌 **Test Prep Course:** Enrolling in the preparation course is statistically associated with a **+9.4 mark boost** in Mathematics.")
            if lunch == "free/reduced":
                tips.append("📌 **Nutritional Support:** Standard lunch access correlates with an **+8.0 mark increase** in overall academic stamina.")
            if writing_score < 60:
                tips.append("📌 **Writing Skills:** Stronger writing practice directly reinforces analytical reasoning.")
            if not tips:
                tips.append("🌟 **Great Standing:** The student has positive indicators across all academic support factors.")
                
            for tip in tips:
                st.info(tip)

# ----------------- TAB 2: EDA & INSIGHTS -----------------
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis & Visual Insights")
    st.markdown("Explore key trends and statistical correlations discovered during Phase 2 analysis.")
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1. Score Distributions")
        p1 = os.path.join(plots_dir, "01_score_distributions.png")
        if os.path.exists(p1):
            st.image(p1, caption="Normal distributions of Math, Reading, and Writing scores.", use_container_width=True)
            
        st.markdown("#### 3. Test Preparation Impact")
        p3 = os.path.join(plots_dir, "03_test_prep_impact.png")
        if os.path.exists(p3):
            st.image(p3, caption="Higher marks across all subjects for test prep completers.", use_container_width=True)
            
    with col_b:
        st.markdown("#### 2. Correlation Matrix")
        p2 = os.path.join(plots_dir, "02_correlation_heatmap.png")
        if os.path.exists(p2):
            st.image(p2, caption="Strong positive Pearson correlations between all 3 subjects.", use_container_width=True)
            
        st.markdown("#### 4. Parental Education Level")
        p4 = os.path.join(plots_dir, "04_parental_education_impact.png")
        if os.path.exists(p4):
            st.image(p4, caption="Higher parental education level shifts median math performance upward.", use_container_width=True)

# ----------------- TAB 3: MODEL LEADERBOARD -----------------
with tab3:
    st.markdown("### 🏆 Machine Learning Model Comparison")
    st.markdown("6 algorithms evaluated on 200 unseen test records during Phase 5:")
    
    metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_metrics.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.dataframe(
            m_df.drop(columns=["Filename"], errors="ignore"),
            use_container_width=True,
            hide_index=True
        )
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        p6 = os.path.join(plots_dir, "06_model_performance_comparison.png")
        if os.path.exists(p6):
            st.image(p6, caption="R² Comparison across all 6 models.", use_container_width=True)
    with col_m2:
        p7 = os.path.join(plots_dir, "07_actual_vs_predicted.png")
        if os.path.exists(p7):
            st.image(p7, caption="Actual vs. Predicted scatter plot for Ridge Regression.", use_container_width=True)

# ----------------- TAB 4: PROJECT GUIDE -----------------
with tab4:
    st.markdown("### 📖 Step-by-Step Machine Learning Architecture")
    st.markdown("""
    This project follows the end-to-end Machine Learning Lifecycle:
    
    1. **Phase 1: Dataset Setup & Definition**
       - 1,000 student records with 8 core demographic and exam features.
    2. **Phase 2: Exploratory Data Analysis (EDA)**
       - Identified 0 null values, bell curve score distributions, and $+9.4$ test prep bonus.
    3. **Phase 3: Data Preprocessing Pipeline**
       - `StandardScaler()` for continuous scores and `OneHotEncoder(drop='first')` for categories.
    4. **Phase 4: Model Training**
       - Trained 6 algorithms: Linear Regression, Ridge, Lasso, KNN, Decision Trees, Random Forest.
    5. **Phase 5: Evaluation & Metric Comparison**
       - Ridge Regression achieved the highest generalization with $R^2 = 76.1\%$ and $MAE = 6.02$ marks.
    6. **Phase 6: Web App Deployment**
       - Streamlit interactive interface with real-time inference and personalized student feedback.
    """)

