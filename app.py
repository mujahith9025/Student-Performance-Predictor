import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.pdf_generator import generate_student_pdf_report
from src.explainability import explain_single_student

# Set Page Config
st.set_page_config(
    page_title="Student Performance & Risk Predictor",
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
    .result-box-pass {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #6EE7B7;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box-risk {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #FCA5A5;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-score {
        font-size: 3.5rem;
        font-weight: 900;
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

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("⚙️ ML Engine Settings")
    
    if all_models:
        default_index = 0
        model_names = sorted(list(all_models.keys()))
        if "Voting Ensemble Regressor" in model_names:
            default_index = model_names.index("Voting Ensemble Regressor")
            
        selected_model_name = st.selectbox(
            "Regression Algorithm:",
            options=model_names,
            index=default_index
        )
        active_model = all_models[selected_model_name]
    else:
        active_model = best_model
        selected_model_name = "Voting Ensemble (Default)"
        
    st.info(f"Classifier: **Support Vector Machine (ROC-AUC 0.933)**")
    
    st.markdown("---")
    st.markdown("### 📊 System Benchmarks")
    st.markdown("- **Regression $R^2$:** **76.44%** ($\pm 5.95$ marks)")
    st.markdown("- **Pass/Fail Accuracy:** **89.5%**")
    st.markdown("- **Explainable AI (XAI):** Enabled")
    st.markdown("- **PDF Generator:** ReportLab 5.0")
    
    st.caption("Student Performance & Dropout Risk Predictor")

# Main Header
st.markdown('<div class="main-header">🎓 Student Performance & Dropout Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dual-task machine learning system with Explainable AI (SHAP attributions), risk diagnostics, and verified PDF report generation.</div>', unsafe_allow_html=True)

# Overview Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-val">76.4%</div><div class="metric-lbl">Regression R²</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-val">0.933</div><div class="metric-lbl">Classifier ROC-AUC</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-val">XAI</div><div class="metric-lbl">SHAP Attributions</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-val">PDF</div><div class="metric-lbl">Verified Export</div></div>', unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Score & Risk Predictor", 
    "🔍 Explainable AI (XAI)",
    "📈 Exploratory Data Analysis", 
    "🏆 Model Leaderboards", 
    "⚙️ Hyperparameter Tuning",
    "📖 System Architecture"
])

# ----------------- TAB 1: PREDICTOR -----------------
with tab1:
    st.markdown("### 📝 Enter Student Academic & Demographic Profile")
    st.markdown("Enter student characteristics to compute predicted marks, pass probability, local SHAP attributions, and dropout risk alerts.")
    
    with st.form("prediction_form"):
        # Student Info Header
        st.markdown("#### 🆔 Student Identification")
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            student_name = st.text_input("Student Full Name", value="Alex Johnson", placeholder="e.g. Alex Johnson")
        with id_col2:
            student_id = st.text_input("Student ID / Roll Number", value="STU-2026-101", placeholder="e.g. STU-2026-101")
            
        st.write("")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 👤 Demographics & Environment")
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
                "Lunch Plan",
                options=["standard", "free/reduced"],
                help="Nutritional lunch plan type"
            )
            test_prep = st.selectbox(
                "Test Preparation Course",
                options=["none", "completed"],
                help="Pre-examination test preparation course completion"
            )
            
        with col_right:
            st.markdown("#### 📚 Existing Examination Marks")
            st.markdown("Scores earned by the student in other subjects (out of 100):")
            
            reading_score = st.slider("Reading Score (0 - 100)", min_value=0, max_value=100, value=75, step=1)
            writing_score = st.slider("Writing Score (0 - 100)", min_value=0, max_value=100, value=72, step=1)
            
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
            transformed_input = preprocessor.transform(input_df)
            
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
                <div style="font-size: 1.1rem; color: {score_color}; font-weight: 600;">Predicted Mathematics Score & Risk Status</div>
                <div class="result-score" style="color: {score_color};">{predicted_math:.1f} <span style="font-size: 1.5rem;">/ 100</span></div>
                <div style="font-size: 1rem; color: {score_color};">
                    <b>Pass Probability: {pass_prob:.1f}%</b> • Status: <b>{'Passed' if is_pass==1 else 'At-Risk / Fail'}</b> • Risk Tier: <b>{risk_level}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric("Math Score (Predicted)", f"{predicted_math:.1f} / 100")
            with r_col2:
                st.metric("3-Subject Average", f"{overall_avg:.1f} / 100")
            with r_col3:
                st.metric("Pass Probability", f"{pass_prob:.1f}%")
            with r_col4:
                st.metric("Predicted Grade", grade)
                
            # Pass Probability Progress Bar
            st.write("")
            st.markdown(f"**Academic Success Confidence:** `{pass_prob:.1f}%`")
            st.progress(int(pass_prob))
                
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
                    color = "green" if impact >= 0 else "red"
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

# ----------------- TAB 2: EXPLAINABLE AI (XAI) -----------------
with tab2:
    st.markdown("### 🔍 Explainable AI (XAI) & Feature Importance Analysis")
    st.markdown("Global interpretability analysis to unpack what factors truly determine student examination marks:")
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("#### 1. Global Feature Importance (Permutation Impact)")
        p10 = os.path.join(plots_dir, "10_global_feature_importance.png")
        if os.path.exists(p10):
            st.image(p10, caption="Writing and Reading scores account for over 77% of predictive power.", use_container_width=True)
            
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

# ----------------- TAB 3: EDA & INSIGHTS -----------------
with tab3:
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

# ----------------- TAB 4: DUAL MODEL LEADERBOARD -----------------
with tab4:
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

# ----------------- TAB 5: HYPERPARAMETER TUNING -----------------
with tab5:
    st.markdown("### ⚙️ 5-Fold Cross-Validation Hyperparameter Optimization")
    tuning_csv = os.path.join(os.path.dirname(__file__), "artifacts", "hyperparameter_tuning_results.csv")
    if os.path.exists(tuning_csv):
        t_df = pd.read_csv(tuning_csv)
        st.dataframe(t_df.drop(columns=["Filename"], errors="ignore"), use_container_width=True, hide_index=True)

# ----------------- TAB 6: SYSTEM ARCHITECTURE -----------------
with tab6:
    st.markdown("### 📖 Dual-Engine Machine Learning Architecture")
    st.markdown("""
    ```
    1. Input Features (Demographics + Reading/Writing Marks)
       └── Transformed via ColumnTransformer (StandardScaler + OneHotEncoder)
    
    2. Dual Machine Learning Pipeline:
       ├── Regression Engine: Voting Ensemble Regressor (R² 76.44%, MAE ±5.95)
       ├── Classification Engine: Support Vector Classifier (Accuracy 89.5%, ROC-AUC 0.933)
       └── Explainable AI (XAI): Permutation Importance & SHAP Directional Attributions
    
    3. Outputs & Deliverables:
       ├── Exact Predicted Marks & Grade (A+ to F)
       ├── Pass Probability & Early Dropout Risk Tier
       ├── Local Real-Time Factor Attribution Breakdown (Points Added/Deducted)
       └── Verified PDF Performance Certificate
    ```
    """)
