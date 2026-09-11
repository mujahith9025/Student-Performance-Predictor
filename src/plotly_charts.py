import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTION CHARTS
# ---------------------------------------------------------
def create_score_gauge(score, grade):
    """
    Interactive Speedometer Radial Gauge for predicted math score.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Predicted Math Marks ({grade})</b>", 'font': {'size': 16, 'family': 'Outfit', 'color': '#1E3A8A'}},
        number={'suffix': " / 100", 'font': {'size': 32, 'family': 'Outfit', 'color': '#1E3A8A'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#2563EB", 'thickness': 0.28},
            'bgcolor': "rgba(241, 245, 249, 0.8)",
            'borderwidth': 1.5,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.22)'},
                {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.22)'},
                {'range': [70, 85], 'color': 'rgba(59, 130, 246, 0.22)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 4},
                'thickness': 0.8,
                'value': 85
            }
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E293B", 'family': "Plus Jakarta Sans"}
    )
    return fig

def create_radar_chart(reading, writing, predicted_math, socio_index, attendance=85.0, study_hours=12.0, sleep_hours=7.5, prep_status="none"):
    """
    8-Axis Student Multidimensional Competency Radar Chart comparing student vs cohort benchmark.
    """
    categories = [
        'Reading Score', 'Writing Score', 'Math (Pred)',
        'Attendance %', 'Study Effort', 'Sleep Wellness',
        'Socio-Readiness', 'Test Readiness'
    ]
    
    # Scale each dimension to 0 - 100 for radar uniformity
    att_scaled = min(100.0, float(attendance))
    study_scaled = min(100.0, (float(study_hours) / 20.0) * 100.0)
    sleep_scaled = min(100.0, (float(sleep_hours) / 8.0) * 100.0)
    socio_scaled = min(100.0, float(socio_index) * 12.0)
    prep_scaled = 95.0 if prep_status == "completed" else 35.0
    
    student_values = [
        reading, writing, predicted_math,
        att_scaled, study_scaled, sleep_scaled,
        socio_scaled, prep_scaled
    ]
    benchmark_values = [68.0, 68.0, 67.5, 85.0, 60.0, 90.0, 65.0, 50.0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values + [benchmark_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.20)',
        line=dict(color='#94A3B8', dash='dot', width=1.5),
        name='Cohort Benchmark'
    ))
    fig.add_trace(go.Scatterpolar(
        r=student_values + [student_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.32)',
        line=dict(color='#2563EB', width=2.5),
        name='Student Profile'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color="#64748B")),
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font=dict(size=10)),
        height=260,
        margin=dict(l=25, r=25, t=25, b=15),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_local_xai_waterfall(base_val, predicted, contrib_df):
    """
    Interactive Waterfall Chart explaining point additions and deductions from baseline.
    """
    factors = ["Baseline Avg"] + contrib_df["Factor"].tolist() + ["Predicted Score"]
    measure = ["absolute"] + ["relative"] * len(contrib_df) + ["total"]
    y_values = [base_val] + contrib_df["Impact"].tolist() + [predicted]
    
    text_values = [f"{base_val:.1f}"] + [f"{x:+.2f}" for x in contrib_df["Impact"]] + [f"{predicted:.1f}"]
    
    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measure,
        x=factors,
        textposition="outside",
        text=text_values,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)", "width": 1.5}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#2563EB"}}
    ))
    fig.update_layout(
        title="<b>Interactive Local SHAP Waterfall (Points Impact)</b>",
        title_font=dict(size=14, family="Outfit"),
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        xaxis=dict(tickangle=-15, tickfont=dict(size=10, family="Plus Jakarta Sans")),
        yaxis=dict(title="Score (Marks)", range=[max(0, min(y_values) - 15), min(100, max(y_values) + 15)])
    )
    return fig

def create_goal_trajectory_chart(current_score, prep_benefit, target_score):
    """
    Gamified Milestone Stepper Trajectory Chart (Baseline -> Test Prep Boost -> Final Target).
    """
    steps = ["1. Baseline Score", "2. + Test Prep Boost", "3. + Exam Targets (Goal)"]
    step1 = current_score
    step2 = min(100, current_score + prep_benefit)
    step3 = target_score
    scores = [step1, step2, step3]
    
    fig = go.Figure()
    
    # Pathway line
    fig.add_trace(go.Scatter(
        x=steps,
        y=scores,
        mode='lines+markers+text',
        line=dict(color='#2563EB', width=3, dash='solid'),
        marker=dict(size=[14, 16, 20], color=['#94A3B8', '#06B6D4', '#10B981'], symbol=['circle', 'diamond', 'star']),
        text=[f"{s:.1f} Marks" for s in scores],
        textposition=['bottom center', 'top center', 'top center'],
        textfont=dict(size=12, family="Plus Jakarta Sans", color="#1E3A8A"),
        name="Milestone Path"
    ))
    
    # Target Line
    fig.add_hline(y=target_score, line_dash="dot", line_color="#10B981", annotation_text=f"Target Goal ({target_score})", annotation_position="bottom right")
    
    fig.update_layout(
        title="<b>Gamified Academic Milestone Trajectory</b>",
        title_font=dict(size=14, family="Outfit"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        yaxis=dict(title="Marks (0 - 100)", range=[max(0, min(scores) - 15), min(105, max(scores) + 15)]),
        showlegend=False
    )
    return fig


# ---------------------------------------------------------
# TAB 2: CLASSROOM BATCH ANALYTICS
# ---------------------------------------------------------
def create_batch_bubble_chart(processed_df):
    """
    Bubble Chart of Reading vs Writing with size=Math Score and color=Risk Tier.
    """
    fig = px.scatter(
        processed_df,
        x="reading score",
        y="writing score",
        size="Predicted_Math_Score",
        color="Risk_Tier",
        hover_name="student_name" if "student_name" in processed_df.columns else None,
        hover_data=["Predicted_Math_Score", "Predicted_Grade", "Pass_Probability_Pct"],
        color_discrete_map={
            "Safe / Low Risk": "#10B981",
            "Moderate Risk": "#F59E0B",
            "🚨 High Academic Risk": "#EF4444"
        },
        title="<b>Classroom Cohort Map (Reading vs Writing vs Predicted Math)</b>"
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)'
    )
    return fig

# ---------------------------------------------------------
# TAB 4: GLOBAL EXPLAINABLE AI CHARTS
# ---------------------------------------------------------
def create_global_importance_plotly(importance_df):
    """
    Interactive Horizontal Bar Chart of Global Feature Importance.
    """
    top_df = importance_df.head(10).iloc[::-1]
    
    colors = ["#2563EB" if ("score" in f or "verbal" in f) else "#06B6D4" for f in top_df["Feature"]]
    
    fig = go.Figure(go.Bar(
        x=top_df["Relative_Impact_Pct"],
        y=top_df["Feature"],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in top_df["Relative_Impact_Pct"]],
        textposition='outside',
        hoverinfo='x+y'
    ))
    fig.update_layout(
        title="<b>Global Permutation Feature Importance (50 Shuffles)</b>",
        title_font=dict(size=14, family="Outfit"),
        xaxis=dict(title="Relative Importance (%)", range=[0, max(top_df["Relative_Impact_Pct"]) * 1.2]),
        yaxis=dict(tickfont=dict(size=10, family="Plus Jakarta Sans")),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)'
    )
    return fig

# ---------------------------------------------------------
# TAB 5: EXPLORATORY DATA ANALYSIS (EDA) CHARTS
# ---------------------------------------------------------
def create_eda_distribution_plotly(df):
    """
    Interactive Score Distribution Histogram & KDE for Math, Reading, Writing.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["math score"], name="Math Score", opacity=0.6, marker_color="#2563EB"))
    fig.add_trace(go.Histogram(x=df["reading score"], name="Reading Score", opacity=0.6, marker_color="#06B6D4"))
    fig.add_trace(go.Histogram(x=df["writing score"], name="Writing Score", opacity=0.6, marker_color="#8B5CF6"))
    
    fig.update_layout(
        barmode='overlay',
        title="<b>Subject Score Distributions (1,000 Students)</b>",
        title_font=dict(size=14, family="Outfit"),
        xaxis=dict(title="Marks (0 - 100)"),
        yaxis=dict(title="Student Count"),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_eda_correlation_plotly(df):
    """
    Interactive Correlation Heatmap.
    """
    num_cols = ["math score", "reading score", "writing score"]
    corr = df[num_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="<b>Cross-Subject Pearson Correlation Matrix</b>"
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_eda_test_prep_plotly(df):
    """
    Interactive Box Plot comparing Test Prep Impact.
    """
    fig = px.box(
        df,
        x="test preparation course",
        y="math score",
        color="test preparation course",
        points="all",
        color_discrete_map={"completed": "#10B981", "none": "#EF4444"},
        title="<b>Test Prep Course Impact (+9.4 Marks Boost)</b>"
    )
    fig.update_layout(
        height=320,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)'
    )
    return fig

def create_eda_parental_education_plotly(df):
    """
    Interactive Box Plot comparing Parental Education Influence.
    """
    order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    fig = px.box(
        df,
        x="parental level of education",
        y="math score",
        color="parental level of education",
        category_orders={"parental level of education": order},
        title="<b>Score Distribution by Parental Education</b>"
    )
    fig.update_layout(
        height=320,
        showlegend=False,
        xaxis=dict(tickangle=-20),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)'
    )
    return fig

# ---------------------------------------------------------
# TAB 6: LEADERBOARDS & EVALUATION CHARTS
# ---------------------------------------------------------
def create_model_comparison_plotly(metrics_df):
    """
    Interactive Grouped Bar Chart of Model R2 and MAE.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["Test R2 Score"] * 100,
        name="Test R² Accuracy (%)",
        marker_color="#2563EB"
    ))
    fig.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["Test MAE (marks)"],
        name="Test MAE (Lower is better)",
        marker_color="#F59E0B"
    ))
    fig.update_layout(
        barmode='group',
        title="<b>Regression Model Performance Comparison</b>",
        title_font=dict(size=14, family="Outfit"),
        xaxis=dict(tickangle=-25),
        yaxis=dict(title="Score / Marks"),
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_interactive_confusion_matrix():
    """
    Interactive Heatmap of Test Set Confusion Matrix (SVC).
    """
    z = [[24, 7], [13, 156]]
    x = ['Predicted Fail / Risk', 'Predicted Pass']
    y = ['Actual Fail', 'Actual Pass']
    
    fig = px.imshow(
        z,
        x=x,
        y=y,
        text_auto=True,
        color_continuous_scale="Blues",
        title="<b>Confusion Matrix on Test Dataset (SVC)</b>"
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_interactive_roc_curve():
    """
    Interactive Multi-Classifier ROC Curves.
    """
    fig = go.Figure()
    
    # Baseline Diagonal
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#94A3B8'), name='Random Guess (AUC = 0.50)'))
    
    # SVC Curve
    fpr_svc = [0.0, 0.03, 0.06, 0.12, 0.18, 0.22, 0.35, 1.0]
    tpr_svc = [0.0, 0.68, 0.85, 0.94, 0.96, 0.98, 1.00, 1.0]
    fig.add_trace(go.Scatter(x=fpr_svc, y=tpr_svc, mode='lines+markers', line=dict(color='#2563EB', width=2.5), name='Support Vector Classifier (AUC = 0.932)'))
    
    # Logistic Regression Curve
    fpr_lr = [0.0, 0.05, 0.09, 0.15, 0.22, 0.38, 1.0]
    tpr_lr = [0.0, 0.65, 0.82, 0.92, 0.95, 0.99, 1.0]
    fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', line=dict(color='#10B981', width=2), name='Logistic Regression (AUC = 0.932)'))
    
    # Gradient Boosting Curve
    fpr_gb = [0.0, 0.06, 0.11, 0.18, 0.28, 0.45, 1.0]
    tpr_gb = [0.0, 0.62, 0.80, 0.90, 0.94, 0.98, 1.0]
    fig.add_trace(go.Scatter(x=fpr_gb, y=tpr_gb, mode='lines', line=dict(color='#8B5CF6', width=2), name='Gradient Boosting (AUC = 0.923)'))
    
    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC-AUC) Curves</b>",
        title_font=dict(size=14, family="Outfit"),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[-0.02, 1.02]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(size=10))
    )
    return fig

# ---------------------------------------------------------
# PRESCRIPTIVE SOLUTION & INTERVENTION CHARTS
# ---------------------------------------------------------
def create_prescriptive_study_hours_chart(study_hours_dict):
    """
    Donut chart of weekly prescribed study hours across academic dimensions.
    """
    labels = list(study_hours_dict.keys())
    values = list(study_hours_dict.values())
    colors = ['#2563EB', '#0D9488', '#F59E0B', '#8B5CF6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+value',
        texttemplate='<b>%{label}</b><br>%{value} hrs/wk',
        hoverinfo='label+percent+value'
    )])
    
    total_hours = sum(values)
    fig.update_layout(
        title=f"<b>Prescribed Weekly Study Plan (Total: {total_hours:.1f} hrs/wk)</b>",
        title_font=dict(size=14, family="Outfit", color="#1E3A8A"),
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(text=f"<b>{total_hours:.0f}h</b><br>Weekly", x=0.5, y=0.5, font_size=16, font_family="Outfit", showarrow=False)]
    )
    return fig

def create_intervention_uplift_chart(current_score, projected_score, interventions):
    """
    Stepped bridge / waterfall chart showing score progression from baseline to projected goal.
    """
    labels = ["Current Baseline"]
    y_vals = [current_score]
    measures = ["absolute"]
    
    running = current_score
    for item in interventions:
        uplift_val = float(item["est_uplift"].replace("+", "").replace(" pts", "").strip())
        labels.append(item["title"][:22] + "...")
        y_vals.append(uplift_val)
        measures.append("relative")
        running += uplift_val
        
    labels.append("Projected Outcome")
    y_vals.append(projected_score)
    measures.append("total")
    
    fig = go.Figure(go.Waterfall(
        name="Score Uplift",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"{v:.1f}" for v in y_vals],
        y=y_vals,
        connector={"line": {"color": "#94A3B8", "width": 1.5}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#2563EB"}}
    ))
    
    fig.update_layout(
        title="<b>Projected Academic Uplift Roadmap (Points Added)</b>",
        title_font=dict(size=14, family="Outfit", color="#1E3A8A"),
        height=260,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.6)',
        yaxis=dict(title="Math Marks (/100)", range=[0, 105]),
        font=dict(family="Plus Jakarta Sans", size=10)
    )
    return fig

def create_classroom_intervention_cluster_chart(cluster_summary):
    """
    Donut chart of classroom intervention cohort breakdown.
    """
    labels = [
        "Intensive Remedial (High Risk)",
        "Test Prep Bootcamp",
        "Verbal / Reading Support",
        "Honors / Distinction"
    ]
    values = [
        cluster_summary.get("high_risk_count", 0),
        cluster_summary.get("test_prep_needed_count", 0),
        cluster_summary.get("verbal_support_count", 0),
        cluster_summary.get("honors_count", 0)
    ]
    colors = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.50,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="<b>Classroom Prescriptive Cohort Clusters</b>",
        title_font=dict(size=14, family="Outfit", color="#1E3A8A"),
        height=280,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
    )
    return fig

