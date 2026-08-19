import os
import sys
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def create_student_pdf_report(*args, **kwargs) -> bytes:
    """
    Generates a professional academic diagnostic & prediction report as a PDF in memory.
    Ultra-robust signature accepting both positional and keyword arguments.
    """
    # Keyword or positional extraction
    arg_keys = [
        "student_name", "subject", "model_name", "confidence_level",
        "pred_score", "lower_bound", "upper_bound", "margin",
        "letter_grade", "standing_desc", "inputs", "engineered_inputs",
        "habit_balance_score", "shap_contributions", "shap_base_value",
        "recommendations", "view_mode", "teacher_notes", "risk_flags"
    ]
    
    params = {}
    for idx, val in enumerate(args):
        if idx < len(arg_keys):
            params[arg_keys[idx]] = val
    params.update(kwargs)
    
    student_name = str(params.get("student_name", "Student"))
    subject = str(params.get("subject", "General Academics"))
    model_name = str(params.get("model_name", "Linear Regression"))
    confidence_level = str(params.get("confidence_level", "95%"))
    
    try:
        pred_score = float(params.get("pred_score", 55.0))
    except (TypeError, ValueError):
        pred_score = 55.0
        
    try:
        lower_bound = float(params.get("lower_bound", pred_score - 4.0))
    except (TypeError, ValueError):
        lower_bound = pred_score - 4.0
        
    try:
        upper_bound = float(params.get("upper_bound", pred_score + 4.0))
    except (TypeError, ValueError):
        upper_bound = pred_score + 4.0
        
    try:
        margin = float(params.get("margin", 4.0))
    except (TypeError, ValueError):
        margin = 4.0
        
    letter_grade = str(params.get("letter_grade", "C"))
    standing_desc = str(params.get("standing_desc", "Satisfactory"))
    
    inputs = params.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
        
    engineered_inputs = params.get("engineered_inputs", {})
    if not isinstance(engineered_inputs, dict):
        engineered_inputs = {}
        
    try:
        habit_balance_score = float(params.get("habit_balance_score", 75.0))
    except (TypeError, ValueError):
        habit_balance_score = 75.0
        
    shap_contributions = params.get("shap_contributions", {})
    if not isinstance(shap_contributions, dict):
        shap_contributions = {}
        
    try:
        shap_base_value = float(params.get("shap_base_value", 54.8))
    except (TypeError, ValueError):
        shap_base_value = 54.8
        
    recommendations = params.get("recommendations", [])
    if not isinstance(recommendations, (list, tuple)):
        recommendations = [str(recommendations)]
        
    view_mode = str(params.get("view_mode", "Student Mode"))
    teacher_notes = params.get("teacher_notes", None)
    if teacher_notes is not None:
        teacher_notes = str(teacher_notes)
        
    risk_flags = params.get("risk_flags", [])
    if not isinstance(risk_flags, (list, tuple)):
        risk_flags = []

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#0284C7")
    dark_text = colors.HexColor("#0F172A")
    muted_text = colors.HexColor("#64748B")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    alert_bg = colors.HexColor("#FEF2F2")
    alert_border = colors.HexColor("#EF4444")
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=muted_text,
        alignment=TA_LEFT
    )
    
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=7,
        spaceAfter=3
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=dark_text
    )
    
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=dark_text
    )

    metric_title_style = ParagraphStyle(
        "MetricTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        textColor=muted_text,
        alignment=TA_CENTER
    )
    
    metric_val_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=17,
        textColor=primary_color,
        alignment=TA_CENTER
    )

    story = []
    
    # 1. Header Banner Table
    date_str = datetime.now().strftime("%B %d, %Y")
    report_tag = "👩‍🏫 Educator Academic Diagnostic & Intervention Report" if "Teacher" in view_mode else "🧑‍🎓 Student Academic Outcome & Habit Report"
    
    header_data = [
        [
            Paragraph(f"<b>🎓 Student Academic Performance Predictor</b><br/><font color='#0284C7' size=8><b>{report_tag}</b></font>", title_style),
            Paragraph(f"<b>Report Date:</b> {date_str}<br/><b>Discipline:</b> {subject}<br/><b>Student:</b> {student_name}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=2, spaceAfter=6))
    
    # Risk Flags (if in Teacher Mode)
    if risk_flags and len(risk_flags) > 0 and "Teacher" in view_mode:
        flags_text = "<b>🚨 Educator Risk & Early Warning Flags:</b><br/>" + "<br/>".join([f"• <font color='#DC2626'>{flag}</font>" for flag in risk_flags])
        flag_table = Table([[Paragraph(flags_text, body_style)]], colWidths=[540])
        flag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), alert_bg),
            ('BOX', (0,0), (-1,-1), 1, alert_border),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(flag_table)
        story.append(Spacer(1, 6))
    
    # 2. Executive Summary Metrics Cards
    story.append(Paragraph("<b>1. Predicted Academic Outcome & Confidence Bounds</b>", section_heading))
    
    metrics_data = [
        [
            Paragraph("PREDICTED SCORE", metric_title_style),
            Paragraph(f"{confidence_level} CONFIDENCE RANGE", metric_title_style),
            Paragraph("EXPECTED GRADE", metric_title_style),
            Paragraph("HABIT BALANCE INDEX", metric_title_style),
        ],
        [
            Paragraph(f"<b>{pred_score:.1f}</b> <font size=8 color='#64748B'>/100</font>", metric_val_style),
            Paragraph(f"<b>[{lower_bound:.1f} — {upper_bound:.1f}]</b><br/><font size=6.5 color='#64748B'>Margin ±{margin:.1f} pts</font>", metric_val_style),
            Paragraph(f"<b>Grade {letter_grade}</b>", metric_val_style),
            Paragraph(f"<b>{habit_balance_score:.1f}</b> <font size=8 color='#64748B'>/100</font>", metric_val_style),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[135, 135, 135, 135])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 8))
    
    # Standing Description
    standing_p = Paragraph(f"<b>Academic Standing:</b> {standing_desc} &nbsp;|&nbsp; <b>Model Engine:</b> {model_name}", body_style)
    story.append(standing_p)
    story.append(Spacer(1, 8))
    
    # 3. Student Study Habits & Engineered Features Table
    story.append(Paragraph("<b>2. Student Study Profile & Engineered Metrics</b>", section_heading))
    
    try:
        s2s_val = float(engineered_inputs.get("Study_to_Sleep_Ratio", 0.71))
    except (TypeError, ValueError):
        s2s_val = 0.71
        
    try:
        pd_val = float(engineered_inputs.get("Practice_Density", 0.5))
    except (TypeError, ValueError):
        pd_val = 0.5
        
    try:
        ses_val = float(engineered_inputs.get("Study_Effort_Score", 4.2))
    except (TypeError, ValueError):
        ses_val = 4.2
        
    profile_data = [
        [
            Paragraph("<b>Input Habit Dimension</b>", body_bold),
            Paragraph("<b>Observed Value</b>", body_bold),
            Paragraph("<b>Custom Metric / Ratio</b>", body_bold),
            Paragraph("<b>Calculated Value</b>", body_bold)
        ],
        [
            Paragraph("⏱️ Daily Hours Studied", body_style),
            Paragraph(f"{inputs.get('Hours_Studied', 5)} hrs/day", body_style),
            Paragraph("⚖️ Study-to-Sleep Ratio", body_style),
            Paragraph(f"{s2s_val:.2f}", body_style)
        ],
        [
            Paragraph("📝 Previous Exam Score", body_style),
            Paragraph(f"{inputs.get('Previous_Score', 75)} / 100", body_style),
            Paragraph("📄 Practice Density", body_style),
            Paragraph(f"{pd_val:.2f} mock/hr", body_style)
        ],
        [
            Paragraph("📄 Practice Mock Papers", body_style),
            Paragraph(f"{inputs.get('Sample_Question_Papers_Practiced', 3)} papers", body_style),
            Paragraph("⚡ Composite Effort Score", body_style),
            Paragraph(f"{ses_val:.2f} / 9.4", body_style)
        ],
        [
            Paragraph("😴 Daily Sleep Hours", body_style),
            Paragraph(f"{inputs.get('Sleep_Hours', 7)} hrs/day", body_style),
            Paragraph("⚽ Extracurricular Activities", body_style),
            Paragraph(f"{inputs.get('Extracurricular_Activities', 'Yes')}", body_style)
        ]
    ]
    profile_table = Table(profile_data, colWidths=[140, 130, 140, 130])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 8))
    
    # 4. SHAP Explainability Breakdown
    story.append(Paragraph("<b>3. SHAP Explainability & Factor Attribution Breakdown</b>", section_heading))
    shap_intro = Paragraph(
        f"The prediction engine decomposes the score relative to the <b>cohort baseline of {shap_base_value:.1f} points</b>. "
        "The table below shows exactly how much each individual habit raised or lowered this student's predicted score:",
        body_style
    )
    story.append(shap_intro)
    story.append(Spacer(1, 4))
    
    sorted_shap = sorted(shap_contributions.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
    shap_table_data = [
        [
            Paragraph("<b>Influencing Factor / Variable</b>", body_bold),
            Paragraph("<b>SHAP Attribution (Point Impact)</b>", body_bold),
            Paragraph("<b>Impact Classification</b>", body_bold)
        ]
    ]
    for feat_name, impact in sorted_shap:
        try:
            imp_val = float(impact)
            impact_str = f"+{imp_val:.2f} pts" if imp_val > 0 else f"{imp_val:.2f} pts"
            status_text = "<font color='#16A34A'><b>🟢 Positive Score Boost</b></font>" if imp_val > 0 else "<font color='#DC2626'><b>🔴 Drag on Score</b></font>" if imp_val < 0 else "Neutral"
        except (TypeError, ValueError):
            impact_str = str(impact)
            status_text = "Neutral"
            
        shap_table_data.append([
            Paragraph(str(feat_name), body_style),
            Paragraph(f"<b>{impact_str}</b>", body_style),
            Paragraph(status_text, body_style)
        ])
        
    shap_table = Table(shap_table_data, colWidths=[200, 160, 180])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 8))
    
    # 5. Personalized Recommendations & Action Plan
    rec_title = "<b>4. Educator Intervention Action Plan & Counseling Roadmap</b>" if "Teacher" in view_mode else "<b>4. Personalized Study Action Plan & Improvement Recommendations</b>"
    story.append(Paragraph(rec_title, section_heading))
    for rec in recommendations:
        rec_p = Paragraph(f"• {str(rec)}", body_style)
        story.append(rec_p)
        story.append(Spacer(1, 2.5))
        
    # Teacher Custom Advisory Notes (if present)
    if teacher_notes and teacher_notes.strip():
        story.append(Spacer(1, 6))
        notes_html = f"<b>👩‍🏫 Educator Counseling & Diagnostic Notes:</b><br/>{teacher_notes.strip()}"
        notes_table = Table([[Paragraph(notes_html, body_style)]], colWidths=[540])
        notes_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#3B82F6")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(notes_table)
        
    story.append(Spacer(1, 8))
    
    # 6. Footer Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=border_color, spaceBefore=3, spaceAfter=4))
    footer_text = Paragraph(
        "<font size=6.5 color='#64748B'><b>Diagnostic Notice:</b> This report is generated by the Student Academic Performance Predictor ML system. "
        "Predictions and confidence intervals are statistical estimates based on 10,000 empirical student records and should be used as advisory academic guidance.</font>",
        body_style
    )
    story.append(footer_text)
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def create_goal_roadmap_pdf_report(*args, **kwargs) -> bytes:
    """
    Generates an official Reverse Goal Solver & Target Achievement Roadmap PDF.
    Ultra-robust signature accepting both positional and keyword arguments.
    """
    arg_keys = ["student_name", "subject", "target_score", "current_pred", "gap", "pathways"]
    params = {}
    for idx, val in enumerate(args):
        if idx < len(arg_keys):
            params[arg_keys[idx]] = val
    params.update(kwargs)
    
    student_name = str(params.get("student_name", "Student"))
    subject = str(params.get("subject", "General Academics"))
    
    try:
        target_score = float(params.get("target_score", 85.0))
    except (TypeError, ValueError):
        target_score = 85.0
        
    try:
        current_pred = float(params.get("current_pred", 70.0))
    except (TypeError, ValueError):
        current_pred = 70.0
        
    try:
        gap = float(params.get("gap", target_score - current_pred))
    except (TypeError, ValueError):
        gap = target_score - current_pred
        
    pathways = params.get("pathways", [])
    if not isinstance(pathways, (list, tuple)):
        pathways = []
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#0284C7")
    dark_text = colors.HexColor("#0F172A")
    muted_text = colors.HexColor("#64748B")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    
    title_style = ParagraphStyle("GoalTitle", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=primary_color)
    subtitle_style = ParagraphStyle("GoalSubtitle", fontName="Helvetica", fontSize=9, leading=12, textColor=muted_text)
    section_heading = ParagraphStyle("GoalHeading", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=primary_color, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("GoalBody", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=dark_text)
    body_bold = ParagraphStyle("GoalBold", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, textColor=dark_text)
    
    story = []
    
    date_str = datetime.now().strftime("%B %d, %Y")
    header_data = [
        [
            Paragraph("<b>🎯 Target Score Solver — Action Roadmap</b><br/><font color='#64748B' size=8>Inverse Goal Optimization & Habit Strategy Plan</font>", title_style),
            Paragraph(f"<b>Report Date:</b> {date_str}<br/><b>Discipline:</b> {subject}<br/><b>Student:</b> {student_name}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (1,0), (1,0), 'RIGHT')]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=2, spaceAfter=6))
    
    # Target vs Current Summary
    story.append(Paragraph("<b>1. Goal Target vs Current Baseline Summary</b>", section_heading))
    summary_data = [
        [
            Paragraph("CURRENT PROJECTED SCORE", ParagraphStyle("C1", fontName="Helvetica", fontSize=8, alignment=TA_CENTER, textColor=muted_text)),
            Paragraph("DESIRED TARGET SCORE", ParagraphStyle("C2", fontName="Helvetica", fontSize=8, alignment=TA_CENTER, textColor=muted_text)),
            Paragraph("REQUIRED SCORE GAIN", ParagraphStyle("C3", fontName="Helvetica", fontSize=8, alignment=TA_CENTER, textColor=muted_text)),
        ],
        [
            Paragraph(f"<b>{current_pred:.1f}</b> <font size=8 color='#64748B'>/100</font>", ParagraphStyle("V1", fontName="Helvetica-Bold", fontSize=16, alignment=TA_CENTER, textColor=primary_color)),
            Paragraph(f"<b>{target_score:.1f}</b> <font size=8 color='#64748B'>/100</font>", ParagraphStyle("V2", fontName="Helvetica-Bold", fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor("#16A34A"))),
            Paragraph(f"<b>+{gap:.1f} pts</b>", ParagraphStyle("V3", fontName="Helvetica-Bold", fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor("#0284C7"))),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[180, 180, 180])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))
    
    # Recommended Pathways
    story.append(Paragraph("<b>2. Optimized Action Pathways to Reach Target Score</b>", section_heading))
    
    pathway_table_data = [
        [
            Paragraph("<b>Pathway Strategy</b>", body_bold),
            Paragraph("<b>Daily Study</b>", body_bold),
            Paragraph("<b>Mock Papers</b>", body_bold),
            Paragraph("<b>Daily Sleep</b>", body_bold),
            Paragraph("<b>Weekly Hours</b>", body_bold),
            Paragraph("<b>Predicted Score</b>", body_bold)
        ]
    ]
    for p in pathways:
        pathway_table_data.append([
            Paragraph(f"<b>{p.get('name', 'Pathway')}</b><br/><font size=7 color='#64748B'>{p.get('tag', '')}</font>", body_style),
            Paragraph(f"<b>{float(p.get('required_hours', 5)):.1f} hrs</b> ({float(p.get('delta_hours', 0)):+0.1f})", body_style),
            Paragraph(f"<b>{int(p.get('required_papers', 3))} papers</b> ({int(p.get('delta_papers', 0)):+d})", body_style),
            Paragraph(f"{float(p.get('required_sleep', 7)):.1f} hrs", body_style),
            Paragraph(f"{float(p.get('weekly_study_hours', 35)):.1f} hrs/wk", body_style),
            Paragraph(f"<b>{float(p.get('predicted_score', 75)):.1f} / 100</b>", body_bold)
        ])
        
    pathway_table = Table(pathway_table_data, colWidths=[150, 75, 75, 70, 80, 90])
    pathway_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pathway_table)
    story.append(Spacer(1, 10))
    
    # Detailed Strategy Notes
    story.append(Paragraph("<b>3. Strategic Recommendations for Execution</b>", section_heading))
    for p in pathways:
        story.append(Paragraph(f"• <b>{p.get('name', 'Pathway')}:</b> {p.get('description', '')} Commitment: {p.get('weekly_study_hours', 35)} hours per week with {p.get('required_papers', 3)} practice exams.", body_style))
        story.append(Spacer(1, 3))
        
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.8, color=border_color, spaceBefore=4, spaceAfter=4))
    story.append(Paragraph("<font size=6.5 color='#64748B'><b>Diagnostic Notice:</b> This target roadmap was computed using the inverse regression solver over 10,000 real student records. Individual student learning speeds may vary.</font>", body_style))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
