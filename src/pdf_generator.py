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

def create_student_pdf_report(
    student_name: str,
    subject: str,
    model_name: str,
    confidence_level: str,
    pred_score: float,
    lower_bound: float,
    upper_bound: float,
    margin: float,
    letter_grade: str,
    standing_desc: str,
    inputs: dict,
    engineered_inputs: dict,
    habit_balance_score: float,
    shap_contributions: dict,
    shap_base_value: float,
    recommendations: list
) -> bytes:
    """
    Generates a professional academic diagnostic & prediction report as a PDF in memory.
    Returns bytes suitable for Streamlit st.download_button.
    """
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
    
    # Custom styles
    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#0284C7")
    dark_text = colors.HexColor("#0F172A")
    muted_text = colors.HexColor("#64748B")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=muted_text,
        alignment=TA_LEFT
    )
    
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=dark_text
    )
    
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=dark_text
    )

    metric_title_style = ParagraphStyle(
        "MetricTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=muted_text,
        alignment=TA_CENTER
    )
    
    metric_val_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        textColor=primary_color,
        alignment=TA_CENTER
    )

    story = []
    
    # 1. Header Banner Table
    date_str = datetime.now().strftime("%B %d, %Y")
    header_data = [
        [
            Paragraph("<b>🎓 Student Academic Performance Predictor</b><br/><font color='#64748B' size=8>AI-Driven Outcome Prediction & Habit Diagnostic Report</font>", title_style),
            Paragraph(f"<b>Report Date:</b> {date_str}<br/><b>Discipline:</b> {subject}<br/><b>Student:</b> {student_name}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceBefore=2, spaceAfter=8))
    
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
            Paragraph(f"<b>{pred_score:.1f}</b> <font size=9 color='#64748B'>/100</font>", metric_val_style),
            Paragraph(f"<b>[{lower_bound:.1f} — {upper_bound:.1f}]</b><br/><font size=7 color='#64748B'>Margin ±{margin:.1f} pts</font>", metric_val_style),
            Paragraph(f"<b>Grade {letter_grade}</b>", metric_val_style),
            Paragraph(f"<b>{habit_balance_score:.1f}</b> <font size=9 color='#64748B'>/100</font>", metric_val_style),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[135, 135, 135, 135])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_bg),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))
    
    # Standing Description
    standing_p = Paragraph(f"<b>Academic Standing:</b> {standing_desc} &nbsp;|&nbsp; <b>Model Engine:</b> {model_name}", body_style)
    story.append(standing_p)
    story.append(Spacer(1, 10))
    
    # 3. Student Study Habits & Engineered Features Table
    story.append(Paragraph("<b>2. Student Study Profile & Engineered Metrics</b>", section_heading))
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
            Paragraph(f"{engineered_inputs.get('Study_to_Sleep_Ratio', 0.71):.2f}", body_style)
        ],
        [
            Paragraph("📝 Previous Exam Score", body_style),
            Paragraph(f"{inputs.get('Previous_Score', 75)} / 100", body_style),
            Paragraph("📄 Practice Density", body_style),
            Paragraph(f"{engineered_inputs.get('Practice_Density', 0.5):.2f} mock/hr", body_style)
        ],
        [
            Paragraph("📄 Practice Mock Papers", body_style),
            Paragraph(f"{inputs.get('Sample_Question_Papers_Practiced', 3)} papers", body_style),
            Paragraph("⚡ Composite Effort Score", body_style),
            Paragraph(f"{engineered_inputs.get('Study_Effort_Score', 4.2):.2f} / 9.4", body_style)
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
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 10))
    
    # 4. SHAP Explainability Breakdown
    story.append(Paragraph("<b>3. SHAP Explainability & Factor Attribution Breakdown</b>", section_heading))
    shap_intro = Paragraph(
        f"The prediction engine decomposes the score relative to the <b>cohort baseline of {shap_base_value:.1f} points</b>. "
        "The table below shows exactly how much each individual habit raised or lowered this student's predicted score:",
        body_style
    )
    story.append(shap_intro)
    story.append(Spacer(1, 6))
    
    sorted_shap = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_table_data = [
        [
            Paragraph("<b>Influencing Factor / Variable</b>", body_bold),
            Paragraph("<b>SHAP Attribution (Point Impact)</b>", body_bold),
            Paragraph("<b>Impact Classification</b>", body_bold)
        ]
    ]
    for feat_name, impact in sorted_shap:
        impact_str = f"+{impact:.2f} pts" if impact > 0 else f"{impact:.2f} pts"
        status_text = "<font color='#16A34A'><b>🟢 Positive Score Boost</b></font>" if impact > 0 else "<font color='#DC2626'><b>🔴 Drag on Score</b></font>" if impact < 0 else "Neutral"
        shap_table_data.append([
            Paragraph(feat_name, body_style),
            Paragraph(f"<b>{impact_str}</b>", body_style),
            Paragraph(status_text, body_style)
        ])
        
    shap_table = Table(shap_table_data, colWidths=[200, 160, 180])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 10))
    
    # 5. Personalized Recommendations & Action Plan
    story.append(Paragraph("<b>4. Personalized Study Action Plan & Improvement Recommendations</b>", section_heading))
    for rec in recommendations:
        rec_p = Paragraph(f"• {rec}", body_style)
        story.append(rec_p)
        story.append(Spacer(1, 3))
        
    story.append(Spacer(1, 10))
    
    # 6. Footer Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=border_color, spaceBefore=4, spaceAfter=6))
    footer_text = Paragraph(
        "<font size=7 color='#64748B'><b>Diagnostic Notice:</b> This report is generated by the Student Academic Performance Predictor ML system. "
        "Predictions and confidence intervals are statistical estimates based on 10,000 empirical student records and should be used as advisory academic guidance.</font>",
        body_style
    )
    story.append(footer_text)
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
