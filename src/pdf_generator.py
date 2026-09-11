import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_student_pdf_report(
    student_name,
    student_id,
    gender,
    race_ethnicity,
    parental_education,
    lunch,
    test_prep,
    reading_score,
    writing_score,
    predicted_math,
    overall_avg,
    grade,
    model_name,
    tips,
    pass_prob=None,
    risk_level="Safe"
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()

    # Custom Color Palette
    primary_color = colors.HexColor("#1E3A8A")   # Navy Blue
    secondary_color = colors.HexColor("#0284C7") # Sky Blue
    accent_green = colors.HexColor("#059669")    # Green
    light_bg = colors.HexColor("#F8FAFC")        # Soft grey
    border_color = colors.HexColor("#CBD5E1")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # 1. Header Banner
    story.append(Paragraph("STUDENT ACADEMIC PERFORMANCE & RISK EVALUATION REPORT", title_style))
    story.append(Paragraph("Machine Learning Diagnostic Forecast & Dropout Risk Assessment", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=12))

    # 2. Student & Evaluation Metadata Table
    current_date = datetime.now().strftime("%B %d, %Y")
    prob_text = f"<b>Pass Probability:</b> {pass_prob:.1f}% ({risk_level})" if pass_prob is not None else "Verified"
    metadata_data = [
        [
            Paragraph("<b>Student Name:</b> " + str(student_name), body_style),
            Paragraph("<b>Student ID / Roll:</b> " + str(student_id), body_style)
        ],
        [
            Paragraph("<b>Date of Evaluation:</b> " + current_date, body_style),
            Paragraph("<b>Academic Risk Status:</b> " + prob_text, body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[260, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Subject Examination & Prediction Breakdown Table
    story.append(Paragraph("1. Subject Competency & Predicted Marks", section_heading))
    
    score_data = [
        [
            Paragraph("<b>Exam Subject</b>", bold_body),
            Paragraph("<b>Evaluation Type</b>", bold_body),
            Paragraph("<b>Score Obtained / Forecast</b>", bold_body),
            Paragraph("<b>Performance Status</b>", bold_body)
        ],
        [
            Paragraph("<b>Mathematics</b>", body_style),
            Paragraph("ML Model Forecast", body_style),
            Paragraph(f"<b>{predicted_math:.1f} / 100</b>", bold_body),
            Paragraph("Predicted Pass" if predicted_math >= 50 else "<font color='red'>At-Risk</font>", body_style)
        ],
        [
            Paragraph("<b>Reading Comprehension</b>", body_style),
            Paragraph("Actual Assessment", body_style),
            Paragraph(f"{reading_score:.1f} / 100", body_style),
            Paragraph("Proficient" if reading_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Writing & Expression</b>", body_style),
            Paragraph("Actual Assessment", body_style),
            Paragraph(f"{writing_score:.1f} / 100", body_style),
            Paragraph("Proficient" if writing_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Cumulative 3-Subject Average</b>", bold_body),
            Paragraph("Combined", bold_body),
            Paragraph(f"<b>{overall_avg:.1f} / 100</b>", bold_body),
            Paragraph(f"<b>Grade: {grade.split()[0]}</b>", bold_body)
        ]
    ]

    score_table = Table(score_data, colWidths=[160, 130, 140, 100])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#ECFDF5")),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(4):
        score_data[0][i].style.textColor = colors.white

    story.append(score_table)
    story.append(Spacer(1, 12))

    # 4. Demographic & Environmental Factors
    story.append(Paragraph("2. Academic Environment & Background Profile", section_heading))
    
    factors_data = [
        [
            Paragraph("<b>Factor</b>", bold_body),
            Paragraph("<b>Student Status</b>", bold_body),
            Paragraph("<b>Empirical Dataset Influence</b>", bold_body)
        ],
        [
            Paragraph("Test Preparation Course", body_style),
            Paragraph(test_prep.capitalize(), body_style),
            Paragraph("+9.4 marks statistical gain for course completers", body_style)
        ],
        [
            Paragraph("Nutritional Lunch Plan", body_style),
            Paragraph(lunch.capitalize(), body_style),
            Paragraph("+8.0 marks correlation with standard nutrition", body_style)
        ],
        [
            Paragraph("Parental Education Level", body_style),
            Paragraph(parental_education.title(), body_style),
            Paragraph("Positive correlation with higher degree attainment", body_style)
        ],
        [
            Paragraph("Demographic Group / Gender", body_style),
            Paragraph(f"{gender.capitalize()} • {race_ethnicity.title()}", body_style),
            Paragraph("Standardized benchmark control baseline", body_style)
        ]
    ]

    factors_table = Table(factors_data, colWidths=[150, 140, 240])
    factors_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(3):
        factors_data[0][i].style.textColor = colors.white

    story.append(factors_table)
    story.append(Spacer(1, 12))

    # 5. Diagnostic Advice & Improvement Plan
    story.append(Paragraph("3. Actionable Academic Recommendations", section_heading))
    for tip in tips:
        clean_tip = tip.replace("📌", "•").replace("🌟", "•").replace("⚠️", "•").replace("**", "")
        story.append(Paragraph(clean_tip, body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 15))

    # 6. Verification Footer
    footer_data = [
        [
            Paragraph("<b>Status:</b> Verified by Dual ML Engine", body_style),
            Paragraph("<b>Confidence:</b> ±5.95 marks (ROC-AUC 0.933)", body_style),
            Paragraph("____________________________<br/>Academic Counselor Signature", body_style)
        ]
    ]
    footer_table = Table(footer_data, colWidths=[170, 190, 170])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    # Build PDF Document
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
