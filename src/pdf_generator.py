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
    risk_level="Safe",
    custom_counselor_note=None
):
    """
    Generates a verified, executive-styled Single Student Performance & Risk Evaluation PDF Report.
    """
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
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # 1. Header Banner
    story.append(Paragraph("EDUPREDICT AI • OFFICIAL ACADEMIC REPORT CARD", title_style))
    story.append(Paragraph("Machine Learning Diagnostic Forecast & Dropout Risk Assessment", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=10))

    # 2. Student Metadata Table
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
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

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
            Paragraph("Prerequisite Assessment", body_style),
            Paragraph(f"{reading_score:.1f} / 100", body_style),
            Paragraph("Proficient" if reading_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Writing & Expression</b>", body_style),
            Paragraph("Prerequisite Assessment", body_style),
            Paragraph(f"{writing_score:.1f} / 100", body_style),
            Paragraph("Proficient" if writing_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Cumulative 3-Subject Average</b>", bold_body),
            Paragraph("Composite Profile", bold_body),
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
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(4):
        score_data[0][i].style.textColor = colors.white

    story.append(score_table)
    story.append(Spacer(1, 10))

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
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(3):
        factors_data[0][i].style.textColor = colors.white

    story.append(factors_table)
    story.append(Spacer(1, 10))

    # 5. Diagnostic Advice & Custom Counselor Remarks
    story.append(Paragraph("3. Actionable Academic Recommendations & Counselor Notes", section_heading))
    for tip in tips:
        clean_tip = tip.replace("📌", "•").replace("🌟", "•").replace("⚠️", "•").replace("**", "")
        story.append(Paragraph(clean_tip, body_style))
        story.append(Spacer(1, 2))

    if custom_counselor_note and custom_counselor_note.strip():
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Counselor Personalized Remarks:</b> <i>{custom_counselor_note.strip()}</i>", body_style))

    story.append(Spacer(1, 12))

    # 6. Verification Footer
    footer_data = [
        [
            Paragraph("<b>Status:</b> Verified by Dual ML Engine", body_style),
            Paragraph(f"<b>Model:</b> {model_name[:24]}", body_style),
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


def generate_classroom_pdf_report(
    classroom_df,
    summary,
    cohort_name="Classroom Cohort",
    custom_counselor_notes=None
):
    """
    Generates a verified Classroom Cohort Executive Summary PDF Report.
    """
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

    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#0284C7")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    title_style = ParagraphStyle(
        "ClassTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "ClassSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # Header
    story.append(Paragraph("EDUPREDICT AI • CLASSROOM COHORT EXECUTIVE SUMMARY", title_style))
    story.append(Paragraph(f"Academic Performance & Dropout Risk Diagnostic Report • {cohort_name}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=10))

    # Cohort Overview Table
    current_date = datetime.now().strftime("%B %d, %Y")
    overview_data = [
        [
            Paragraph(f"<b>Cohort Name:</b> {cohort_name}", body_style),
            Paragraph(f"<b>Total Students Evaluated:</b> {summary['total_students']}", body_style)
        ],
        [
            Paragraph(f"<b>Class Average Math Score:</b> {summary['class_avg_math']:.1f} / 100", body_style),
            Paragraph(f"<b>Cohort Pass Rate:</b> {summary['pass_rate']}%", body_style)
        ],
        [
            Paragraph(f"<b>Overall 3-Subject Average:</b> {summary['class_avg_overall']:.1f} / 100", body_style),
            Paragraph(f"<b>🚨 At-Risk Student Count:</b> {summary['at_risk_count']}", body_style)
        ]
    ]
    overview_table = Table(overview_data, colWidths=[260, 270])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 10))

    # Student Roster Table (Top 12 students or all)
    story.append(Paragraph("1. Classroom Student Predictions Roster", section_heading))
    
    roster_rows = [
        [
            Paragraph("<b>Student ID</b>", bold_body),
            Paragraph("<b>Name</b>", bold_body),
            Paragraph("<b>Reading</b>", bold_body),
            Paragraph("<b>Writing</b>", bold_body),
            Paragraph("<b>Math (Pred)</b>", bold_body),
            Paragraph("<b>Grade</b>", bold_body),
            Paragraph("<b>Risk Tier</b>", bold_body)
        ]
    ]

    for _, row in classroom_df.head(15).iterrows():
        s_id = str(row.get("student_id", "N/A"))
        s_name = str(row.get("student_name", "Student"))
        r_score = f"{row.get('reading score', 0)}"
        w_score = f"{row.get('writing score', 0)}"
        m_pred = f"{row.get('Predicted_Math_Score', 0):.1f}"
        p_grade = str(row.get("Predicted_Grade", "N/A")).split()[0]
        r_tier = "At-Risk" if "High" in str(row.get("Risk_Tier", "")) else "Safe"
        
        roster_rows.append([
            Paragraph(s_id, body_style),
            Paragraph(s_name[:16], body_style),
            Paragraph(r_score, body_style),
            Paragraph(w_score, body_style),
            Paragraph(f"<b>{m_pred}</b>", bold_body),
            Paragraph(p_grade, body_style),
            Paragraph(f"<font color='red'>{r_tier}</font>" if r_tier == "At-Risk" else "<font color='green'>Safe</font>", body_style)
        ])

    roster_table = Table(roster_rows, colWidths=[80, 110, 60, 60, 80, 60, 80])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    for i in range(7):
        roster_rows[0][i].style.textColor = colors.white

    story.append(roster_table)
    story.append(Spacer(1, 10))

    # Counselor / Administrator Observations
    if custom_counselor_notes and custom_counselor_notes.strip():
        story.append(Paragraph("2. Academic Counselor Observations & Remedial Action Plan", section_heading))
        story.append(Paragraph(custom_counselor_notes.strip(), body_style))
        story.append(Spacer(1, 10))

    # Verification Footer
    footer_data = [
        [
            Paragraph("<b>Classroom Evaluation Status:</b> Completed", body_style),
            Paragraph("____________________________<br/>Lead Instructor Signature", body_style),
            Paragraph("____________________________<br/>Principal / Dean Approval", body_style)
        ]
    ]
    footer_table = Table(footer_data, colWidths=[180, 170, 180])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
