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
    custom_counselor_note=None,
    prescriptive_solution=None,
    attendance_rate=85.0,
    weekly_study_hours=12.0,
    sleep_hours_per_day=7.5,
    past_failures=0,
    tutoring_support="none",
    internet_access="yes"
):
    """
    Generates a verified, executive-styled Single Student Performance & Risk Evaluation PDF Report,
    including 14 multidimensional academic, behavioral, lifestyle, and prescriptive solutions.
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
        fontSize=16,
        leading=20,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=primary_color,
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # 1. Header Banner
    story.append(Paragraph("EDUPREDICT AI • OFFICIAL ACADEMIC REPORT CARD", title_style))
    story.append(Paragraph("Machine Learning Multi-Dimensional Diagnostic & Prescriptive Action Plan", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=8))

    # 2. Student Metadata & Engagement Table
    current_date = datetime.now().strftime("%B %d, %Y")
    prob_text = f"<b>Pass Probability:</b> {pass_prob:.1f}% ({risk_level})" if pass_prob is not None else "Verified"
    tutor_display = tutoring_support.replace('_', ' ').title() if tutoring_support != "none" else "None"
    
    metadata_data = [
        [
            Paragraph("<b>Student Name:</b> " + str(student_name), body_style),
            Paragraph("<b>Student ID / Roll:</b> " + str(student_id), body_style)
        ],
        [
            Paragraph("<b>Date of Evaluation:</b> " + current_date, body_style),
            Paragraph("<b>Academic Risk Status:</b> " + prob_text, body_style)
        ],
        [
            Paragraph(f"<b>Attendance:</b> {attendance_rate:.1f}% • <b>Study Effort:</b> {weekly_study_hours:.1f}h/wk", body_style),
            Paragraph(f"<b>Tutoring:</b> {tutor_display} • <b>Past Backlogs:</b> {past_failures} • <b>Sleep:</b> {sleep_hours_per_day:.1f}h/day", body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

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

    score_table = Table(score_data, colWidths=[160, 130, 140, 110])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#ECFDF5")),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(4):
        score_data[0][i].style.textColor = colors.white

    story.append(score_table)
    story.append(Spacer(1, 6))

    # 4. Prescriptive Solutions & Action Plan (If provided)
    if prescriptive_solution:
        story.append(Paragraph("2. AI Prescriptive Solutions & Projected Uplift", section_heading))
        
        proj_score = prescriptive_solution.get("projected_score", predicted_math)
        proj_grade = prescriptive_solution.get("projected_grade", grade)
        uplift_pts = prescriptive_solution.get("total_uplift", 0.0)
        
        rx_header = [
            [
                Paragraph(f"<b>Prescribed Weekly Study Plan:</b> {prescriptive_solution.get('total_study_hours', 12):.0f} hrs/week", body_style),
                Paragraph(f"<b>Projected Outcome:</b> <font color='#059669'><b>{proj_score:.1f}/100 ({proj_grade})</b></font> (+{uplift_pts:.1f} pts)", body_style)
            ]
        ]
        rx_meta_table = Table(rx_header, colWidths=[270, 270])
        rx_meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#86EFAC")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(rx_meta_table)
        story.append(Spacer(1, 4))
        
        # Interventions List
        interv_rows = [
            [
                Paragraph("<b>Priority Intervention</b>", bold_body),
                Paragraph("<b>Timeline</b>", bold_body),
                Paragraph("<b>Uplift</b>", bold_body),
                Paragraph("<b>Action Required</b>", bold_body)
            ]
        ]
        for itm in prescriptive_solution.get("interventions", [])[:3]:
            interv_rows.append([
                Paragraph(itm["title"], bold_body),
                Paragraph(itm["timeline"], body_style),
                Paragraph(f"<font color='#059669'><b>{itm['est_uplift']}</b></font>", body_style),
                Paragraph(itm["action"], body_style)
            ])
            
        interv_table = Table(interv_rows, colWidths=[140, 85, 65, 250])
        interv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        for i in range(4):
            interv_rows[0][i].style.textColor = colors.white
        story.append(interv_table)
        story.append(Spacer(1, 6))

    # 5. Diagnostic Advice & Custom Counselor Remarks
    story.append(Paragraph("3. Counselor Observations & Advisory Guidance", section_heading))
    for tip in tips[:2]:
        clean_tip = tip.replace("📌", "•").replace("🌟", "•").replace("⚠️", "•").replace("**", "")
        story.append(Paragraph(clean_tip, body_style))
        story.append(Spacer(1, 2))

    if custom_counselor_note and custom_counselor_note.strip():
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"<b>Counselor Notes:</b> <i>{custom_counselor_note.strip()}</i>", body_style))

    story.append(Spacer(1, 8))

    # 6. Verification Footer
    footer_data = [
        [
            Paragraph("<b>Status:</b> Verified by Dual ML Engine", body_style),
            Paragraph(f"<b>Model:</b> {model_name[:24]}", body_style),
            Paragraph("____________________________<br/>Academic Counselor Signature", body_style)
        ]
    ]
    footer_table = Table(footer_data, colWidths=[180, 180, 180])
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
    custom_counselor_notes=None,
    intervention_matrix=None
):
    """
    Generates a verified Classroom Cohort Executive Summary PDF Report with Prescriptive Intervention Matrix.
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
        fontSize=16,
        leading=20,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "ClassSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=primary_color,
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    # Header
    story.append(Paragraph("EDUPREDICT AI • CLASSROOM COHORT EXECUTIVE SUMMARY", title_style))
    story.append(Paragraph(f"Academic Diagnostics & Prescriptive Intervention Report • {cohort_name}", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=8))

    # Cohort Overview Table
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
    overview_table = Table(overview_data, colWidths=[270, 270])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 6))

    # Student Roster Table (Top 12 students)
    story.append(Paragraph("1. Classroom Student Predictions & Prescribed Interventions", section_heading))
    
    roster_rows = [
        [
            Paragraph("<b>Student ID</b>", bold_body),
            Paragraph("<b>Name</b>", bold_body),
            Paragraph("<b>Reading</b>", bold_body),
            Paragraph("<b>Writing</b>", bold_body),
            Paragraph("<b>Math (Pred)</b>", bold_body),
            Paragraph("<b>Grade</b>", bold_body),
            Paragraph("<b>Prescribed Action</b>", bold_body)
        ]
    ]

    for _, row in classroom_df.head(12).iterrows():
        s_id = str(row.get("student_id", "N/A"))
        s_name = str(row.get("student_name", "Student"))
        r_score = f"{row.get('reading score', 0)}"
        w_score = f"{row.get('writing score', 0)}"
        m_pred = f"{row.get('Predicted_Math_Score', 0):.1f}"
        p_grade = str(row.get("Predicted_Grade", "N/A")).split()[0]
        presc_act = str(row.get("Prescribed_Intervention", "Standard Study"))
        
        roster_rows.append([
            Paragraph(s_id, body_style),
            Paragraph(s_name[:15], body_style),
            Paragraph(r_score, body_style),
            Paragraph(w_score, body_style),
            Paragraph(f"<b>{m_pred}</b>", bold_body),
            Paragraph(p_grade, body_style),
            Paragraph(f"<font size='7'>{presc_act}</font>", body_style)
        ])

    roster_table = Table(roster_rows, colWidths=[70, 95, 45, 45, 65, 50, 170])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    for i in range(7):
        roster_rows[0][i].style.textColor = colors.white

    story.append(roster_table)
    story.append(Spacer(1, 6))

    # Classroom Prescriptive Intervention Matrix
    if intervention_matrix:
        story.append(Paragraph("2. Targeted Classroom Intervention Cohorts", section_heading))
        summary_m = intervention_matrix.get("summary", {})
        action_plans = intervention_matrix.get("action_plans", {})
        
        cohort_data = [
            [
                Paragraph("<b>Intervention Cohort</b>", bold_body),
                Paragraph("<b>Count</b>", bold_body),
                Paragraph("<b>Recommended Institutional Action Plan</b>", bold_body)
            ],
            [
                Paragraph("🚨 Intensive Remedial", bold_body),
                Paragraph(f"<b>{summary_m.get('high_risk_count', 0)} students</b>", body_style),
                Paragraph(action_plans.get("Intensive Remedial (High Risk)", ""), body_style)
            ],
            [
                Paragraph("🎯 Test Prep Bootcamp", bold_body),
                Paragraph(f"<b>{summary_m.get('test_prep_needed_count', 0)} students</b>", body_style),
                Paragraph(action_plans.get("Test Prep Bootcamp (Moderate Gap)", ""), body_style)
            ],
            [
                Paragraph("📚 Verbal & Reading", bold_body),
                Paragraph(f"<b>{summary_m.get('verbal_support_count', 0)} students</b>", body_style),
                Paragraph(action_plans.get("Verbal / Reading Support", ""), body_style)
            ],
            [
                Paragraph("🏆 Honors & Distinction", bold_body),
                Paragraph(f"<b>{summary_m.get('honors_count', 0)} students</b>", body_style),
                Paragraph(action_plans.get("Honors / Distinction Mentorship", ""), body_style)
            ]
        ]
        
        cohort_table = Table(cohort_data, colWidths=[130, 70, 340])
        cohort_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        for i in range(3):
            cohort_data[0][i].style.textColor = colors.white
        story.append(cohort_table)
        story.append(Spacer(1, 6))

    # Counselor / Administrator Observations
    if custom_counselor_notes and custom_counselor_notes.strip():
        story.append(Paragraph("3. Instructor & Counselor Observations", section_heading))
        story.append(Paragraph(custom_counselor_notes.strip(), body_style))
        story.append(Spacer(1, 6))

    # Verification Footer
    footer_data = [
        [
            Paragraph("<b>Classroom Evaluation Status:</b> Completed", body_style),
            Paragraph("____________________________<br/>Lead Instructor Signature", body_style),
            Paragraph("____________________________<br/>Principal / Dean Approval", body_style)
        ]
    ]
    footer_table = Table(footer_data, colWidths=[180, 180, 180])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
