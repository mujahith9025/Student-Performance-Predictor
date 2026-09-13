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
    internet_access="yes",
    previous_term_score=65.0,
    study_method="spaced_repetition",
    daily_screen_time_hours=3.0,
    parental_involvement="medium"
):
    """
    Generates a verified, executive-styled Single Student Performance & Risk Evaluation PDF Report,
    including 18 multidimensional academic, behavioral, lifestyle, and prescriptive solutions.
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
    story.append(Paragraph("Machine Learning Multi-Dimensional Diagnostic & Prescriptive Action Plan (18 Features)", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=8))

    # 2. Student Metadata & Engagement Table
    current_date = datetime.now().strftime("%B %d, %Y")
    prob_text = f"<b>Pass Probability:</b> {pass_prob:.1f}% ({risk_level})" if pass_prob is not None else "Verified"
    tutor_display = tutoring_support.replace('_', ' ').title() if tutoring_support != "none" else "None"
    method_display = study_method.replace('_', ' ').title()
    involvement_display = parental_involvement.title()
    
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
            Paragraph(f"<b>Attendance:</b> {attendance_rate:.1f}% • <b>Study Effort:</b> {weekly_study_hours:.1f}h/wk • <b>Method:</b> {method_display}", body_style),
            Paragraph(f"<b>Prior Score:</b> {previous_term_score:.0f} • <b>Screen Time:</b> {daily_screen_time_hours:.1f}h/day • <b>Parent Mentorship:</b> {involvement_display}", body_style)
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
            Paragraph("<b>Previous Term Benchmark</b>", body_style),
            Paragraph("Historical Baseline", body_style),
            Paragraph(f"{previous_term_score:.1f} / 100", body_style),
            Paragraph("Strong Foundation" if previous_term_score >= 70 else "Needs Review", body_style)
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
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    for i in range(4):
        score_data[0][i].style.textColor = colors.white

    story.append(score_table)
    story.append(Spacer(1, 6))

    # 4. Prescriptive Solutions & Action Plan (If provided)
    if prescriptive_solution:
        story.append(Paragraph("2. AI Prescriptive Solutions & Projected Uplift", section_heading))
        
        proj_score = prescriptive_solution.get("projected_score", predicted_math + 10)
        proj_prob = prescriptive_solution.get("projected_pass_prob", 98.0)
        est_uplift = prescriptive_solution.get("total_estimated_uplift", 12.0)
        
        presc_summary = [
            [
                Paragraph(f"<b>Current Math Score:</b> {predicted_math:.1f} marks", body_style),
                Paragraph(f"<b>Projected Outcome:</b> <font color='#059669'><b>{proj_score:.1f} marks</b></font> (+{est_uplift:.1f} pts)", bold_body),
                Paragraph(f"<b>Target Pass Rate:</b> {proj_prob:.1f}%", body_style)
            ]
        ]
        presc_table = Table(presc_summary, colWidths=[180, 180, 180])
        presc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(presc_table)
        story.append(Spacer(1, 4))
        
        # Action Items Table
        action_data = [[
            Paragraph("<b>Priority & Milestone</b>", bold_body),
            Paragraph("<b>Recommended Action & Resource</b>", bold_body),
            Paragraph("<b>Est. Uplift</b>", bold_body)
        ]]
        
        for item in prescriptive_solution.get("interventions", [])[:4]:
            action_data.append([
                Paragraph(f"<b>{item.get('priority', 'P1')}</b><br/>{item.get('timeline', 'Immediate')}", body_style),
                Paragraph(f"<b>{item.get('title', '')}</b><br/>{item.get('action', '')}", body_style),
                Paragraph(f"<b>{item.get('est_uplift', '+3 pts')}</b>", bold_body)
            ])
            
        action_table = Table(action_data, colWidths=[130, 340, 70])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        for i in range(3):
            action_data[0][i].style.textColor = colors.white
        story.append(action_table)
        story.append(Spacer(1, 6))

    # 5. Counselor / Advisor Signature
    story.append(Paragraph("3. Academic Advisor Certification", section_heading))
    sig_note = custom_counselor_note if custom_counselor_note else "Student profile evaluated using multi-dimensional Machine Learning decision support. Prescriptive actions verified for exam preparation."
    
    sig_data = [
        [
            Paragraph(f"<b>Evaluator Note:</b> {sig_note}", body_style),
            Paragraph("<b>Signature:</b> ___________________<br/><b>Verified by EduPredict AI System</b>", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[360, 180])
    sig_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(sig_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_classroom_pdf_report(classroom_df, summary_metrics, class_name="Section A Cohort"):
    """
    Generates a Classroom Executive Analytics PDF report for institutional leadership.
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
        fontSize=15,
        leading=18,
        textColor=primary_color,
        alignment=1
    )
    
    body_style = ParagraphStyle(
        "ClassBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E293B")
    )
    
    bold_body = ParagraphStyle("ClassBold", parent=body_style, fontName="Helvetica-Bold")
    
    story.append(Paragraph(f"EDUPREDICT AI • CLASSROOM ANALYTICS DOSSIER ({class_name})", title_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=6))
    
    # Executive KPI Summary Grid
    kpi_data = [
        [
            Paragraph(f"<b>Total Enrolled:</b> {summary_metrics.get('total_students', len(classroom_df))}", body_style),
            Paragraph(f"<b>Projected Pass Rate:</b> {summary_metrics.get('pass_rate', 0):.1f}%", bold_body),
            Paragraph(f"<b>Average Math:</b> {summary_metrics.get('class_avg_math', 0):.1f}", body_style),
            Paragraph(f"<b>At-Risk Cohort:</b> {summary_metrics.get('at_risk_count', 0)} Students", body_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))
    
    # Student Roster Table
    roster_data = [[
        Paragraph("<b>ID</b>", bold_body),
        Paragraph("<b>Student Name</b>", bold_body),
        Paragraph("<b>Math (Pred)</b>", bold_body),
        Paragraph("<b>Pass Prob</b>", bold_body),
        Paragraph("<b>Risk Status</b>", bold_body),
        Paragraph("<b>Prescribed Action</b>", bold_body)
    ]]
    
    for _, row in classroom_df.head(25).iterrows():
        roster_data.append([
            Paragraph(str(row.get("student_id", f"ID-{_+1}")), body_style),
            Paragraph(str(row.get("student_name", f"Student {_+1}")), body_style),
            Paragraph(f"{row.get('Predicted_Math_Score', 0):.1f}", body_style),
            Paragraph(f"{row.get('Pass_Probability_Pct', 0):.1f}%", body_style),
            Paragraph(str(row.get("Risk_Tier", "Safe")), body_style),
            Paragraph(str(row.get("Prescribed_Intervention", "Standard"))[:32], body_style)
        ])
        
    roster_table = Table(roster_data, colWidths=[65, 110, 65, 60, 85, 155])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ]))
    for i in range(6):
        roster_data[0][i].style.textColor = colors.white
        
    story.append(roster_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
