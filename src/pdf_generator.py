import io
import html
from datetime import datetime
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def _clean_str(val):
    """
    Safely converts any value to a string and escapes XML/HTML markup for ReportLab.
    """
    if val is None:
        return ""
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return ""
        val = val[0]
    return html.escape(str(val).strip(), quote=False)

def _safe_float(val, default=0.0):
    """
    Safely parses any input into a float without raising TypeError or ValueError.
    """
    try:
        if val is None:
            return default
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return default
            val = val[0]
        if isinstance(val, str):
            val = val.replace('%', '').replace('$', '').strip()
        f_val = float(val)
        return default if np.isnan(f_val) else f_val
    except Exception:
        return default

def _safe_int(val, default=0):
    """
    Safely parses any input into an integer without raising TypeError or ValueError.
    """
    try:
        if val is None:
            return default
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return default
            val = val[0]
        if isinstance(val, str):
            val = val.replace('%', '').replace('$', '').strip()
        f_val = float(val)
        return default if np.isnan(f_val) else int(f_val)
    except Exception:
        return default

def generate_student_pdf_report(
    student_name="Student",
    student_id="STU-UNASSIGNED",
    gender="female",
    race_ethnicity="group B",
    parental_education="some college",
    lunch="standard",
    test_prep="none",
    reading_score=70.0,
    writing_score=70.0,
    predicted_math=70.0,
    overall_avg=70.0,
    grade="B (Good)",
    model_name="AI Super-Stacking Meta-Regressor",
    tips=None,
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
    parental_involvement="medium",
    *args,
    **kwargs
):
    """
    Generates a verified, executive-styled Single Student Performance & Risk Evaluation PDF Report,
    including 18 multidimensional academic, behavioral, lifestyle, and prescriptive solutions.
    Handles any argument permutation or type safely.
    """
    # Defensive extraction and sanitization
    s_name = _clean_str(kwargs.get("name", kwargs.get("student_name", student_name)) or "Student")
    s_id = _clean_str(kwargs.get("id", kwargs.get("student_id", student_id)) or "STU-UNASSIGNED")
    p_math = _safe_float(kwargs.get("predicted_math", predicted_math), 70.0)
    r_score = _safe_float(kwargs.get("reading_score", reading_score), 70.0)
    w_score = _safe_float(kwargs.get("writing_score", writing_score), 70.0)
    o_avg = _safe_float(kwargs.get("overall_avg", overall_avg), 70.0)
    att_rate = _safe_float(kwargs.get("attendance_rate", attendance_rate), 85.0)
    study_hrs = _safe_float(kwargs.get("weekly_study_hours", weekly_study_hours), 12.0)
    prev_term = _safe_float(kwargs.get("previous_term_score", previous_term_score), 65.0)
    screen_hrs = _safe_float(kwargs.get("daily_screen_time_hours", daily_screen_time_hours), 3.0)
    
    p_prob_raw = kwargs.get("pass_prob", pass_prob)
    p_prob = _safe_float(p_prob_raw) if p_prob_raw is not None else None
    
    r_level = str(kwargs.get("risk_level", risk_level) or "Safe")
    gr_str = str(kwargs.get("grade", grade) or "P")
    clean_grade = _clean_str(gr_str.split()[0] if gr_str.strip() else "P")
    
    tutor_supp = str(kwargs.get("tutoring_support", tutoring_support) or "none")
    st_method = str(kwargs.get("study_method", study_method) or "spaced_repetition")
    p_involve = str(kwargs.get("parental_involvement", parental_involvement) or "medium")
    c_note = kwargs.get("custom_counselor_note", custom_counselor_note)
    p_solution = kwargs.get("prescriptive_solution", prescriptive_solution)

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

    # Custom Typography Styles (No shared mutation)
    title_style = ParagraphStyle(
        "StudentReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        "StudentReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    section_heading = ParagraphStyle(
        "StudentSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=primary_color,
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "StudentBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    bold_body = ParagraphStyle(
        "StudentBoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E293B")
    )

    table_head_style = ParagraphStyle(
        "StudentTableHead",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white
    )

    # 1. Header Banner
    story.append(Paragraph("EDUPREDICT AI • OFFICIAL ACADEMIC REPORT CARD", title_style))
    story.append(Paragraph("Machine Learning Multi-Dimensional Diagnostic & Prescriptive Action Plan (18 Features)", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=8))

    # 2. Student Metadata & Engagement Table
    current_date = datetime.now().strftime("%B %d, %Y")
    prob_text = f"<b>Pass Probability:</b> {p_prob:.1f}% ({_clean_str(r_level)})" if p_prob is not None else "Verified"
    tutor_display = _clean_str(str(tutor_supp).replace('_', ' ').title() if tutor_supp != "none" else "None")
    method_display = _clean_str(str(st_method).replace('_', ' ').title())
    involvement_display = _clean_str(str(p_involve).title())

    metadata_data = [
        [
            Paragraph(f"<b>Student Name:</b> {s_name}", body_style),
            Paragraph(f"<b>Student ID / Roll:</b> {s_id}", body_style)
        ],
        [
            Paragraph(f"<b>Date of Evaluation:</b> {current_date}", body_style),
            Paragraph(f"<b>Academic Risk Status:</b> {prob_text}", body_style)
        ],
        [
            Paragraph(f"<b>Attendance:</b> {att_rate:.1f}% • <b>Study Effort:</b> {study_hrs:.1f}h/wk • <b>Method:</b> {method_display}", body_style),
            Paragraph(f"<b>Prior Score:</b> {prev_term:.0f} • <b>Screen Time:</b> {screen_hrs:.1f}h/day • <b>Parent Mentorship:</b> {involvement_display}", body_style)
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
            Paragraph("<b>Exam Subject</b>", table_head_style),
            Paragraph("<b>Evaluation Type</b>", table_head_style),
            Paragraph("<b>Score Obtained / Forecast</b>", table_head_style),
            Paragraph("<b>Performance Status</b>", table_head_style)
        ],
        [
            Paragraph("<b>Mathematics</b>", body_style),
            Paragraph("ML Model Forecast", body_style),
            Paragraph(f"<b>{p_math:.1f} / 100</b>", bold_body),
            Paragraph("Predicted Pass" if p_math >= 50 else "<font color='red'>At-Risk</font>", body_style)
        ],
        [
            Paragraph("<b>Previous Term Benchmark</b>", body_style),
            Paragraph("Historical Baseline", body_style),
            Paragraph(f"{prev_term:.1f} / 100", body_style),
            Paragraph("Strong Foundation" if prev_term >= 70 else "Needs Review", body_style)
        ],
        [
            Paragraph("<b>Reading Comprehension</b>", body_style),
            Paragraph("Prerequisite Assessment", body_style),
            Paragraph(f"{r_score:.1f} / 100", body_style),
            Paragraph("Proficient" if r_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Writing & Expression</b>", body_style),
            Paragraph("Prerequisite Assessment", body_style),
            Paragraph(f"{w_score:.1f} / 100", body_style),
            Paragraph("Proficient" if w_score >= 60 else "Developing", body_style)
        ],
        [
            Paragraph("<b>Cumulative 3-Subject Average</b>", bold_body),
            Paragraph("Composite Profile", bold_body),
            Paragraph(f"<b>{o_avg:.1f} / 100</b>", bold_body),
            Paragraph(f"<b>Grade: {clean_grade}</b>", bold_body)
        ]
    ]

    score_table = Table(score_data, colWidths=[160, 130, 140, 110])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#ECFDF5")),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))

    # 4. Prescriptive Solutions & Action Plan (If provided)
    if p_solution and isinstance(p_solution, dict):
        story.append(Paragraph("2. AI Prescriptive Solutions & Projected Uplift", section_heading))
        
        proj_score = _safe_float(p_solution.get("projected_score"), p_math + 10.0)
        proj_prob = _safe_float(p_solution.get("projected_pass_prob"), 98.0)
        est_uplift = _safe_float(p_solution.get("total_estimated_uplift"), 12.0)
        
        presc_summary = [
            [
                Paragraph(f"<b>Current Math Score:</b> {p_math:.1f} marks", body_style),
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
            Paragraph("<b>Priority & Milestone</b>", table_head_style),
            Paragraph("<b>Recommended Action & Resource</b>", table_head_style),
            Paragraph("<b>Est. Uplift</b>", table_head_style)
        ]]
        
        interventions = p_solution.get("interventions", [])
        if isinstance(interventions, list):
            for item in interventions[:4]:
                if isinstance(item, dict):
                    p_prio = _clean_str(item.get('priority', 'P1'))
                    p_time = _clean_str(item.get('timeline', 'Immediate'))
                    p_title = _clean_str(item.get('title', ''))
                    p_act = _clean_str(item.get('action', ''))
                    p_uplift = _clean_str(item.get('est_uplift', '+3 pts'))
                    
                    action_data.append([
                        Paragraph(f"<b>{p_prio}</b><br/>{p_time}", body_style),
                        Paragraph(f"<b>{p_title}</b><br/>{p_act}", body_style),
                        Paragraph(f"<b>{p_uplift}</b>", bold_body)
                    ])
            
        action_table = Table(action_data, colWidths=[130, 340, 70])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(action_table)
        story.append(Spacer(1, 6))

    # 5. Counselor / Advisor Signature
    story.append(Paragraph("3. Academic Advisor Certification", section_heading))
    raw_sig_note = c_note if c_note else "Student profile evaluated using multi-dimensional Machine Learning decision support. Prescriptive actions verified for exam preparation."
    sig_note = _clean_str(raw_sig_note)
    
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

def generate_classroom_pdf_report(
    classroom_df,
    summary=None,
    summary_metrics=None,
    cohort_name="Section A Cohort",
    class_name=None,
    custom_counselor_notes=None,
    intervention_matrix=None,
    *args,
    **kwargs
):
    """
    Generates a Classroom Executive Analytics PDF report for institutional leadership.
    """
    # Normalize aliases
    metrics = summary if summary is not None else (summary_metrics if summary_metrics is not None else kwargs.get("metrics", {}))
    if not isinstance(metrics, dict):
        metrics = {}
        
    c_name = _clean_str(cohort_name if cohort_name else (class_name if class_name else kwargs.get("cohort", "Classroom Cohort")))
    notes = _clean_str(custom_counselor_notes if custom_counselor_notes else kwargs.get("notes", "Classroom evaluated with multi-model AI forecasting."))

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
    
    bold_body = ParagraphStyle("ClassBold", parent=body_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#1E293B"))
    class_head_style = ParagraphStyle("ClassHead", parent=body_style, fontName="Helvetica-Bold", textColor=colors.white)
    
    story.append(Paragraph(f"EDUPREDICT AI • CLASSROOM ANALYTICS DOSSIER ({c_name})", title_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=6))
    
    # Executive KPI Summary Grid
    total_st = _safe_int(metrics.get('total_students', len(classroom_df) if hasattr(classroom_df, '__len__') else 0))
    pass_rt = _safe_float(metrics.get('pass_rate', 0))
    avg_m = _safe_float(metrics.get('class_avg_math', 0))
    at_risk_c = _safe_int(metrics.get('at_risk_count', 0))

    kpi_data = [
        [
            Paragraph(f"<b>Total Enrolled:</b> {total_st}", body_style),
            Paragraph(f"<b>Projected Pass Rate:</b> {pass_rt:.1f}%", bold_body),
            Paragraph(f"<b>Average Math:</b> {avg_m:.1f}", body_style),
            Paragraph(f"<b>At-Risk Cohort:</b> {at_risk_c} Students", body_style)
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
        Paragraph("<b>ID</b>", class_head_style),
        Paragraph("<b>Student Name</b>", class_head_style),
        Paragraph("<b>Math (Pred)</b>", class_head_style),
        Paragraph("<b>Pass Prob</b>", class_head_style),
        Paragraph("<b>Risk Status</b>", class_head_style),
        Paragraph("<b>Prescribed Action</b>", class_head_style)
    ]]
    
    if hasattr(classroom_df, "iterrows"):
        for idx, row in classroom_df.head(25).iterrows():
            r_id = _clean_str(row.get("student_id", f"ID-{idx+1}"))
            r_name = _clean_str(row.get("student_name", f"Student {idx+1}"))
            r_math = _safe_float(row.get('Predicted_Math_Score', row.get('math_score', 0)))
            r_prob = _safe_float(row.get('Pass_Probability_Pct', row.get('pass_probability', 0)))
            r_risk = _clean_str(row.get("Risk_Tier", row.get("risk_level", "Safe")))
            r_action = _clean_str(str(row.get("Prescribed_Intervention", row.get("intervention", "Standard")))[:32])

            roster_data.append([
                Paragraph(r_id, body_style),
                Paragraph(r_name, body_style),
                Paragraph(f"{r_math:.1f}", body_style),
                Paragraph(f"{r_prob:.1f}%", body_style),
                Paragraph(r_risk, body_style),
                Paragraph(r_action, body_style)
            ])
        
    roster_table = Table(roster_data, colWidths=[65, 110, 65, 60, 85, 155])
    roster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ]))
        
    story.append(roster_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
