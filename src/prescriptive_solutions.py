import numpy as np
import pandas as pd

def diagnose_student_weaknesses(profile, predicted_math, pass_prob=None):
    """
    Diagnoses root academic, behavioral, lifestyle, and socioeconomic bottlenecks across 18 dimensions.
    """
    bottlenecks = []
    r = float(profile.get("reading score", 65))
    w = float(profile.get("writing score", 65))
    prev = float(profile.get("previous_term_score", 65))
    att = float(profile.get("attendance_rate", 85))
    study = float(profile.get("weekly_study_hours", 12))
    sleep = float(profile.get("sleep_hours_per_day", 7.5))
    screen = float(profile.get("daily_screen_time_hours", 3.0))
    fails = int(profile.get("past_failures", 0))
    
    prep = profile.get("test preparation course", "none")
    lunch = profile.get("lunch", "standard")
    internet = profile.get("internet_access", "yes")
    tutoring = profile.get("tutoring_support", "none")
    method = profile.get("study_method", "spaced_repetition")
    involvement = profile.get("parental_involvement", "medium")
    
    # 1. Critical Attendance Risk
    if att < 75.0:
        bottlenecks.append({
            "category": "Severe Attendance / Absenteeism Risk",
            "severity": "High",
            "icon": "🚨",
            "detail": f"Attendance rate is {att:.1f}% (below the 75% critical threshold). Chronic absenteeism causes severe concept gaps and is the primary leading indicator of course failure."
        })
    elif att < 85.0:
        bottlenecks.append({
            "category": "Moderate Attendance Inconsistency",
            "severity": "Medium",
            "icon": "⚠️",
            "detail": f"Attendance is {att:.1f}%. Missed classroom lectures create friction in multi-step problem solving and retention."
        })
        
    # 2. Digital Distraction & Screen Time Load
    if screen > 4.5:
        bottlenecks.append({
            "category": "High Screen Time & Digital Distraction",
            "severity": "High",
            "icon": "📱",
            "detail": f"Daily screen time of {screen:.1f} hrs exceeds healthy limits. Digital media fragmentation reduces deep focus capacity and displaces deliberate practice."
        })
        
    # 3. Passive Study Technique Fragility
    if method == "passive_reading":
        bottlenecks.append({
            "category": "Inefficient Passive Study Method",
            "severity": "High",
            "icon": "📖",
            "detail": "Relying on passive re-reading and highlighting yields low long-term retention. Transitioning to Active Recall and Practice Questions produces superior results."
        })
        
    # 4. Study Hours & Practice Deficit
    if study < 6.0:
        bottlenecks.append({
            "category": "Insufficient Deliberate Study Hours",
            "severity": "High",
            "icon": "⏱️",
            "detail": f"Student reports only {study:.1f} hrs/week of self-study. A minimum of 10-14 weekly hours is required for mathematical concept consolidation."
        })
    elif study < 10.0:
        bottlenecks.append({
            "category": "Moderate Practice Deficit",
            "severity": "Medium",
            "icon": "📖",
            "detail": f"Current self-study of {study:.1f} hrs/week is below the recommended 12-16 hrs benchmark for top tier mastery."
        })
        
    # 5. Past Course Backlogs / Failures & Prior Momentum
    if fails >= 2:
        bottlenecks.append({
            "category": "Accumulated Academic Backlogs",
            "severity": "High",
            "icon": "📉",
            "detail": f"Student has {fails} prior course backlogs. Foundational prerequisites from previous terms require targeted diagnostic re-teaching."
        })
    elif fails == 1:
        bottlenecks.append({
            "category": "Prior Subject Backlog",
            "severity": "Medium",
            "icon": "🔄",
            "detail": "1 past course backlog indicates localized concept fragility that must be resolved prior to final examinations."
        })
        
    if prev < 50.0:
        bottlenecks.append({
            "category": "Historical Foundation Deficit",
            "severity": "High",
            "icon": "📊",
            "detail": f"Previous term performance ({prev:.0f} marks) indicates foundational concept gaps requiring systematic review before advanced topics."
        })
        
    # 6. Sleep & Cognitive Fatigue Factor
    if sleep < 6.0:
        bottlenecks.append({
            "category": "Chronic Sleep Deprivation & Fatigue",
            "severity": "High",
            "icon": "😴",
            "detail": f"Student averages only {sleep:.1f} hrs of daily sleep. Sleep deprivation degrades working memory, problem-solving speed, and exam focus."
        })
        
    # 7. Quantitative Math Deficiency
    if predicted_math < 50:
        bottlenecks.append({
            "category": "Critical Mathematics Deficit",
            "severity": "High",
            "icon": "🚨",
            "detail": f"Predicted math score ({predicted_math:.1f}) is below passing threshold (50.0). Immediate 1-on-1 tutoring and remedial clinics required."
        })
    elif predicted_math < 65:
        bottlenecks.append({
            "category": "Moderate Mathematical Gap",
            "severity": "Medium",
            "icon": "⚠️",
            "detail": f"Predicted math score ({predicted_math:.1f}) indicates good foundation but difficulty with complex multi-step problems."
        })
        
    # 8. Verbal Comprehension Lag
    if r < 60:
        bottlenecks.append({
            "category": "Reading Literacy & Word Problem Lag",
            "severity": "Medium",
            "icon": "📚",
            "detail": f"Reading score ({r:.0f}) is limiting mathematical word problem interpretation and problem translation."
        })
        
    # 9. Test Prep, Mentorship & Tutoring Deficit
    if prep == "none":
        bottlenecks.append({
            "category": "Uncompleted Test Prep Course",
            "severity": "Medium",
            "icon": "🎯",
            "detail": "Student has not completed the standardized exam prep course, missing out on ~5-8 points of empirical score boost."
        })
        
    if involvement == "low":
        bottlenecks.append({
            "category": "Low Home Academic Engagement",
            "severity": "Medium",
            "icon": "🏡",
            "detail": "Low home mentorship engagement. Scheduling structured bi-weekly academic progress check-ins creates positive accountability."
        })
        
    # 10. Digital & Nutrition Support
    if internet == "no":
        bottlenecks.append({
            "category": "Digital Access Barrier",
            "severity": "Medium",
            "icon": "🌐",
            "detail": "Lack of home high-speed internet impedes access to online practice portals and video walkthroughs."
        })
        
    if lunch == "free/reduced":
        bottlenecks.append({
            "category": "Nutritional & Study Environment Support",
            "severity": "Low",
            "icon": "🍎",
            "detail": "Free/reduced lunch indicates potential household socioeconomic headwinds; student would benefit from quiet study hall access and meal support."
        })
        
    return bottlenecks

def generate_prescriptive_solution(profile, predicted_math, pass_prob=None, grade=None, target_score=None):
    """
    Generates actionable prescriptive solutions, study hour allocations, milestone timeline,
    and projected score uplift across 18 student dimensions.
    """
    r = float(profile.get("reading score", 65))
    w = float(profile.get("writing score", 65))
    prev = float(profile.get("previous_term_score", 65))
    att = float(profile.get("attendance_rate", 85))
    study = float(profile.get("weekly_study_hours", 12))
    sleep = float(profile.get("sleep_hours_per_day", 7.5))
    screen = float(profile.get("daily_screen_time_hours", 3.0))
    fails = int(profile.get("past_failures", 0))
    prep = profile.get("test preparation course", "none")
    lunch = profile.get("lunch", "standard")
    internet = profile.get("internet_access", "yes")
    tutoring = profile.get("tutoring_support", "none")
    method = profile.get("study_method", "spaced_repetition")
    involvement = profile.get("parental_involvement", "medium")
    
    current_score = float(predicted_math)
    if target_score is None:
        target_score = min(100.0, current_score + (14.0 if current_score < 70 else 9.0))
        
    bottlenecks = diagnose_student_weaknesses(profile, predicted_math, pass_prob)
    
    interventions = []
    total_uplift = 0.0
    
    # 1. Study Technique Optimization (Active Recall & Problem Solving)
    if method == "passive_reading" or method == "group_study":
        uplift = 4.2 if method == "passive_reading" else 2.5
        total_uplift += uplift
        interventions.append({
            "title": "Upgrade to Active Problem Solving & Spaced Recall",
            "priority": "P1 - Cognitive Mastery",
            "timeline": "Immediate (Daily)",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Replace passive re-reading with active practice problems, timed past paper questions, and spaced flashcards.",
            "resource": "Digital Question Bank & Spaced Recall Platform"
        })
        
    # 2. Digital Distraction & Screen Time Management
    if screen > 3.5:
        uplift = round(min(4.0, (screen - 2.5) * 1.2), 1)
        total_uplift += uplift
        interventions.append({
            "title": f"Digital Screen Diet ({screen:.1f}h -> 2.5h/day)",
            "priority": "P1 - Focus & Attention",
            "timeline": "Weeks 1 - 4",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Implement app timers and 'Do Not Disturb' study blocks to eliminate notification fragmentation during study sessions.",
            "resource": "Digital Focus & Screen Time Management Plan"
        })
        
    # 3. Attendance & Classroom Continuity
    if att < 85.0:
        uplift = round(min(7.5, (92.0 - att) * 0.35), 1)
        total_uplift += uplift
        interventions.append({
            "title": f"Attendance Recovery Plan ({att:.0f}% -> 92%)",
            "priority": "P1 - Critical Foundation",
            "timeline": "Immediate (Weeks 1 - 4)",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Establish daily attendance tracking with automated SMS/email check-ins to bridge concept continuity gaps.",
            "resource": "Student Attendance Portal & Advisor Check-In"
        })
        
    # 4. Tutoring & Remedial Program
    if current_score < 65 or fails > 0 or tutoring == "none":
        uplift = 6.0 if current_score < 50 else 4.0
        total_uplift += uplift
        interventions.append({
            "title": "Enroll in 1-on-1 Peer & Teacher Tutoring (3 hrs/wk)",
            "priority": "P1 - Remedial",
            "timeline": "Weeks 1 - 8",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Pair with top-performing math peer tutor for twice-weekly 90-minute structured problem-solving sessions.",
            "resource": "Math Lab & Peer Tutoring Center"
        })
        
    # 5. Test Prep Course Completion
    if prep == "none":
        uplift = 5.0
        total_uplift += uplift
        interventions.append({
            "title": "Complete Standardized Test Preparation Course",
            "priority": "P2 - High Impact",
            "timeline": "Weeks 2 - 6",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Complete 4-week exam prep curriculum focusing on timed mock exams, formula memorization, and question triage.",
            "resource": "Online Test Prep Portal & Practice Question Bank"
        })
        
    # 6. Weekly Study Hours Increase
    if study < 14.0:
        recommended_study = min(20.0, study + 6.0)
        uplift = round((recommended_study - study) * 0.38, 1)
        total_uplift += uplift
        interventions.append({
            "title": f"Expand Deliberate Study ({study:.0f}h -> {recommended_study:.0f}h/wk)",
            "priority": "P2 - Skill Practice",
            "timeline": "Weeks 1 - 12",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": f"Allocate +{recommended_study - study:.0f} dedicated hours weekly using the Pomodoro technique (25 min study / 5 min break).",
            "resource": "Weekly Study Timetable Planner"
        })
        
    # 7. Sleep & Wellness Optimization
    if sleep < 6.5:
        uplift = 3.0
        total_uplift += uplift
        interventions.append({
            "title": "Sleep & Cognitive Wellness Protocol (7.5+ hrs)",
            "priority": "P3 - Cognitive Wellness",
            "timeline": "Daily Habit",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Adopt consistent 11 PM bedtime routine, eliminating screen time 30 mins before sleep to maximize memory consolidation.",
            "resource": "Student Wellness & Sleep Hygiene Guide"
        })
        
    # 8. Home Academic Mentorship & Check-ins
    if involvement == "low":
        uplift = 2.8
        total_uplift += uplift
        interventions.append({
            "title": "Bi-Weekly Home Mentorship & Progress Review",
            "priority": "P3 - Accountability",
            "timeline": "Ongoing",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Schedule 20-minute bi-weekly review sessions between student, parent, and advisor to maintain momentum.",
            "resource": "Parent-Teacher Communication Portal"
        })
        
    # Study Hours Allocation Breakdown
    base_hours = max(10.0, study + 4.0) if current_score < 60 else max(8.0, study + 2.0)
    math_pct = 0.50 if current_score < r else 0.40
    reading_pct = 0.30 if r < w else 0.35
    writing_pct = max(0.15, 1.0 - math_pct - reading_pct)
    
    study_hours = {
        "Mathematics Practice": round(base_hours * math_pct, 1),
        "Reading Comprehension": round(base_hours * reading_pct, 1),
        "Writing & Essay Drills": round(base_hours * writing_pct, 1),
        "Mock Exam & Review": 2.5
    }
    total_study_hours = sum(study_hours.values())
    
    # Projected Post-Intervention Score
    projected_score = min(100.0, round(current_score + min(total_uplift, 28.0), 1))
    projected_prob = min(99.9, round(pass_prob + (15.0 if pass_prob < 80 else 4.0), 1)) if pass_prob else (98.0 if projected_score >= 60 else 75.0)
    
    # Projected Risk Tier
    if projected_score >= 80 or projected_prob >= 85:
        projected_risk = "Safe / Low Risk"
    elif projected_score >= 60 or projected_prob >= 60:
        projected_risk = "Moderate (Monitor)"
    else:
        projected_risk = "High Academic Risk"
        
    # Structured 12-Week Roadmap Milestones
    milestones = [
        {
            "phase": "Weeks 1 - 3: Baseline Diagnostics & Foundational Stabilization",
            "focus": "Diagnostic quiz, attendance recovery, and study schedule setup.",
            "target": f"Eliminate absenteeism, achieve 7.5h sleep/night, and reach {min(100.0, current_score + 4.0):.1f} marks in weekly checks."
        },
        {
            "phase": "Weeks 4 - 8: Core Concept Reinforcement & Active Practice Drills",
            "focus": "1-on-1 tutoring, active problem solving, and completing standardized test preparation modules.",
            "target": f"Master quadratic equations, trigonometry, and reach {min(100.0, current_score + 9.0):.1f} marks in mid-term mocks."
        },
        {
            "phase": "Weeks 9 - 12: Timed Exam Simulation & Distinction Mastery",
            "focus": "Full-length past paper simulations, speed drills, and exam triage strategies.",
            "target": f"Consolidate scores to projected {projected_score:.1f} marks ({projected_risk})."
        }
    ]
    
    return {
        "bottlenecks": bottlenecks,
        "interventions": interventions,
        "total_estimated_uplift": round(total_uplift, 1),
        "current_score": current_score,
        "projected_score": projected_score,
        "projected_pass_prob": projected_prob,
        "projected_risk_tier": projected_risk,
        "weekly_study_allocation": study_hours,
        "total_prescribed_hours": total_study_hours,
        "milestones": milestones
    }

def generate_classroom_intervention_matrix(cohort_df):
    """
    Generates prescriptive intervention matrix for a cohort of students.
    """
    matrix_rows = []
    
    high_risk_count = 0
    test_prep_needed_count = 0
    tutoring_needed_count = 0
    screen_distracted_count = 0
    honors_count = 0
    
    for _, row in cohort_df.iterrows():
        p_dict = row.to_dict()
        pred_math = float(p_dict.get("Predicted_Math_Score", p_dict.get("math score", 65)))
        prob = float(p_dict.get("Pass_Probability_Pct", 95.0))
        
        # Diagnostics
        diag = diagnose_student_weaknesses(p_dict, pred_math, prob)
        
        # Prescriptive assignment
        if pred_math < 50 or prob < 60:
            rec_action = "🚨 Intensive 1-on-1 Remedial & Attendance Tracking"
            urgency = "Immediate (Week 1)"
            high_risk_count += 1
        elif p_dict.get("daily_screen_time_hours", 3.0) > 4.5:
            rec_action = "📱 Digital Distraction & Focus Intervention"
            urgency = "High (Week 1)"
            screen_distracted_count += 1
        elif p_dict.get("test preparation course", "none") == "none":
            rec_action = "🎯 Test Prep Course Enrollment & Mock Drills"
            urgency = "High (Weeks 2 - 4)"
            test_prep_needed_count += 1
        elif float(p_dict.get("weekly_study_hours", 12)) < 10.0:
            rec_action = "⏱️ Expand Study Hours & Peer Study Pod"
            urgency = "Moderate"
            tutoring_needed_count += 1
        else:
            rec_action = "🌟 Honors / Advanced Problem-Solving Track"
            urgency = "Routine"
            honors_count += 1
            
        matrix_rows.append({
            "Student Name": p_dict.get("student_name", f"Student {_ + 1}"),
            "Roll / ID": p_dict.get("student_id", f"ID-{_ + 101}"),
            "Predicted Math": f"{pred_math:.1f}",
            "Risk Level": p_dict.get("Risk_Tier", "Safe"),
            "Primary Bottleneck": diag[0]["category"] if diag else "None Identified",
            "Recommended Prescription": rec_action,
            "Urgency": urgency
        })
        
    cluster_summary = {
        "high_risk_count": high_risk_count,
        "test_prep_needed_count": test_prep_needed_count,
        "tutoring_needed_count": tutoring_needed_count,
        "screen_distracted_count": screen_distracted_count,
        "honors_count": honors_count,
        "total_students": len(cohort_df)
    }
    
    return pd.DataFrame(matrix_rows), cluster_summary
