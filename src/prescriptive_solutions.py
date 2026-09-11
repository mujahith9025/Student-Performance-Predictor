import numpy as np
import pandas as pd

def diagnose_student_weaknesses(profile, predicted_math, pass_prob=None):
    """
    Diagnoses root academic, behavioral, lifestyle, and socioeconomic bottlenecks across 14 dimensions.
    """
    bottlenecks = []
    r = float(profile.get("reading score", 65))
    w = float(profile.get("writing score", 65))
    att = float(profile.get("attendance_rate", 85))
    study = float(profile.get("weekly_study_hours", 12))
    sleep = float(profile.get("sleep_hours_per_day", 7.5))
    fails = int(profile.get("past_failures", 0))
    
    prep = profile.get("test preparation course", "none")
    lunch = profile.get("lunch", "standard")
    internet = profile.get("internet_access", "yes")
    tutoring = profile.get("tutoring_support", "none")
    
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
        
    # 2. Study Hours & Practice Deficit
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
        
    # 3. Past Course Backlogs / Failures
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
        
    # 4. Sleep & Cognitive Fatigue Factor
    if sleep < 6.0:
        bottlenecks.append({
            "category": "Chronic Sleep Deprivation & Fatigue",
            "severity": "High",
            "icon": "😴",
            "detail": f"Student averages only {sleep:.1f} hrs of daily sleep. Sleep deprivation degrades working memory, problem-solving speed, and exam focus by up to 25%."
        })
        
    # 5. Quantitative Math Deficiency
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
        
    # 6. Verbal Comprehension Lag
    if r < 60:
        bottlenecks.append({
            "category": "Reading Literacy & Word Problem Lag",
            "severity": "Medium",
            "icon": "📚",
            "detail": f"Reading score ({r:.0f}) is limiting mathematical word problem interpretation and problem translation."
        })
        
    # 7. Test Prep & Tutoring Deficit
    if prep == "none":
        bottlenecks.append({
            "category": "Uncompleted Test Prep Course",
            "severity": "Medium",
            "icon": "🎯",
            "detail": "Student has not completed the standardized exam prep course, missing out on ~5-8 points of empirical score boost."
        })
        
    # 8. Digital & Nutrition Support
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
    and projected score uplift across 14 student dimensions.
    """
    r = float(profile.get("reading score", 65))
    w = float(profile.get("writing score", 65))
    att = float(profile.get("attendance_rate", 85))
    study = float(profile.get("weekly_study_hours", 12))
    sleep = float(profile.get("sleep_hours_per_day", 7.5))
    fails = int(profile.get("past_failures", 0))
    prep = profile.get("test preparation course", "none")
    lunch = profile.get("lunch", "standard")
    internet = profile.get("internet_access", "yes")
    tutoring = profile.get("tutoring_support", "none")
    
    current_score = float(predicted_math)
    if target_score is None:
        target_score = min(100.0, current_score + (14.0 if current_score < 70 else 9.0))
        
    bottlenecks = diagnose_student_weaknesses(profile, predicted_math, pass_prob)
    
    interventions = []
    total_uplift = 0.0
    
    # 1. Attendance & Classroom Continuity
    if att < 85.0:
        uplift = round(min(8.0, (92.0 - att) * 0.35), 1)
        total_uplift += uplift
        interventions.append({
            "title": f"Attendance Recovery Plan ({att:.0f}% -> 92%)",
            "priority": "P1 - Critical Foundation",
            "timeline": "Immediate (Weeks 1 - 4)",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": f"Establish daily attendance tracking with automated SMS/email check-ins to bridge concept continuity gaps.",
            "resource": "Student Attendance Portal & Advisor Check-In"
        })
        
    # 2. Tutoring & Remedial Program
    if current_score < 65 or fails > 0 or tutoring == "none":
        uplift = 6.5 if current_score < 50 else 4.5
        total_uplift += uplift
        interventions.append({
            "title": "Enroll in 1-on-1 Peer & Teacher Tutoring (3 hrs/wk)",
            "priority": "P1 - Remedial",
            "timeline": "Weeks 1 - 8",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Pair with top-performing math peer tutor for twice-weekly 90-minute structured problem-solving sessions.",
            "resource": "Math Lab & Peer Tutoring Center"
        })
        
    # 3. Test Prep Course Completion
    if prep == "none":
        uplift = 5.5
        total_uplift += uplift
        interventions.append({
            "title": "Complete Standardized Test Preparation Course",
            "priority": "P2 - High Impact",
            "timeline": "Weeks 2 - 6",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Complete 4-week exam prep curriculum focusing on timed mock exams, formula memorization, and question triage.",
            "resource": "Online Test Prep Portal & Practice Question Bank"
        })
        
    # 4. Weekly Study Hours Increase
    if study < 14.0:
        recommended_study = min(20.0, study + 6.0)
        uplift = round((recommended_study - study) * 0.40, 1)
        total_uplift += uplift
        interventions.append({
            "title": f"Expand Deliberate Study ({study:.0f}h -> {recommended_study:.0f}h/wk)",
            "priority": "P2 - Skill Practice",
            "timeline": "Weeks 1 - 12",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": f"Allocate +{recommended_study - study:.0f} dedicated hours weekly using the Pomodoro technique (25 min study / 5 min break).",
            "resource": "Weekly Study Timetable Planner"
        })
        
    # 5. Sleep & Wellness Optimization
    if sleep < 6.5:
        uplift = 3.5
        total_uplift += uplift
        interventions.append({
            "title": "Sleep & Cognitive Wellness Protocol (7.5+ hrs)",
            "priority": "P3 - Cognitive Wellness",
            "timeline": "Daily Habit",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Adopt consistent 11 PM bedtime routine, eliminating screen time 30 mins before sleep to maximize memory consolidation.",
            "resource": "Student Wellness & Sleep Hygiene Guide"
        })
        
    # 6. Digital Access Support
    if internet == "no":
        uplift = 3.0
        total_uplift += uplift
        interventions.append({
            "title": "School Digital Library & Hotspot Lending",
            "priority": "P3 - Resource Access",
            "timeline": "Ongoing",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Provide student with LTE hotspot lending package and after-school computer lab access.",
            "resource": "Campus IT & Library Lending Program"
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
    projected_score = min(100.0, round(current_score + min(total_uplift, 26.0), 1))
    
    def calc_grade(score):
        if score >= 90: return "A+ (Outstanding)"
        elif score >= 80: return "A (Excellent)"
        elif score >= 70: return "B (Good)"
        elif score >= 60: return "C (Satisfactory)"
        elif score >= 50: return "D (Pass)"
        else: return "F (Needs Remedial)"
        
    projected_grade = calc_grade(projected_score)
    
    # 12-Week Milestone Timeline
    milestones = [
        {
            "week": "Weeks 1 - 2: Attendance & Diagnostics",
            "target": f"Baseline Stabilize ({current_score:.1f} -> {min(100.0, current_score + total_uplift*0.25):.1f})",
            "milestone": f"Achieve 95%+ attendance, resolve prior backlogs, set weekly timetable of {total_study_hours:.0f} study hours."
        },
        {
            "week": "Weeks 3 - 6: Active Tutoring & Prep",
            "target": f"Mid-Term Checkpoint ({min(100.0, current_score + total_uplift*0.55):.1f} pts)",
            "milestone": "Attend 8 tutoring sessions, complete 50% of test prep course, maintain 7.5h daily sleep."
        },
        {
            "week": "Weeks 7 - 10: Mock Exams & Error Logs",
            "target": f"Mastery Checkpoint ({min(100.0, current_score + total_uplift*0.85):.1f} pts)",
            "milestone": "Complete 4 full-length timed mock exams with detailed error logs; score 80%+ on weekly practice quizzes."
        },
        {
            "week": "Weeks 11 - 12: Final Review Sprint",
            "target": f"Goal Attained ({projected_score:.1f} pts / {projected_grade})",
            "milestone": "Consolidate cheat-sheets, complete final benchmark test, enter examination hall with peak readiness."
        }
    ]
    
    teacher_guidance = [
        f"Monitor weekly attendance log; trigger immediate notification if attendance dips below 85%.",
        f"Provide partial credit rubrics on homework to encourage detailed written calculations.",
        f"Conduct bi-weekly 10-minute check-ins to monitor self-study consistency and reduce test anxiety."
    ]
    
    return {
        "current_score": current_score,
        "target_score": target_score,
        "projected_score": projected_score,
        "projected_grade": projected_grade,
        "total_uplift": round(projected_score - current_score, 1),
        "bottlenecks": bottlenecks,
        "interventions": interventions,
        "study_hours": study_hours,
        "total_study_hours": total_study_hours,
        "milestones": milestones,
        "teacher_guidance": teacher_guidance
    }

def generate_classroom_intervention_matrix(processed_df):
    """
    Clusters students in a classroom batch into targeted 14-feature intervention cohorts.
    """
    df = processed_df.copy()
    
    clusters = {
        "Intensive Remedial (High Risk)": [],
        "Attendance & Habit Recovery": [],
        "Test Prep & Tutoring Bootcamp": [],
        "Honors / Distinction Mentorship": []
    }
    
    for _, row in df.iterrows():
        s_id = row.get("student_id", f"STU-{row.name+1}")
        name = row.get("student_name", f"Student {row.name+1}")
        math = row.get("Predicted_Math_Score", 65.0)
        att = row.get("attendance_rate", 85.0)
        study = row.get("weekly_study_hours", 12.0)
        fails = row.get("past_failures", 0)
        risk = row.get("Risk_Tier", "Moderate Risk")
        prep = row.get("test preparation course", "none")
        
        entry = {
            "id": s_id,
            "name": name,
            "predicted_math": math,
            "attendance": att,
            "study_hours": study,
            "past_failures": fails,
            "risk": risk
        }
        
        if "High" in str(risk) or math < 50 or fails >= 2:
            clusters["Intensive Remedial (High Risk)"].append(entry)
        elif att < 80.0 or study < 8.0:
            clusters["Attendance & Habit Recovery"].append(entry)
        elif prep == "none" and math < 80.0:
            clusters["Test Prep & Tutoring Bootcamp"].append(entry)
        else:
            clusters["Honors / Distinction Mentorship"].append(entry)
            
    summary = {
        "high_risk_count": len(clusters["Intensive Remedial (High Risk)"]),
        "attendance_habit_count": len(clusters["Attendance & Habit Recovery"]),
        "test_prep_needed_count": len(clusters["Test Prep & Tutoring Bootcamp"]),
        "honors_count": len(clusters["Honors / Distinction Mentorship"]),
        "total": len(df)
    }
    
    action_plans = {
        "Intensive Remedial (High Risk)": "Assign dedicated teacher aide for 1-on-1 tutoring 3x weekly; mandatory math clinic; parent conference.",
        "Attendance & Habit Recovery": "Initiate automated attendance check-ins; pair with mentor; establish mandatory 10h/wk quiet study hall.",
        "Test Prep & Tutoring Bootcamp": "Auto-enroll in 4-week weekend exam prep cohort; weekly mock exams with error log reviews.",
        "Honors / Distinction Mentorship": "Fast-track to Advanced Placement / Math Olympiad modules; student peer mentoring leadership."
    }
    
    return {
        "clusters": clusters,
        "summary": summary,
        "action_plans": action_plans
    }
