import numpy as np
import pandas as pd

def diagnose_student_weaknesses(profile, predicted_math, pass_prob=None):
    """
    Diagnoses root academic and environmental bottlenecks based on student features and predictions.
    """
    bottlenecks = []
    reading = profile.get("reading score", 70)
    writing = profile.get("writing score", 70)
    lunch = profile.get("lunch", "standard")
    prep = profile.get("test preparation course", "none")
    edu = profile.get("parental level of education", "some college")
    
    # 1. Quantitative Deficit
    if predicted_math < 50:
        bottlenecks.append({
            "category": "Critical Mathematics Deficit",
            "severity": "High",
            "icon": "🚨",
            "detail": f"Predicted math score ({predicted_math:.1f}) is below passing threshold (50.0). Fundamental arithmetic and algebraic concepts require immediate structured remediation."
        })
    elif predicted_math < 65:
        bottlenecks.append({
            "category": "Moderate Mathematics Gap",
            "severity": "Medium",
            "icon": "⚠️",
            "detail": f"Predicted math score ({predicted_math:.1f}) indicates good foundation but difficulty with advanced problem solving and multi-step logic."
        })
        
    # 2. Reading Comprehension Lag
    if reading < 50:
        bottlenecks.append({
            "category": "Critical Reading Literacy Lag",
            "severity": "High",
            "icon": "📚",
            "detail": f"Reading score ({reading}) is severely limiting problem comprehension, word problem translation, and conceptual retention."
        })
    elif reading < 65:
        bottlenecks.append({
            "category": "Reading Fluency & Vocabulary Gap",
            "severity": "Medium",
            "icon": "📖",
            "detail": f"Reading score ({reading}) suggests a need for targeted reading comprehension drills and academic vocabulary reinforcement."
        })
        
    # 3. Writing & Synthesis Deficit
    if writing < 50:
        bottlenecks.append({
            "category": "Written Expression & Proof Deficit",
            "severity": "High",
            "icon": "✍️",
            "detail": f"Writing score ({writing}) is low, impairing structured problem write-ups, clarity of reasoning, and short-answer exam sections."
        })
    elif writing < 65:
        bottlenecks.append({
            "category": "Written Structure & Synthesis Polish",
            "severity": "Low",
            "icon": "📝",
            "detail": f"Writing score ({writing}) indicates room for refinement in structured outlining, synthesis, and written mathematical justification."
        })
        
    # 4. Preparation Gap
    if prep == "none":
        bottlenecks.append({
            "category": "Uncompleted Test Prep Course",
            "severity": "High" if predicted_math < 60 else "Medium",
            "icon": "🎯",
            "detail": "Student has not completed the standardized test preparation curriculum, missing out on ~5-8 points of empirical score boost and test-taking strategies."
        })
        
    # 5. Socioeconomic & Nutrition Support
    if lunch == "free/reduced":
        bottlenecks.append({
            "category": "Nutritional & Study Environment Support",
            "severity": "Medium",
            "icon": "🍎",
            "detail": "Free/reduced lunch indicates potential household socioeconomic headwinds; student would benefit from school breakfast programs, quiet after-school study spaces, and provided textbooks/calculators."
        })
        
    return bottlenecks

def generate_prescriptive_solution(profile, predicted_math, pass_prob=None, grade=None, target_score=None):
    """
    Generates actionable prescriptive solutions, study hour allocations, milestone timeline,
    and projected score uplift for an individual student.
    """
    reading = float(profile.get("reading score", 70))
    writing = float(profile.get("writing score", 70))
    prep = profile.get("test preparation course", "none")
    lunch = profile.get("lunch", "standard")
    
    current_score = float(predicted_math)
    if target_score is None:
        target_score = min(100.0, current_score + (15.0 if current_score < 70 else 10.0))
        
    bottlenecks = diagnose_student_weaknesses(profile, predicted_math, pass_prob)
    
    # Prescriptive Action Items & Uplift Modeling
    interventions = []
    total_uplift = 0.0
    
    # 1. Test Prep Completion
    if prep == "none":
        uplift = 6.5
        total_uplift += uplift
        interventions.append({
            "title": "Complete Standardized Test Preparation Course",
            "priority": "P1 - High Impact",
            "timeline": "Weeks 1 - 4",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Enroll in the 4-week structured exam prep course focusing on timed practice tests, formula sheets, and question triage techniques.",
            "resource": "Online Practice Portal + Mock Exam Modules"
        })
        
    # 2. Subject Tutoring
    if current_score < 60 or reading < 60 or writing < 60:
        uplift = 7.0 if current_score < 50 else 4.5
        total_uplift += uplift
        interventions.append({
            "title": "1-on-1 Peer & Teacher Tutoring (3 hrs/week)",
            "priority": "P1 - Remedial",
            "timeline": "Weeks 1 - 8",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Schedule twice-weekly 90-minute tutoring sessions covering foundational algebra, fractions, word problem modeling, and active reading.",
            "resource": "Math Lab & Peer Tutoring Center"
        })
    else:
        uplift = 3.5
        total_uplift += uplift
        interventions.append({
            "title": "Advanced Problem Solving & Olympiad Drills",
            "priority": "P2 - Enrichment",
            "timeline": "Weeks 2 - 6",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Engage in higher-order thinking problems, proof techniques, and multi-step word problem synthesis to reach distinction levels.",
            "resource": "Advanced Calculus & Analytical Thinking Guides"
        })
        
    # 3. Reading & Active Annotation Protocol
    if reading < 70:
        uplift = 4.0
        total_uplift += uplift
        interventions.append({
            "title": "Daily 25-Minute Reading & Annotation Routine",
            "priority": "P2 - Core Skill",
            "timeline": "Continuous (Daily)",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Read 2 technical or analytical passages daily using the SQ3R (Survey, Question, Read, Retrieve, Review) method to boost comprehension speed.",
            "resource": "Academic Journal Reader & Vocabulary Flashcards"
        })
        
    # 4. Nutrition & Focused Study Space
    if lunch == "free/reduced":
        uplift = 3.0
        total_uplift += uplift
        interventions.append({
            "title": "Nutrition & Quiet Study Hall Access",
            "priority": "P3 - Wellness & Environment",
            "timeline": "Ongoing",
            "est_uplift": f"+{uplift:.1f} pts",
            "action": "Connect with the school guidance counselor for breakfast club access, quiet study hall access with high-speed internet, and scientific calculator lending.",
            "resource": "Counseling & Student Support Services"
        })
        
    # Study Hours Allocation (Total ~ 10-16 hrs/week based on need)
    base_hours = 12 if current_score < 60 else (10 if current_score < 80 else 8)
    math_pct = 0.50 if current_score < reading else 0.40
    reading_pct = 0.30 if reading < writing else 0.35
    writing_pct = max(0.15, 1.0 - math_pct - reading_pct)
    
    study_hours = {
        "Mathematics Practice": round(base_hours * math_pct, 1),
        "Reading Comprehension": round(base_hours * reading_pct, 1),
        "Writing & Essay Drills": round(base_hours * writing_pct, 1),
        "Mock Exam & Review": 2.0
    }
    total_study_hours = sum(study_hours.values())
    
    # Projected Score Post-Intervention
    projected_score = min(100.0, round(current_score + min(total_uplift, 28.0), 1))
    
    def calc_grade(score):
        if score >= 90: return "A+ (Outstanding)"
        elif score >= 80: return "A (Excellent)"
        elif score >= 70: return "B (Good)"
        elif score >= 60: return "C (Satisfactory)"
        elif score >= 50: return "D (Pass)"
        else: return "F (Needs Remedial)"
        
    projected_grade = calc_grade(projected_score)
    
    # Step-by-Step 12-Week Milestone Timeline
    milestones = [
        {
            "week": "Weeks 1 - 2: Foundation & Diagnostic",
            "target": f"Baseline check ({current_score:.1f} -> {min(100.0, current_score + total_uplift*0.25):.1f})",
            "milestone": "Identify core algebra & vocabulary gaps, complete diagnostic test, set up daily study timetable."
        },
        {
            "week": "Weeks 3 - 6: Active Coursework & Prep",
            "target": f"Mid-term progress ({min(100.0, current_score + total_uplift*0.55):.1f} pts)",
            "milestone": "Complete 50% of test prep course, attend 8 tutoring sessions, finish 15 timed reading passages."
        },
        {
            "week": "Weeks 7 - 10: Mock Testing & Error Analysis",
            "target": f"Advanced mastery ({min(100.0, current_score + total_uplift*0.85):.1f} pts)",
            "milestone": "Complete 3 full-length timed mock exams with detailed error logs; score 75%+ on practice quizzes."
        },
        {
            "week": "Weeks 11 - 12: Final Sprint & Exam Readiness",
            "target": f"Projected Goal ({projected_score:.1f} pts / {projected_grade})",
            "milestone": "Consolidate formula sheets, complete final benchmark test, enter final exam with peak confidence."
        }
    ]
    
    # Counselor & Teacher Direct Guidance
    teacher_guidance = [
        f"Assign a supportive peer mentor who scored 85+ in mathematics for weekly study accountability.",
        f"Provide partial credit worksheets to encourage written intermediate steps rather than guessing.",
        f"Conduct bi-weekly 10-minute check-ins to monitor practice test consistency and reduce exam anxiety."
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
    Clusters students in a classroom batch into targeted intervention cohorts with custom solutions.
    """
    df = processed_df.copy()
    
    clusters = {
        "Intensive Remedial (High Risk)": [],
        "Test Prep Bootcamp (Moderate Gap)": [],
        "Verbal / Reading Support": [],
        "Honors / Distinction Mentorship": []
    }
    
    for _, row in df.iterrows():
        s_id = row.get("student_id", f"STU-{row.name+1}")
        name = row.get("student_name", f"Student {row.name+1}")
        math = row.get("Predicted_Math_Score", 65.0)
        reading = row.get("reading score", 65.0)
        writing = row.get("writing score", 65.0)
        prep = row.get("test preparation course", "none")
        risk = row.get("Risk_Tier", "Moderate Risk")
        
        entry = {
            "id": s_id,
            "name": name,
            "predicted_math": math,
            "reading": reading,
            "writing": writing,
            "prep": prep,
            "risk": risk
        }
        
        if "High" in risk or math < 50:
            clusters["Intensive Remedial (High Risk)"].append(entry)
        elif prep == "none" and math < 75:
            clusters["Test Prep Bootcamp (Moderate Gap)"].append(entry)
        elif reading < 65 or writing < 65:
            clusters["Verbal / Reading Support"].append(entry)
        else:
            clusters["Honors / Distinction Mentorship"].append(entry)
            
    summary = {
        "high_risk_count": len(clusters["Intensive Remedial (High Risk)"]),
        "test_prep_needed_count": len(clusters["Test Prep Bootcamp (Moderate Gap)"]),
        "verbal_support_count": len(clusters["Verbal / Reading Support"]),
        "honors_count": len(clusters["Honors / Distinction Mentorship"]),
        "total": len(df)
    }
    
    action_plans = {
        "Intensive Remedial (High Risk)": "Assign dedicated teacher aide for 1-on-1 tutoring 3x weekly; mandatory math clinic; parent conference.",
        "Test Prep Bootcamp (Moderate Gap)": "Auto-enroll in 4-week weekend SAT/Standardized prep cohort; practice mock exams weekly.",
        "Verbal / Reading Support": "Introduce daily 20-min reading comprehension drills; pairing with library reading circles.",
        "Honors / Distinction Mentorship": "Fast-track to Advanced Placement / Math Olympiad modules; peer mentoring leadership roles."
    }
    
    return {
        "clusters": clusters,
        "summary": summary,
        "action_plans": action_plans
    }
