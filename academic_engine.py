import joblib
import numpy as np
import pandas as pd
import os

# -------------------------------------------------
# Load Models
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ann_model = joblib.load(os.path.join(BASE_DIR, "models/ann_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
cluster_model = joblib.load(os.path.join(BASE_DIR, "models/cluster_model.pkl"))

feature_columns = [
    "Attendance_Rate",
    "Concept_Clarity",
    "Practice_Frequency",
    "Internal_Assessment",
    "Assignment_Quality",
    "Time_Management",
    "Learning_Consistency"
]

# -------------------------------------------------
# Utility
# -------------------------------------------------

def clamp(value):
    return max(0, min(100, value))

# -------------------------------------------------
# Derived Soft Variable Computation
# -------------------------------------------------

def compute_concept_clarity(quiz_avg, confidence, anxiety):
    confidence_map = {"low": 30, "medium": 60, "high": 90}
    anxiety_map = {"low": 20, "medium": 50, "high": 80}

    confidence_score = confidence_map.get(confidence.lower(), 50)
    anxiety_score = anxiety_map.get(anxiety.lower(), 50)

    clarity = (
        0.6 * quiz_avg +
        0.25 * confidence_score -
        0.15 * anxiety_score
    )

    return clamp(clarity)


def compute_learning_consistency(study_days, study_hours, sleep_hours):
    study_days_score = (study_days / 7) * 100
    study_hours_score = min(study_hours / 5, 1) * 100
    sleep_score = min(sleep_hours / 8, 1) * 100

    consistency = (
        0.5 * study_days_score +
        0.3 * study_hours_score +
        0.2 * sleep_score
    )

    return clamp(consistency)


def compute_time_management(study_hours, screen_time, delay_days):
    study_score = min(study_hours / 5, 1) * 100
    screen_penalty = min(screen_time / 8, 1) * 100
    delay_penalty = min(delay_days / 7, 1) * 100

    tm = (
        0.6 * study_score -
        0.25 * screen_penalty -
        0.15 * delay_penalty
    )

    return clamp(tm)


def compute_cognitive_load(study_hours, screen_time, sleep_hours):
    sleep_deficit = max(0, 8 - sleep_hours) / 8 * 100
    screen_load = min(screen_time / 8, 1) * 100
    overstudy = max(0, study_hours - 6) / 6 * 100

    load = (
        0.5 * screen_load +
        0.3 * sleep_deficit +
        0.2 * overstudy
    )

    return clamp(load)


def compute_motivation(attendance, practice_per_week, confidence):
    confidence_map = {"low": 30, "medium": 60, "high": 90}
    confidence_score = confidence_map.get(confidence.lower(), 50)

    practice_score = min(practice_per_week / 30, 1) * 100

    motivation = (
        0.5 * attendance +
        0.3 * practice_score +
        0.2 * confidence_score
    )

    return clamp(motivation)

# -------------------------------------------------
# Personalized Intervention Engine
# -------------------------------------------------

def generate_action_plan(concept, consistency, time_mgmt, cognitive_load, motivation):

    actions = []

    if concept < 50:
        actions.append({
            "title": "Improve Concept Clarity",
            "suggestions": [
                "Use Retrieval Practice: Solve 5 questions daily without notes.",
                "Apply Spaced Repetition (1-3-7 day cycle).",
                "Use Feynman Technique to teach concepts aloud."
            ]
        })

    if consistency < 60:
        actions.append({
            "title": "Enhance Learning Consistency",
            "suggestions": [
                "Fix a daily study time.",
                "Follow 90-minute focused sessions.",
                "Create weekly structured revision plan."
            ]
        })

    if time_mgmt < 60:
        actions.append({
            "title": "Optimize Time Management",
            "suggestions": [
                "Apply 2-hour Deep Work rule.",
                "Use time blocking in calendar.",
                "Limit screen time after 9PM."
            ]
        })

    if cognitive_load > 60:
        actions.append({
            "title": "Reduce Cognitive Load",
            "suggestions": [
                "Sleep minimum 7–8 hours daily.",
                "Practice 4-7-8 breathing technique.",
                "Use 20-20-20 rule for eye relaxation."
            ]
        })

    if motivation < 60:
        actions.append({
            "title": "Boost Academic Motivation",
            "suggestions": [
                "Set weekly performance targets.",
                "Track daily study score.",
                "Reward yourself after milestones."
            ]
        })

    return actions

# -------------------------------------------------
# Core Prediction
# -------------------------------------------------

def predict_effectiveness(input_data):
    df = pd.DataFrame([input_data], columns=feature_columns)
    scaled = scaler.transform(df)
    return ann_model.predict(scaled)[0]


def assign_strategy(score):
    if score < 40:
        return "Reinforcement Strategy"
    elif score < 70:
        return "Structured Growth Strategy"
    else:
        return "Advanced Enrichment Strategy"

# -------------------------------------------------
# Improvement Simulation
# -------------------------------------------------

def simulate_improvement(input_data):
    original_score = predict_effectiveness(input_data)
    improved = input_data.copy()

    for key in improved:
        improved[key] = min(100, improved[key] + 10)

    projected_score = predict_effectiveness(improved)

    improvement = projected_score - original_score
    improvement_percent = (improvement / original_score) * 100 if original_score != 0 else 0

    return {
        "Original": round(original_score, 2),
        "Projected": round(projected_score, 2),
        "Improvement": round(improvement, 2),
        "Improvement_Percent": round(improvement_percent, 2)
    }

# -------------------------------------------------
# Main Analysis Function
# -------------------------------------------------

def analyze_student(form_input):

    attendance = float(form_input["attendance"])
    quiz_avg = float(form_input["quiz_avg"])
    internal = float(form_input["internal"])
    study_hours = float(form_input["study_hours"])
    study_days = float(form_input["study_days"])
    practice_per_week = float(form_input["practice_per_week"])
    delay_days = float(form_input["assignment_delay"])
    sleep_hours = float(form_input["sleep_hours"])
    screen_time = float(form_input["screen_time"])
    confidence = form_input["confidence"]
    anxiety = form_input["anxiety"]

    concept = compute_concept_clarity(quiz_avg, confidence, anxiety)
    consistency = compute_learning_consistency(study_days, study_hours, sleep_hours)
    time_mgmt = compute_time_management(study_hours, screen_time, delay_days)
    cognitive_load = compute_cognitive_load(study_hours, screen_time, sleep_hours)
    motivation = compute_motivation(attendance, practice_per_week, confidence)

    practice_score = min(practice_per_week / 30, 1) * 100
    assignment_quality = 100 - min(delay_days / 7, 1) * 100

    ann_input = {
        "Attendance_Rate": attendance,
        "Concept_Clarity": concept,
        "Practice_Frequency": practice_score,
        "Internal_Assessment": internal,
        "Assignment_Quality": assignment_quality,
        "Time_Management": time_mgmt,
        "Learning_Consistency": consistency
    }

    effectiveness = predict_effectiveness(ann_input)

    cluster = cluster_model.predict(
        scaler.transform(pd.DataFrame([ann_input], columns=feature_columns))
    )[0]

    strategy = assign_strategy(effectiveness)
    simulation = simulate_improvement(ann_input)
    projected_strategy = assign_strategy(simulation["Projected"])

    risk_score = 80 if cluster == 2 else 60 if cluster == 0 else 40 if cluster == 3 else 20

    feature_values = {
        "Concept Clarity": concept,
        "Learning Consistency": consistency,
        "Time Management": time_mgmt,
        "Practice Frequency": practice_score,
        "Motivation Index": motivation
    }

    sorted_weak = sorted(feature_values.items(), key=lambda x: x[1])[:3]

    top_issues = [
        {"name": name, "severity": round(100 - value, 2)}
        for name, value in sorted_weak
    ]

    action_plan = generate_action_plan(
        concept,
        consistency,
        time_mgmt,
        cognitive_load,
        motivation
    )

    summary = f"""
    The student demonstrates a predicted academic effectiveness of {round(effectiveness,2)}%.
    Primary improvement areas include {', '.join([issue['name'] for issue in top_issues])}.
    Cognitive load is estimated at {round(cognitive_load,2)}%.
    Recommended approach: {strategy}.
    """

    return {
        "Effectiveness": round(effectiveness, 2),
        "Cluster": int(cluster),
        "Strategy": strategy,
        "Simulation": simulation,
        "Projected_Strategy": projected_strategy,
        "Risk_Score": risk_score,
        "Top_Issues": top_issues,
        "AI_Summary": summary.strip(),
        "Feature_Data": ann_input,
        "Action_Plan": action_plan
    }
