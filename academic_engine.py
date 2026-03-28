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
# 🔥 FUZZY LOGIC SYSTEM (WITH EXPLANATION)
# -------------------------------------------------

def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def quiz_membership(quiz):
    return {
        "low": triangular(quiz, 0, 0, 50),
        "medium": triangular(quiz, 30, 50, 70),
        "high": triangular(quiz, 60, 80, 100)
    }


def confidence_membership(conf):
    mapping = {
        "low": {"low": 1, "medium": 0, "high": 0},
        "medium": {"low": 0, "medium": 1, "high": 0},
        "high": {"low": 0, "medium": 0, "high": 1}
    }
    return mapping[conf.lower()]


def anxiety_membership(anx):
    mapping = {
        "low": {"low": 1, "medium": 0, "high": 0},
        "medium": {"low": 0, "medium": 1, "high": 0},
        "high": {"low": 0, "medium": 0, "high": 1}
    }
    return mapping[anx.lower()]


def fuzzy_concept_clarity(quiz, confidence, anxiety, explain=False):

    q = quiz_membership(quiz)
    c = confidence_membership(confidence)
    a = anxiety_membership(anxiety)

    r1 = min(q["high"], c["high"], a["low"])
    r2 = min(q["medium"], c["medium"])
    r3 = max(q["low"], a["high"])

    numerator = r1 * 90 + r2 * 60 + r3 * 30
    denominator = r1 + r2 + r3

    score = numerator / denominator if denominator != 0 else 50

    if explain:
        explanation = []

        q_max = max(q, key=q.get)
        c_max = max(c, key=c.get)
        a_max = max(a, key=a.get)

        explanation.append(f"Quiz level is {q_max.upper()} ({round(q[q_max],2)})")
        explanation.append(f"Confidence is {c_max.upper()} ({round(c[c_max],2)})")
        explanation.append(f"Anxiety is {a_max.upper()} ({round(a[a_max],2)})")

        if r3 > r1 and r3 > r2:
            explanation.append("High anxiety or low quiz performance is reducing clarity.")
        elif r1 > r2:
            explanation.append("Strong quiz performance and confidence improve clarity.")
        else:
            explanation.append("Moderate performance results in balanced clarity.")

        return score, explanation

    return score

# -------------------------------------------------
# Derived Soft Variable Computation
# -------------------------------------------------

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
# 🔥 FULL STRATEGY SYSTEM (NEW)
# -------------------------------------------------

def assign_strategy(score, cluster):

    if score < 40:
        return {
            "name": "Reinforcement Strategy",
            "definition": "Student lacks foundational understanding and consistency.",
            "goal": "Build strong academic fundamentals.",
            "actions": [
                "Revise core concepts daily",
                "Solve minimum 20 practice problems per week",
                "Attend doubt-solving sessions",
                "Focus on understanding over memorization"
            ],
            "weekly_plan": [
                "Day 1-3: Concept learning",
                "Day 4-5: Practice problems",
                "Day 6: Revision",
                "Day 7: Self-test"
            ]
        }

    elif score < 70:
        return {
            "name": "Structured Growth Strategy",
            "definition": "Student has moderate performance but lacks consistency.",
            "goal": "Improve stability and efficiency in learning.",
            "actions": [
                "Follow fixed study schedule",
                "Practice active recall",
                "Take 2 mock tests per month",
                "Track weekly performance"
            ],
            "weekly_plan": [
                "5 days focused study",
                "1 day mock test",
                "1 day revision"
            ]
        }

    else:
        return {
            "name": "Advanced Enrichment Strategy",
            "definition": "Student is performing well and ready for advanced challenges.",
            "goal": "Maximize academic excellence.",
            "actions": [
                "Solve advanced-level problems",
                "Work on projects/research",
                "Participate in competitions",
                "Mentor peers"
            ],
            "weekly_plan": [
                "Advanced problem solving",
                "Project work",
                "Peer teaching sessions"
            ]
        }

# -------------------------------------------------
# Core Prediction
# -------------------------------------------------

def predict_effectiveness(input_data):
    df = pd.DataFrame([input_data], columns=feature_columns)
    scaled = scaler.transform(df)
    return ann_model.predict(scaled)[0]

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

    concept, concept_explanation = fuzzy_concept_clarity(
        quiz_avg, confidence, anxiety, explain=True
    )

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

    strategy = assign_strategy(effectiveness, cluster)

    simulation = simulate_improvement(ann_input)
    projected_strategy = assign_strategy(simulation["Projected"], cluster)

    risk_score = 80 if cluster == 2 else 60 if cluster == 0 else 40 if cluster == 3 else 20

    summary = f"""
    The student demonstrates a predicted academic effectiveness of {round(effectiveness,2)}%.
    Cognitive load is estimated at {round(cognitive_load,2)}%.
    Recommended approach: {strategy['name']}.
    """

    return {
        "Effectiveness": round(effectiveness, 2),
        "Cluster": int(cluster),
        "Strategy": strategy["name"],
        "Strategy_Details": strategy,
        "Simulation": simulation,
        "Projected_Strategy": projected_strategy["name"],
        "Risk_Score": risk_score,
        "AI_Summary": summary.strip(),
        "Feature_Data": ann_input,
        "Concept_Explanation": concept_explanation
    }
