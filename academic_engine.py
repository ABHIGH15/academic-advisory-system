import joblib
import numpy as np
import pandas as pd
import os

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

def clamp(v):
    return max(0, min(100, v))

# -------------------------------------------------
# FUZZY (FIXED + SAFE)
# -------------------------------------------------

def fuzzy_concept_clarity(q, conf, anx):

    # 🔥 SAFE handling (FIXES 500 ERROR)
    conf = str(conf).lower().strip()
    anx = str(anx).lower().strip()

    conf_map = {"low": 30, "medium": 60, "high": 90}
    anx_map = {"low": 20, "medium": 50, "high": 80}

    conf_score = conf_map.get(conf, 60)
    anx_score = anx_map.get(anx, 50)

    score = 0.6*q + 0.25*conf_score - 0.15*anx_score
    score = clamp(score)

    # 🧠 HUMAN EXPLANATION
    explanation = []

    if q < 50:
        explanation.append("Your quiz performance is low → weak conceptual base")
    elif q < 70:
        explanation.append("Your quiz performance is moderate → partial understanding")
    else:
        explanation.append("Your quiz performance is strong")

    if conf == "low":
        explanation.append("Low confidence is reducing your performance")
    elif conf == "medium":
        explanation.append("Confidence is moderate but can be improved")
    else:
        explanation.append("High confidence is helping your learning")

    if anx == "high":
        explanation.append("High anxiety is affecting your performance under pressure")

    return score, explanation

# -------------------------------------------------
# DERIVED FEATURES
# -------------------------------------------------

def compute_learning_consistency(sd, sh, sl):
    return clamp(0.5*(sd/7*100) + 0.3*(min(sh/5,1)*100) + 0.2*(min(sl/8,1)*100))

def compute_time_management(sh, st, d):
    return clamp(0.6*(min(sh/5,1)*100) - 0.25*(min(st/8,1)*100) - 0.15*(min(d/7,1)*100))

def compute_cognitive_load(sh, st, sl):
    return clamp(0.5*(min(st/8,1)*100) + 0.3*(max(0,8-sl)/8*100) + 0.2*(max(0,sh-6)/6*100))

# -------------------------------------------------
# STRATEGY
# -------------------------------------------------

def assign_strategy(score):
    if score < 40:
        return "Foundation Building Strategy"
    elif score < 70:
        return "Consistency Growth Strategy"
    else:
        return "Advanced Optimization Strategy"

# -------------------------------------------------
# CORE MODEL (FIXED)
# -------------------------------------------------

def predict_effectiveness(x):
    df = pd.DataFrame([x], columns=feature_columns)
    scaled = scaler.transform(df)
    return ann_model.predict(scaled)[0]

# -------------------------------------------------
# OPTIMIZATION (FIXED OUTPUT)
# -------------------------------------------------

def generate_optimization_plan(data):

    current = data.copy()

    for _ in range(10):
        base = predict_effectiveness(current)

        best_gain = 0
        best_feature = None

        for k in current:
            temp = current.copy()
            temp[k] = min(100, temp[k] + 5)

            gain = predict_effectiveness(temp) - base

            if gain > best_gain:
                best_gain = gain
                best_feature = k

        if not best_feature:
            break

        current[best_feature] += 5

    # 🔥 CLEAN + NON-REPETITIVE OUTPUT
    improvements = []
    for k in data:
        if current[k] > data[k]:
            improvements.append({
                "feature": k,
                "from": round(data[k], 1),
                "to": round(current[k], 1),
                "change": round(current[k] - data[k], 1)
            })

    return improvements

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------

def interpret_cluster(c):
    if c == 2:
        return "You belong to a high-risk group with low consistency and high academic pressure."
    elif c == 0:
        return "You are in a moderate group but need better planning and consistency."
    elif c == 3:
        return "You are improving but still have conceptual gaps."
    else:
        return "You are a high-performing student with strong academic behavior."

def interpret_risk(score):
    if score > 70:
        return "High risk due to poor consistency and high cognitive load."
    elif score > 40:
        return "Moderate risk — improvement needed in key areas."
    return "Low risk — stable and well-managed performance."

def generate_tips(sh, sl, st, p):

    tips = []

    if sl < 6:
        tips.append("Improve sleep: avoid screens 30 minutes before bed and maintain fixed sleep timing")

    if sh < 3:
        tips.append("Increase study hours using Pomodoro technique (25 min focused sessions)")

    if st > 6:
        tips.append("Reduce screen time — excessive usage reduces focus and retention")

    if p < 15:
        tips.append("Practice daily: solve at least 10 problems using active recall technique")

    return tips

# -------------------------------------------------
# MAIN FUNCTION (FIXED CLUSTER BUG)
# -------------------------------------------------

def analyze_student(f):

    attendance = float(f["attendance"])
    quiz = float(f["quiz_avg"])
    internal = float(f["internal"])
    sh = float(f["study_hours"])
    sd = float(f["study_days"])
    p = float(f["practice_per_week"])
    d = float(f["assignment_delay"])
    sl = float(f["sleep_hours"])
    st = float(f["screen_time"])
    conf = f["confidence"]
    anx = f["anxiety"]

    concept, explanation = fuzzy_concept_clarity(quiz, conf, anx)

    consistency = compute_learning_consistency(sd, sh, sl)
    tm = compute_time_management(sh, st, d)
    load = compute_cognitive_load(sh, st, sl)

    ann_input = {
        "Attendance_Rate": attendance,
        "Concept_Clarity": concept,
        "Practice_Frequency": min(p/30, 1)*100,
        "Internal_Assessment": internal,
        "Assignment_Quality": 100 - min(d/7,1)*100,
        "Time_Management": tm,
        "Learning_Consistency": consistency
    }

    eff = predict_effectiveness(ann_input)

    # 🔥 FIXED CLUSTER ERROR
    cluster = int(
        cluster_model.predict(
            scaler.transform(pd.DataFrame([ann_input], columns=feature_columns))
        )[0]
    )

    risk = 80 if cluster==2 else 60 if cluster==0 else 40 if cluster==3 else 20

    return {
        "Effectiveness": round(eff, 2),
        "Strategy": assign_strategy(eff),
        "Cluster_Insight": interpret_cluster(cluster),
        "Risk_Score": risk,
        "Risk_Reason": interpret_risk(risk),
        "Concept_Explanation": explanation,
        "Optimization_Insights": generate_optimization_plan(ann_input),
        "Smart_Tips": generate_tips(sh, sl, st, p),
        "Feature_Data": ann_input
    }
