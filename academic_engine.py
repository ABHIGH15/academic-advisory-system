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

# ---------------- UTILITY ----------------

def clamp(v):
    return max(0, min(100, v))

# ---------------- INTERPRETATION ----------------

def interpret_cluster(c):
    if c == 2:
        return "High-risk academic profile: inconsistent learning and high cognitive stress"
    elif c == 0:
        return "Moderate profile: improvement possible with structured planning"
    elif c == 3:
        return "Developing profile: needs conceptual strengthening"
    else:
        return "Strong academic profile with stable performance"

def interpret_risk(score):
    if score > 70:
        return "High academic risk due to overload and weak learning habits"
    elif score > 40:
        return "Moderate risk — improvement needed in key areas"
    else:
        return "Low academic risk — performance is stable"

def interpret_cognitive_load(load):
    if load > 70:
        return "High cognitive load → burnout risk"
    elif load > 40:
        return "Moderate load → manageable but needs optimization"
    else:
        return "Low cognitive load → healthy learning state"

# ---------------- FUZZY ----------------

def fuzzy_concept_clarity(q, conf, anx):

    conf_map = {"low": 30, "medium": 60, "high": 90}
    anx_map = {"low": 20, "medium": 50, "high": 80}

    conf_score = conf_map.get(str(conf).lower(), 60)
    anx_score = anx_map.get(str(anx).lower(), 50)

    score = clamp(0.6*q + 0.25*conf_score - 0.15*anx_score)

    explanation = []

    if q < 50:
        explanation.append("Weak conceptual base due to low quiz performance")
    elif q < 70:
        explanation.append("Partial understanding — needs reinforcement")
    else:
        explanation.append("Strong conceptual understanding")

    if conf == "low":
        explanation.append("Low confidence reducing performance efficiency")

    if anx == "high":
        explanation.append("High anxiety negatively impacting performance")

    return score, explanation

# ---------------- DERIVED ----------------

def compute_learning_consistency(sd, sh, sl):
    return clamp(0.5*(sd/7*100) + 0.3*(min(sh/5,1)*100) + 0.2*(min(sl/8,1)*100))

def compute_time_management(sh, st, d):
    return clamp(0.6*(min(sh/5,1)*100) - 0.25*(min(st/8,1)*100) - 0.15*(min(d/7,1)*100))

def compute_cognitive_load(sh, st, sl):
    return clamp(0.5*(min(st/8,1)*100) + 0.3*(max(0,8-sl)/8*100) + 0.2*(max(0,sh-6)/6*100))

# ---------------- MODEL ----------------

def predict_effectiveness(x):
    df = pd.DataFrame([x], columns=feature_columns)
    return ann_model.predict(scaler.transform(df))[0]

# ---------------- STRATEGY ----------------

def assign_strategy(score):
    if score < 40:
        return "Foundation Building Strategy"
    elif score < 70:
        return "Consistency Growth Strategy"
    else:
        return "Advanced Optimization Strategy"

# ---------------- ADVANCED RULE ENGINE ----------------

def generate_advanced_insights(sh, concept, st, sl, p):

    insights = []

    if sh > 5 and concept < 50:
        insights.append("High study time but low learning → inefficient study method")

    if st > 6 and sl < 6:
        insights.append("High screen time + low sleep → cognitive overload")

    if p > 30 and concept < 60:
        insights.append("Excessive practice without understanding → revise concepts")

    if sh < 2 and p < 10:
        insights.append("Low engagement → inconsistent learning behavior")

    return insights

# ---------------- TECHNIQUE ENGINE ----------------

def generate_techniques(sh, concept, st, sl, p):

    techniques = []

    if concept < 60:
        techniques.append("Active Recall → test yourself instead of rereading")
        techniques.append("Spaced Repetition → revise at intervals")

    if p < 15:
        techniques.append("Pomodoro Technique (25/5 focus cycles)")

    if sh < 3:
        techniques.append("Eat That Frog → start with hardest task")
        techniques.append("1-2-5 Rule → structured daily planning")

    if st > 5:
        techniques.append("Digital Detox during study sessions")
        techniques.append("Seinfeld Strategy → maintain consistency streaks")

    techniques.append("Pareto Principle → focus on high-impact topics")

    return techniques

# ---------------- WELLNESS ----------------

def generate_wellness_tips(sl, st, anx):

    tips = []

    if sl < 6:
        tips.append("4-7-8 breathing for sleep improvement")
        tips.append("Military sleep method for faster sleep")
        tips.append("10-3-2-1-0 sleep rule")

    if anx == "high":
        tips.append("Box breathing for anxiety control")
        tips.append("5-4-3-2-1 grounding technique")
        tips.append("Progressive Muscle Relaxation (PMR)")

    if st > 6:
        tips.append("20-minute rule to reduce screen addiction")

    return tips

# ---------------- PRIORITY ----------------

def get_priority_areas(data):
    sorted_features = sorted(data.items(), key=lambda x: x[1])[:3]
    return [{"feature": k, "value": round(v,1)} for k,v in sorted_features]

# ---------------- OPTIMIZATION ----------------

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

        current[best_feature] = min(100, current[best_feature] + 5)

    return [
        {"feature": k, "from": round(data[k],1), "to": round(current[k],1)}
        for k in data if current[k] > data[k]
    ]

# ---------------- TARGET ----------------

def generate_target_plan(data, target=80):

    current = data.copy()

    for _ in range(15):
        score = predict_effectiveness(current)

        if score >= target:
            break

        best_gain = 0
        best_feature = None

        for k in current:
            temp = current.copy()
            temp[k] = min(100, temp[k] + 5)

            gain = predict_effectiveness(temp) - score

            if gain > best_gain:
                best_gain = gain
                best_feature = k

        if not best_feature:
            break

        current[best_feature] = min(100, current[best_feature] + 5)

    return [
        {"feature": k, "target": round(current[k],1)}
        for k in data if current[k] > data[k]
    ]

# ---------------- MAIN ----------------

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
        "Practice_Frequency": min(p/30,1)*100,
        "Internal_Assessment": internal,
        "Assignment_Quality": 100 - min(d/7,1)*100,
        "Time_Management": tm,
        "Learning_Consistency": consistency
    }

    eff = predict_effectiveness(ann_input)

    cluster = int(
        cluster_model.predict(
            scaler.transform(pd.DataFrame([ann_input], columns=feature_columns))
        )[0]
    )

    risk = 80 if cluster==2 else 60 if cluster==0 else 40 if cluster==3 else 20

    return {
        "Effectiveness": round(eff,2),
        "Strategy": assign_strategy(eff),
        "Student_Level": "Beginner" if eff < 40 else "Intermediate" if eff < 70 else "Advanced",

        "Cluster_Insight": interpret_cluster(cluster),
        "Risk_Score": risk,
        "Risk_Reason": interpret_risk(risk),

        "Cognitive_Load": round(load,2),
        "Cognitive_Insight": interpret_cognitive_load(load),

        "Concept_Explanation": explanation,

        "Priority_Areas": get_priority_areas(ann_input),
        "Target_Plan": generate_target_plan(ann_input),
        "Optimization_Insights": generate_optimization_plan(ann_input),

        "Advanced_Insights": generate_advanced_insights(sh, concept, st, sl, p),
        "Techniques": generate_techniques(sh, concept, st, sl, p),
        "Wellness_Tips": generate_wellness_tips(sl, st, anx),

        "Feature_Data": ann_input
    }
