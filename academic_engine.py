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

def explain_feature(feature, value):

    if feature == "Concept_Clarity":
        if value < 50:
            return "You struggle to clearly understand concepts"
        elif value < 70:
            return "Your understanding is moderate but needs improvement"
        else:
            return "You have strong conceptual clarity"

    if feature == "Time_Management":
        if value < 50:
            return "Poor time management is limiting your productivity"
        elif value < 70:
            return "You manage time moderately well"
        else:
            return "You manage your time efficiently"

    if feature == "Assignment_Quality":
        return "Assignment quality reflects your discipline and consistency"

    return "Needs improvement"

# ---------------- ACTION PLANS ----------------

def generate_action_plan(feature):

    plans = {
        "Concept_Clarity": [
            "Use Active Recall (test yourself instead of rereading)",
            "Revise concepts within 24 hours",
            "Solve 10–15 problems daily"
        ],
        "Time_Management": [
            "Use Pomodoro Technique (25 min focus)",
            "Reduce phone usage while studying",
            "Plan your day using 1-2-5 rule"
        ],
        "Assignment_Quality": [
            "Break assignments into smaller tasks",
            "Start early to avoid last-day pressure",
            "Follow Parkinson’s Law to control time usage"
        ]
    }

    return plans.get(feature, ["Work consistently on this area"])

# ---------------- FUZZY ----------------

def fuzzy_concept_clarity(q, conf, anx):

    conf_map = {"low": 30, "medium": 60, "high": 90}
    anx_map = {"low": 20, "medium": 50, "high": 80}

    conf_score = conf_map.get(str(conf).lower(), 60)
    anx_score = anx_map.get(str(anx).lower(), 50)

    score = clamp(0.6*q + 0.25*conf_score - 0.15*anx_score)

    explanation = []

    if q < 50:
        explanation.append("Low quiz score → weak fundamentals")
    elif q < 70:
        explanation.append("Average quiz score → partial understanding")
    else:
        explanation.append("Strong quiz performance")

    if conf == "low":
        explanation.append("Low confidence is reducing performance")

    if anx == "high":
        explanation.append("High anxiety affects performance")

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
        return "Focus on building fundamentals and consistency"
    elif score < 70:
        return "Improve consistency and optimize study habits"
    else:
        return "Optimize performance and maximize efficiency"

# ---------------- PRIORITY ----------------

def get_priority_areas(data):

    sorted_features = sorted(data.items(), key=lambda x: x[1])[:3]

    output = []

    for f, v in sorted_features:
        output.append({
            "feature": f.replace("_", " "),
            "value": round(v,1),
            "meaning": explain_feature(f, v),
            "actions": generate_action_plan(f)
        })

    return output

# ---------------- TARGET PLAN ----------------

def generate_target_plan(data):

    priority = get_priority_areas(data)

    return [
        {
            "feature": item["feature"],
            "current": item["value"],
            "target": min(100, item["value"] + 20),
            "meaning": item["meaning"],
            "actions": item["actions"]
        }
        for item in priority[:2]
    ]

# ---------------- ADVANCED INSIGHTS ----------------

def generate_advanced_insights(sh, concept, st, sl, p):

    insights = []

    if sh > 5 and concept < 50:
        insights.append("You are studying a lot but not learning effectively")

    if st > 6 and sl < 6:
        insights.append("High screen time + low sleep → brain fatigue")

    if p > 30 and concept < 60:
        insights.append("Practice is high but understanding is low → wrong method")

    return insights

# ---------------- TECHNIQUES ----------------

def generate_techniques(sh, concept, st, sl, p):

    techniques = []

    if concept < 60:
        techniques.append("Active Recall + Spaced Repetition")

    if sh < 3:
        techniques.append("Eat That Frog + 1-2-5 planning")

    if st > 5:
        techniques.append("Digital Detox + Seinfeld consistency method")

    techniques.append("Pareto Principle (focus on important topics)")

    return techniques

# ---------------- WELLNESS ----------------

def generate_wellness(sl, st, anx):

    tips = []

    if sl < 6:
        tips.append("Use 4-7-8 breathing for sleep")
        tips.append("Follow 10-3-2-1-0 sleep rule")

    if anx == "high":
        tips.append("Use Box Breathing")
        tips.append("Use 5-4-3-2-1 grounding")

    if st > 6:
        tips.append("Apply 20-minute rule for screen control")

    return tips

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

    cluster = int(cluster_model.predict(
        scaler.transform(pd.DataFrame([ann_input], columns=feature_columns))
    )[0])

    risk = 80 if cluster==2 else 60 if cluster==0 else 40 if cluster==3 else 20

    return {
        "Effectiveness": round(eff,2),
        "Strategy": assign_strategy(eff),
        "Student_Level": "Beginner" if eff < 40 else "Intermediate" if eff < 70 else "Advanced",

        "Cognitive_Load": round(load,2),
        "Concept_Explanation": explanation,

        "Priority_Areas": get_priority_areas(ann_input),
        "Target_Plan": generate_target_plan(ann_input),

        "Advanced_Insights": generate_advanced_insights(sh, concept, st, sl, p),
        "Techniques": generate_techniques(sh, concept, st, sl, p),
        "Wellness_Tips": generate_wellness(sl, st, anx),

        "Feature_Data": ann_input
    }
