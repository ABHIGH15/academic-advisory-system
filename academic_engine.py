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

def clamp(value):
    return max(0, min(100, value))

# ---------------- FUZZY ----------------

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

        return score, explanation

    return score

# ---------------- DERIVED ----------------

def compute_learning_consistency(sd, sh, sl):
    return clamp(0.5*(sd/7*100) + 0.3*(min(sh/5,1)*100) + 0.2*(min(sl/8,1)*100))


def compute_time_management(sh, st, d):
    return clamp(0.6*(min(sh/5,1)*100) - 0.25*(min(st/8,1)*100) - 0.15*(min(d/7,1)*100))


def compute_cognitive_load(sh, st, sl):
    return clamp(0.5*(min(st/8,1)*100) + 0.3*(max(0,8-sl)/8*100) + 0.2*(max(0,sh-6)/6*100))


def compute_motivation(att, p, conf):
    return clamp(0.5*att + 0.3*(min(p/30,1)*100) + 0.2*(60 if conf=="medium" else 90 if conf=="high" else 30))

# ---------------- STRATEGY ----------------

def assign_strategy(score, cluster):

    if score < 40:
        return {
            "name": "Reinforcement Strategy",
            "definition": "Weak foundation",
            "goal": "Build basics",
            "actions": ["Revise basics", "Practice daily"],
            "weekly_plan": ["Concept", "Practice", "Test"]
        }

    elif score < 70:
        return {
            "name": "Structured Growth Strategy",
            "definition": "Moderate performance",
            "goal": "Improve consistency",
            "actions": ["Scheduled study", "Mock tests"],
            "weekly_plan": ["Study", "Test", "Revise"]
        }

    else:
        return {
            "name": "Advanced Strategy",
            "definition": "High performer",
            "goal": "Maximize output",
            "actions": ["Advanced problems", "Projects"],
            "weekly_plan": ["Deep work", "Research"]
        }

# ---------------- CORE ----------------

def predict_effectiveness(input_data):
    df = pd.DataFrame([input_data], columns=feature_columns)
    return ann_model.predict(scaler.transform(df))[0]

# ---------------- SIMULATION ----------------

def simulate_improvement(input_data):
    original = predict_effectiveness(input_data)
    improved = {k:min(100,v+10) for k,v in input_data.items()}
    projected = predict_effectiveness(improved)

    return {
        "Original": round(original,2),
        "Projected": round(projected,2),
        "Improvement_Percent": round(((projected-original)/original)*100 if original else 0,2)
    }

# ---------------- OPTIMIZATION ----------------

def generate_optimization_plan(data, target=80):

    plan = []
    current = data.copy()

    for _ in range(10):
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

        plan.append({
            "feature": best_feature,
            "value": current[best_feature]
        })

    return plan

# ---------------- MAIN ----------------

def analyze_student(form_input):

    attendance = float(form_input["attendance"])
    quiz = float(form_input["quiz_avg"])
    internal = float(form_input["internal"])
    sh = float(form_input["study_hours"])
    sd = float(form_input["study_days"])
    p = float(form_input["practice_per_week"])
    d = float(form_input["assignment_delay"])
    sl = float(form_input["sleep_hours"])
    st = float(form_input["screen_time"])
    conf = form_input["confidence"]
    anx = form_input["anxiety"]

    concept, explanation = fuzzy_concept_clarity(quiz, conf, anx, True)

    consistency = compute_learning_consistency(sd, sh, sl)
    tm = compute_time_management(sh, st, d)

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
    cluster = cluster_model.predict(scaler.transform(pd.DataFrame([ann_input])))[0]

    strategy = assign_strategy(eff, cluster)
    sim = simulate_improvement(ann_input)
    opt = generate_optimization_plan(ann_input)

    return {
        "Effectiveness": round(eff,2),
        "Cluster": int(cluster),
        "Strategy": strategy["name"],
        "Strategy_Details": strategy,
        "Simulation": sim,
        "Risk_Score": 80 if cluster==2 else 60 if cluster==0 else 40 if cluster==3 else 20,
        "Feature_Data": ann_input,
        "Concept_Explanation": explanation,
        "Optimization_Plan": opt
    }
