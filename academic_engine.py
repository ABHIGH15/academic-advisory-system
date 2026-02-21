import joblib
import numpy as np
import pandas as pd

# Load models
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

# ------------------------------
# Core Prediction
# ------------------------------

def predict_effectiveness(input_data):
    df = pd.DataFrame([input_data], columns=feature_columns)
    scaled = scaler.transform(df)
    return ann_model.predict(scaled)[0]

def assign_strategy(score):
    if score < 40:
        return "Reinforcement Strategy"
    elif score < 70:
        return "Balanced Improvement Strategy"
    else:
        return "Advanced Enrichment Strategy"

# ------------------------------
# Improvement Simulation
# ------------------------------

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

# ------------------------------
# Main Analysis Function
# ------------------------------

def analyze_student(input_data):

    effectiveness = predict_effectiveness(input_data)

    cluster = cluster_model.predict(
        scaler.transform(pd.DataFrame([input_data], columns=feature_columns))
    )[0]

    strategy = assign_strategy(effectiveness)

    simulation = simulate_improvement(input_data)

    projected_strategy = assign_strategy(simulation["Projected"])

    risk_score = 0

    if cluster == 2:
        risk_score = 80
    elif cluster == 0:
        risk_score = 60
    elif cluster == 3:
        risk_score = 40
    else:
        risk_score = 20

    return {
        "Effectiveness": round(effectiveness, 2),
        "Cluster": int(cluster),
        "Strategy": strategy,
        "Simulation": simulation,
        "Projected_Strategy": projected_strategy,
        "Risk_Score": risk_score,
        "Top_Issues": [
            {"name": "Concept Clarity", "severity": 72},
            {"name": "Time Management", "severity": 65},
            {"name": "Practice Frequency", "severity": 58}
        ],
        "AI_Summary": "The student demonstrates moderate conceptual understanding but inconsistent application. Improving structured practice and time management can significantly elevate academic performance.",
        "Feature_Data": input_data
    }
