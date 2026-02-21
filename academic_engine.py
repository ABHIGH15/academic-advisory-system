import joblib
import numpy as np
import pandas as pd

# Load models
ann_model = joblib.load("models/ann_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cluster_model = joblib.load("models/cluster_model.pkl")

feature_columns = [
    "Attendance_Rate",
    "Concept_Clarity",
    "Practice_Frequency",
    "Internal_Assessment",
    "Assignment_Quality",
    "Time_Management",
    "Learning_Consistency"
]

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

def analyze_student(input_data):
    effectiveness = predict_effectiveness(input_data)
    
    cluster = cluster_model.predict(
        scaler.transform(pd.DataFrame([input_data], columns=feature_columns))
    )[0]
    
    strategy = assign_strategy(effectiveness)
    
    return {
        "Effectiveness": round(effectiveness, 2),
        "Cluster": int(cluster),
        "Strategy": strategy
    }
