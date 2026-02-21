from flask import Flask, render_template, request
from academic_engine import analyze_student

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    input_data = {
        "Attendance_Rate": float(request.form["attendance"]),
        "Concept_Clarity": float(request.form["concept"]),
        "Practice_Frequency": float(request.form["practice"]),
        "Internal_Assessment": float(request.form["internal"]),
        "Assignment_Quality": float(request.form["assignment"]),
        "Time_Management": float(request.form["time"]),
        "Learning_Consistency": float(request.form["consistency"]),
    }
    
    result = analyze_student(input_data)
    
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
