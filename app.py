from flask import Flask, render_template, request
from academic_engine import analyze_student
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from flask import send_file
import io

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
@app.route("/download_report", methods=["POST"])
def download_report():

    result = analyze_student(request.form)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Academic Diagnostic Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Current Effectiveness: {result['Simulation']['Original']}", styles["Normal"]))
    elements.append(Paragraph(f"Projected Effectiveness: {result['Simulation']['Projected']}", styles["Normal"]))
    elements.append(Paragraph(f"Improvement %: {result['Simulation']['Improvement_Percent']}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {result['Risk_Score']}%", styles["Normal"]))
    elements.append(Paragraph(f"Strategy: {result['Strategy']}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Advisory Summary:", styles["Heading2"]))
    elements.append(Paragraph(result["AI_Summary"], styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer,
                     as_attachment=True,
                     download_name="Academic_Report.pdf",
                     mimetype="application/pdf")

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
