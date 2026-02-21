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

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    import io

    # Get values directly from form
    original = request.form.get("original")
    projected = request.form.get("projected")
    improvement = request.form.get("improvement")
    risk = request.form.get("risk")
    strategy = request.form.get("strategy")
    summary = request.form.get("summary")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Academic Diagnostic Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Current Effectiveness: {original}", styles["Normal"]))
    elements.append(Paragraph(f"Projected Effectiveness: {projected}", styles["Normal"]))
    elements.append(Paragraph(f"Improvement %: {improvement}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {risk}%", styles["Normal"]))
    elements.append(Paragraph(f"Strategy: {strategy}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Advisory Summary:", styles["Heading2"]))
    elements.append(Paragraph(summary, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Academic_Report.pdf",
        mimetype="application/pdf"
    )
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
