from flask import Flask, render_template, request, send_file
from academic_engine import analyze_student
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io
import os

app = Flask(__name__)

# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# Analyze Route
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    # Collect NEW realistic form inputs
    form_input = {
        "attendance": request.form["attendance"],
        "quiz_avg": request.form["quiz_avg"],
        "internal": request.form["internal"],
        "study_hours": request.form["study_hours"],
        "study_days": request.form["study_days"],
        "practice_per_week": request.form["practice_per_week"],
        "assignment_delay": request.form["assignment_delay"],
        "sleep_hours": request.form["sleep_hours"],
        "screen_time": request.form["screen_time"],
        "confidence": request.form["confidence"],
        "anxiety": request.form["anxiety"]
    }

    # Run full hybrid soft computing engine
    result = analyze_student(form_input)

    return render_template("result.html", result=result)

# --------------------------------------------------
# Download PDF Report
# --------------------------------------------------

@app.route("/download_report", methods=["POST"])
def download_report():

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

    elements.append(Paragraph("Academic Optimization Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Current Effectiveness: {original}%", styles["Normal"]))
    elements.append(Paragraph(f"Projected Effectiveness: {projected}%", styles["Normal"]))
    elements.append(Paragraph(f"Improvement Potential: {improvement}%", styles["Normal"]))
    elements.append(Paragraph(f"Academic Risk Score: {risk}%", styles["Normal"]))
    elements.append(Paragraph(f"Recommended Strategy: {strategy}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Advisory Summary:", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(summary, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Academic_Optimization_Report.pdf",
        mimetype="application/pdf"
    )

# --------------------------------------------------
# Run App
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
