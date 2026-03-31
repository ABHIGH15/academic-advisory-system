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
# Analyze Route (SAFE + DEBUG ENABLED)
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    try:
        # ✅ Safe input handling (no crash if missing)
        form_input = {
            "attendance": float(request.form.get("attendance", 0)),
            "quiz_avg": float(request.form.get("quiz_avg", 0)),
            "internal": float(request.form.get("internal", 0)),
            "study_hours": float(request.form.get("study_hours", 0)),
            "study_days": float(request.form.get("study_days", 0)),
            "practice_per_week": float(request.form.get("practice_per_week", 0)),
            "assignment_delay": float(request.form.get("assignment_delay", 0)),
            "sleep_hours": float(request.form.get("sleep_hours", 0)),
            "screen_time": float(request.form.get("screen_time", 0)),
            "confidence": request.form.get("confidence", "medium"),
            "anxiety": request.form.get("anxiety", "medium")
        }

        # 🔥 Debug print (visible in Render logs)
        print("FORM INPUT:", form_input)

        # Run engine
        result = analyze_student(form_input)

        return render_template("result.html", result=result)

    except Exception as e:
        # 🔥 Show exact error (VERY IMPORTANT for debugging)
        return f"""
        <h1>❌ Internal Server Error</h1>
        <pre>{str(e)}</pre>
        """

# --------------------------------------------------
# Download PDF Report
# --------------------------------------------------

@app.route("/download_report", methods=["POST"])
def download_report():

    import json
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter

    # 👇 FULL result passed as JSON
    result = json.loads(request.form.get("full_result"))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # ===== TITLE =====
    elements.append(Paragraph("Academic Intelligence Report", styles["Title"]))
    elements.append(Spacer(1, 15))

    # ===== PERFORMANCE =====
    elements.append(Paragraph(f"Performance Score: {result['Effectiveness']}%", styles["Heading2"]))
    elements.append(Paragraph(f"Level: {result['Student_Level']}", styles["Normal"]))
    elements.append(Paragraph(f"Strategy: {result['Strategy']}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    # ===== PRIORITY =====
    elements.append(Paragraph("Priority Areas:", styles["Heading2"]))
    for item in result["Priority_Areas"]:
        elements.append(Paragraph(f"{item['feature']} ({item['value']})", styles["Normal"]))
        elements.append(Paragraph(item["meaning"], styles["Normal"]))

        bullet = ListFlowable(
            [Paragraph(a, styles["Normal"]) for a in item["actions"]]
        )
        elements.append(bullet)
        elements.append(Spacer(1, 10))

    # ===== TARGET PLAN =====
    elements.append(Paragraph("Improvement Plan:", styles["Heading2"]))
    for step in result["Target_Plan"]:
        elements.append(Paragraph(f"{step['feature']}: {step['current']} → {step['target']}", styles["Normal"]))
        elements.append(Paragraph(step["meaning"], styles["Normal"]))

        bullet = ListFlowable(
            [Paragraph(a, styles["Normal"]) for a in step["actions"]]
        )
        elements.append(bullet)
        elements.append(Spacer(1, 10))

    # ===== INSIGHTS =====
    if result.get("Advanced_Insights"):
        elements.append(Paragraph("Behavioral Insights:", styles["Heading2"]))
        bullet = ListFlowable(
            [Paragraph(i, styles["Normal"]) for i in result["Advanced_Insights"]]
        )
        elements.append(bullet)
        elements.append(Spacer(1, 10))

    # ===== TECHNIQUES =====
    if result.get("Techniques"):
        elements.append(Paragraph("Recommended Techniques:", styles["Heading2"]))
        bullet = ListFlowable(
            [Paragraph(t, styles["Normal"]) for t in result["Techniques"]]
        )
        elements.append(bullet)
        elements.append(Spacer(1, 10))

    # ===== WELLNESS =====
    if result.get("Wellness_Tips"):
        elements.append(Paragraph("Wellness Tips:", styles["Heading2"]))
        bullet = ListFlowable(
            [Paragraph(t, styles["Normal"]) for t in result["Wellness_Tips"]]
        )
        elements.append(bullet)
        elements.append(Spacer(1, 10))

    # ===== CONCEPT =====
    elements.append(Paragraph("Concept Analysis:", styles["Heading2"]))
    bullet = ListFlowable(
        [Paragraph(c, styles["Normal"]) for c in result["Concept_Explanation"]]
    )
    elements.append(bullet)

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Academic_Report.pdf",
        mimetype="application/pdf"
    )
    
# --------------------------------------------------
# Run App
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
