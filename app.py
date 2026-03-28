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

    try:
        # Get values safely
        effectiveness = request.form.get("effectiveness", "N/A")
        risk = request.form.get("risk", "N/A")
        strategy = request.form.get("strategy", "N/A")
        summary = request.form.get("summary", "No summary available")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("Academic Intelligence Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        # Core Metrics
        elements.append(Paragraph(f"Effectiveness: {effectiveness}%", styles["Normal"]))
        elements.append(Paragraph(f"Risk Score: {risk}%", styles["Normal"]))
        elements.append(Paragraph(f"Strategy: {strategy}", styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Summary
        elements.append(Paragraph("Summary:", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(summary, styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="Academic_Report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return f"""
        <h1>❌ PDF Generation Error</h1>
        <pre>{str(e)}</pre>
        """

# --------------------------------------------------
# Run App
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
