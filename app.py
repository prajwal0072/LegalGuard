from flask import Flask, request, jsonify, render_template
import pdfplumber
import os
from model_engine import analyze_contract

app = Flask(__name__)

# ── Allow the browser to talk to Flask during local dev ──────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    text = ""

    # File input
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        try:
            with pdfplumber.open(file) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                text = " ".join(pages)
        except Exception as e:
            return render_template("result.html", error=f"PDF Error: {str(e)}")

        if not text.strip():
            return render_template("result.html", error="No text found in PDF")

    # Text input
    elif "text" in request.form and request.form["text"].strip():
        text = request.form["text"]

    else:
        return render_template("result.html", error="No input provided")

    # Run model
    try:
        report = analyze_contract(text)
    except Exception as e:
        return render_template("result.html", error=f"Analysis Error: {str(e)}")

    return render_template("result.html", report=report)


@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    # debug=True auto-reloads on file changes during development
    app.run(debug=True, port=5000)