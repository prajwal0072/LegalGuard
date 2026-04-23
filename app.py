from flask import Flask, request, jsonify, render_template
import fitz
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
    return render_template("home.html")

# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    text_content = ""
    
    # 1. Check for Pasted Text
    pasted_text = request.form.get('contract_text')
    if pasted_text:
        text_content = pasted_text
        
    # 2. Check for File Upload
    if 'contract_file' in request.files:
        file = request.files['contract_file']
        if file.filename != '' and file.filename.endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text_content = "".join([page.get_text() for page in doc])

    if not text_content:
        return render_template('result.html', error="No content provided or invalid file type.")

    # 3. Run your Legal-BERT logic
    results = analyze_contract(text_content)

    # 4. Return a results page (or the same page with results)
    return render_template('result.html', report=results)


if __name__ == "__main__":
    # debug=True auto-reloads on file changes during development
    app.run(debug=True, port=5000)