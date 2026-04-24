from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_connection, create_tables
from flask import Flask, request, jsonify, render_template, redirect, flash
import fitz
from model_engine import analyze_contract
from functools import wraps

app = Flask(__name__)
app.secret_key = "supersecret"

create_tables()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ── CORS (local dev) ──────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response
# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if not user:
            flash("User does not exist.", "error")
            return redirect(url_for("login"))

        if not check_password_hash(user["password"], password):
            flash("Incorrect password.", "error")
            return redirect(url_for("login"))

        session["user_id"]    = user["id"]
        session["user_email"] = user["email"]
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("register"))

        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            flash("Email already registered.", "error")
            return redirect(url_for("register"))

        cur.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, generate_password_hash(password))
        )
        conn.commit()
        conn.close()

        flash("Account created! Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Analysis endpoint ─────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if "user_id" not in session:
        return redirect(url_for("login"))

    text_content = ""

    pasted_text = request.form.get('contract_text')

    if pasted_text and pasted_text.strip():
        text_content = pasted_text

    if 'contract_file' in request.files:
        file = request.files['contract_file']
        if file.filename != '' and file.filename.endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text_content = "".join([page.get_text() for page in doc])

    # 🔴 THIS IS THE FIX
    if not text_content:
        return render_template("home.html", error="No content provided or invalid file type.")

    results = analyze_contract(text_content)
    return render_template("result.html", report=results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")