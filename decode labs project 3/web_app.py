"""
================================================================================
 TECH STACK RECOMMENDER -- WEB GUI (Flask)
 DecodeLabs | AI Project 3
================================================================================

A fully working browser-based interface for the recommendation engine.

Run:
    python web_app.py

Then open:
    http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request

from recommender import load_dataset, recommend, explain_match

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_skills.csv")
ITEMS = load_dataset(CSV_PATH)
@app.route('/')
def index():
    # Flask automatically looks inside the templates/ folder for this file
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    user_input = ""
    top_n = 3
    explanations = {}

    if request.method == "POST":
        user_input = request.form.get("skills", "").strip()
        top_n_raw = request.form.get("top_n", "3").strip()

        try:
            top_n = max(1, int(top_n_raw))
        except ValueError:
            top_n = 3

        skills = [s.strip() for s in user_input.split(",") if s.strip()]

        try:
            scored = recommend(skills, items=ITEMS, top_n=top_n)
            results = []
            for role, score in scored:
                overlap = explain_match(skills, role, items=ITEMS)
                results.append({
                    "role": role,
                    "score": score,
                    "pct": round(score * 100, 1),
                    "overlap": overlap,
                })
        except ValueError as e:
            error = str(e)

    all_tags = sorted({tag for item in ITEMS for tag in item["tags"]})
    all_roles = sorted({item["role"] for item in ITEMS})

    return render_template(
        "index.html",
        results=results,
        error=error,
        user_input=user_input,
        top_n=top_n,
        all_tags=all_tags,
        all_roles=all_roles,
        total_roles=len(ITEMS),
    )


if __name__ == "__main__":
    print("=" * 64)
    print("  DecodeLabs Tech Stack Recommender -- Web Interface")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("=" * 64)
    app.run(debug=True, port=5000)
