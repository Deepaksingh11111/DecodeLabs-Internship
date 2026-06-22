# Tech Stack Recommender — DecodeLabs AI Project 3

A content-based recommendation engine that maps a user's skills to the
closest-matching career paths, using **TF-IDF weighting** + **Cosine
Similarity** — built from first principles in pure Python, with two
interfaces (terminal + browser).

This satisfies the Project 3 brief: *"Create a simple recommendation
system based on user preferences. Take user input → match preferences
using logic/similarity → display recommended items."*

## Files

| File | Purpose |
|---|---|
| `raw_skills.csv` | Dataset: 20 job roles mapped to their required skills |
| `recommender.py` | Core engine — TF-IDF + Cosine Similarity from scratch (no ML libraries) |
| `app.py` | Interactive **command-line interface** |
| `web_app.py` | Interactive **browser interface** (Flask) |
| `templates/index.html` | Web UI styling/markup |

## How it works (the IPO pipeline)

```
INPUT (your skills)  →  PROCESS (TF-IDF vectorize + Cosine Similarity)  →  OUTPUT (ranked Top-N list)
```

1. **Ingestion** — your comma-separated skills are cleaned and normalized
   into the shared vocabulary space (lowercased, trimmed). Minimum 3
   skills are required for reliable matching.
2. **Scoring** — every job role and your profile are converted into
   TF-IDF weighted vectors (rare/specific skills count more than common
   ones), then compared using cosine similarity (the angle between
   vectors — magnitude-independent).
3. **Sorting** — all 20 roles are ranked by descending similarity score.
4. **Filtering** — only the Top-N (default 3) are shown, to avoid choice
   overload.

It also handles the **Cold Start problem** described in the kit: if your
skills share zero vocabulary with any role, similarity returns `0.0`
safely instead of crashing.

## Running it

### Option A — Command line
```bash
python app.py
```
Follow the on-screen menu: enter skills, get ranked recommendations,
browse the skill vocabulary.

### Option B — Browser interface
```bash
pip install flask
python web_app.py
```
Then open **http://127.0.0.1:5000** in your browser.

## Example

Input: `Python, Cloud Computing, Automation`

Output (Top 3):
1. Site Reliability Engineer — 47.9%
2. Network Engineer — 47.4%
3. Systems Administrator — 45.7%

## Extending it

- Add more rows to `raw_skills.csv` to grow the job-role dataset.
- Adjust `top_n` to show more/fewer results.
- `explain_match()` in `recommender.py` shows exactly which skills
  overlapped for any result — useful for debugging or building trust
  in a recommendation.
- Swap `raw_skills.csv` for a movies/books/products dataset to repurpose
  this exact engine for a different recommendation domain — the IPO
  pipeline doesn't change, only the data does.
