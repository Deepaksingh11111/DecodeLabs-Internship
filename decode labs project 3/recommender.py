"""
================================================================================
 TECH STACK RECOMMENDER ENGINE
 DecodeLabs | AI Project 3 — Content-Based Filtering via TF-IDF + Cosine Similarity
================================================================================

This module implements the "Digital Matchmaker" described in the training
kit, following the strict IPO (Input -> Process -> Output) architecture and
the 4-step ranking pipeline:

    1. INGESTION  -> capture user skills (minimum 3 inputs)
    2. SCORING    -> TF-IDF vectorize + Cosine Similarity against job roles
    3. SORTING    -> rank job roles by descending similarity score
    4. FILTERING  -> truncate to Top-N results

All math (TF-IDF weighting + Cosine Similarity) is implemented from first
principles using pure Python -- no scikit-learn shortcuts -- so the
underlying "similarity logic" the kit talks about is fully transparent.
"""

import csv
import math
import os
from collections import Counter


# ------------------------------------------------------------------------
# STEP 0: DATA LOADING
# ------------------------------------------------------------------------

def load_dataset(csv_path):
    """
    Loads job_role -> required_skills from raw_skills.csv.

    Returns:
        list of dicts: [{"role": "Data Scientist", "tags": ["python", "sql", ...]}, ...]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            raw_skills = row["required_skills"]
            tags = normalize_tags(raw_skills.split(","))
            items.append({"role": role, "tags": tags})
    return items


def normalize_tags(raw_tags):
    """
    Cleans and standardizes tags into the shared vocabulary space.
    Lowercases, strips whitespace, and removes empties so that
    'Cloud Computing' and 'cloud computing ' map to the same dimension.
    This directly addresses the kit's "Bridging the Language Barrier"
    requirement: item features and user features must map to the exact
    same vocabulary.
    """
    cleaned = []
    for tag in raw_tags:
        t = tag.strip().lower()
        if t:
            cleaned.append(t)
    return cleaned


# ------------------------------------------------------------------------
# STEP 1: INGESTION (build vocabulary + raw term-frequency vectors)
# ------------------------------------------------------------------------

def build_vocabulary(items, user_tags):
    """
    Builds the shared vocabulary space (the "dimensions" the kit refers to)
    from ALL item tags plus the user's tags, so every term --- whether it
    comes from a job role or the user --- maps to the same index.
    """
    vocab = set()
    for item in items:
        vocab.update(item["tags"])
    vocab.update(user_tags)
    return sorted(vocab)  # sorted for stable, reproducible ordering


def term_frequency_vector(tags, vocabulary):
    """
    Computes a raw term-frequency (TF) vector for a single document
    (a job role's tag list, or the user's tag list) over the shared vocabulary.

    TF = (count of term t in document) / (total terms in document)
    """
    counts = Counter(tags)
    total_terms = len(tags) if len(tags) > 0 else 1  # avoid divide-by-zero
    return [counts.get(term, 0) / total_terms for term in vocabulary]


# ------------------------------------------------------------------------
# STEP 2: SCORING -- TF-IDF WEIGHTING + COSINE SIMILARITY
# ------------------------------------------------------------------------

def compute_idf(all_documents_tags, vocabulary):
    """
    Computes Inverse Document Frequency for every term in the vocabulary,
    across all "documents" (here: every job role's tag list).

    IDF = log( Total Documents / Documents containing term t )

    The log dampens the penalty for high-frequency (generic) words,
    exactly as described in "The mathematical mechanics of TF-IDF".
    We add 1 inside the log denominator (Laplace-style smoothing) to
    avoid division-by-zero if a vocabulary term never appears (e.g. a
    brand-new user-only skill not present in any job role yet -- this
    is the "Item/User Cold Start" edge case from the kit).
    """
    total_docs = len(all_documents_tags)
    idf = {}
    for term in vocabulary:
        doc_count = sum(1 for doc_tags in all_documents_tags if term in doc_tags)
        # Smoothed IDF: prevents math errors on unseen terms (cold start safety)
        idf[term] = math.log((total_docs + 1) / (doc_count + 1)) + 1
    return idf


def tfidf_vector(tf_vector, idf_dict, vocabulary):
    """
    Combines TF and IDF: weight(term) = TF(term) * IDF(term)
    Returns the final weighted feature vector for one document.
    """
    return [tf_vector[i] * idf_dict[vocabulary[i]] for i in range(len(vocabulary))]


def cosine_similarity(vector_a, vector_b):
    """
    Cosine Similarity = (A . B) / (||A|| * ||B||)

    Measures the angle between two vectors -- invariant to magnitude,
    so a user with 3 skills and a job role with 8 skills can still
    score perfectly aligned (1.0) if their *direction* (relative skill
    emphasis) matches. This is why the kit chooses Cosine over Euclidean.

    Handles the "Cold Start" case explicitly: if either vector is all
    zeros (no shared vocabulary terms at all), similarity is defined as 0
    rather than raising a divide-by-zero error.
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold start safeguard -- no meaningful direction to compare

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------------------------------
# STEP 3 & 4: SORTING + FILTERING (the public-facing recommend function)
# ------------------------------------------------------------------------

def recommend(user_skills, items=None, csv_path=None, top_n=3):
    """
    Full IPO pipeline entry point.

    Args:
        user_skills (list[str]): raw skill strings from the user
                                  (e.g. ["Python", "Cloud Computing", "Automation"])
        items (list[dict]): pre-loaded dataset (optional, used for speed in the GUI)
        csv_path (str): path to raw_skills.csv (used if items is None)
        top_n (int): how many results to return (default Top-3, per the kit)

    Returns:
        list of (role: str, score: float) tuples, sorted descending by score,
        truncated to top_n. Also returns the cleaned user vocabulary used,
        for transparency in the interface.

    Raises:
        ValueError: if fewer than 3 user skills are provided (the kit's
                    "minimum of three user inputs" requirement for
                    sufficient data density).
    """
    if items is None:
        if csv_path is None:
            raise ValueError("Either 'items' or 'csv_path' must be provided.")
        items = load_dataset(csv_path)

    # ---- STEP 1: INGESTION ----
    if user_skills is None or len([s for s in user_skills if s.strip()]) < 3:
        raise ValueError(
            "Insufficient data density: at least 3 user skills are required "
            "for accurate matching (per Project 3 requirements)."
        )

    user_tags = normalize_tags(user_skills)

    # ---- Build shared vocabulary + document corpus ----
    vocabulary = build_vocabulary(items, user_tags)
    all_documents_tags = [item["tags"] for item in items]

    idf_dict = compute_idf(all_documents_tags, vocabulary)

    # User profile vector (TF-IDF weighted)
    user_tf = term_frequency_vector(user_tags, vocabulary)
    user_vector = tfidf_vector(user_tf, idf_dict, vocabulary)

    # ---- STEP 2: SCORING ----
    scored = []
    for item in items:
        item_tf = term_frequency_vector(item["tags"], vocabulary)
        item_vector = tfidf_vector(item_tf, idf_dict, vocabulary)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((item["role"], score))

    # ---- STEP 3: SORTING ----
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # ---- STEP 4: FILTERING ----
    top_results = scored[:top_n]

    return top_results


def explain_match(user_skills, role_name, items=None, csv_path=None):
    """
    Bonus transparency helper: shows exactly which skills overlapped
    between the user and a specific job role -- useful for the interface
    to explain *why* a recommendation was made.
    """
    if items is None:
        items = load_dataset(csv_path)

    user_tags = set(normalize_tags(user_skills))
    role_item = next((i for i in items if i["role"] == role_name), None)
    if role_item is None:
        return []

    role_tags = set(role_item["tags"])
    overlap = sorted(user_tags & role_tags)
    return overlap


if __name__ == "__main__":
    # Quick smoke test when running this file directly
    sample_skills = ["Python", "Cloud Computing", "Automation"]
    csv_file = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
    results = recommend(sample_skills, csv_path=csv_file, top_n=3)

    print(f"Input skills: {sample_skills}\n")
    print("Top 3 Recommended Career Paths:")
    for rank, (role, score) in enumerate(results, start=1):
        print(f"  {rank}. {role:<28} | Match Score: {score:.4f}")
