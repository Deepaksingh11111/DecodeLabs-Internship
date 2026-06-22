"""
================================================================================
 TECH STACK RECOMMENDER -- INTERACTIVE CLI
 DecodeLabs | AI Project 3
================================================================================

A fully working command-line interface around the recommender engine.
Run this file directly: `python app.py`
"""

import os
import sys

from recommender import load_dataset, recommend, explain_match


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_skills.csv")


def print_banner():
    print("=" * 64)
    print("   DECODELABS AI ENGINE -- TECH STACK RECOMMENDER (Project 3)")
    print("   Content-Based Filtering | TF-IDF + Cosine Similarity")
    print("=" * 64)


def print_available_skills(items):
    all_tags = set()
    for item in items:
        all_tags.update(item["tags"])
    print("\nAvailable skill vocabulary (sample):")
    sample = sorted(all_tags)
    line = ", ".join(sample)
    # wrap at ~70 chars for readability
    while line:
        print("  " + line[:70])
        line = line[70:]


def get_user_skills():
    print("\nEnter at least 3 skills, separated by commas.")
    print("Example: Python, Cloud Computing, Automation")
    while True:
        raw = input("\nYour skills > ").strip()
        if not raw:
            print("  Please enter something.")
            continue
        skills = [s.strip() for s in raw.split(",") if s.strip()]
        if len(skills) < 3:
            print(f"  You entered {len(skills)} skill(s). Minimum 3 required for accurate matching.")
            continue
        return skills


def get_top_n():
    raw = input("How many recommendations would you like? [default 3] > ").strip()
    if not raw:
        return 3
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        print("  Invalid number, defaulting to 3.")
        return 3


def display_results(user_skills, results, items):
    print("\n" + "-" * 64)
    print(f"  INPUT PROFILE : {user_skills}")
    print("-" * 64)
    print(f"  {'RANK':<6}{'CAREER PATH':<28}{'MATCH SCORE':<14}{'CONFIDENCE'}")
    print("-" * 64)

    for rank, (role, score) in enumerate(results, start=1):
        pct = score * 100
        bar_len = int(pct // 5)  # 20-char bar max
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {rank:<6}{role:<28}{score:<14.4f}[{bar}] {pct:5.1f}%")

    print("-" * 64)

    # Offer explanation for top match
    if results:
        top_role = results[0][0]
        overlap = explain_match(user_skills, top_role, items=items)
        if overlap:
            print(f"\n  Why '{top_role}'? Shared skills: {', '.join(overlap)}")
        else:
            print(f"\n  '{top_role}' was the closest directional match, "
                  f"though no exact skill terms overlapped.")


def main():
    print_banner()

    try:
        items = load_dataset(CSV_PATH)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\nLoaded {len(items)} job roles from raw_skills.csv")

    while True:
        print("\nMENU:")
        print("  1. Get career recommendations")
        print("  2. View available skill vocabulary")
        print("  3. Exit")
        choice = input("\nSelect an option (1-3) > ").strip()

        if choice == "1":
            user_skills = get_user_skills()
            top_n = get_top_n()
            try:
                results = recommend(user_skills, items=items, top_n=top_n)
            except ValueError as e:
                print(f"\n  ERROR: {e}")
                continue
            display_results(user_skills, results, items)

        elif choice == "2":
            print_available_skills(items)

        elif choice == "3":
            print("\nGoodbye! Keep building, future AI Engineer. 🚀")
            break

        else:
            print("  Invalid option. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
