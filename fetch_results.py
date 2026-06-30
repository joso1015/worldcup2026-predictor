"""
Fetch match results from Wikipedia and update data/results.csv.
Any new or updated scores are written; the app auto-detects the change via file mtime.

Usage:  python fetch_results.py
        python fetch_results.py --generate   # also re-run generate_outputs.py
"""
import argparse
import csv
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(DATA, "results.csv")

KNOCKOUT_URL = "https://en.wikipedia.org/w/index.php?title=2026_FIFA_World_Cup_knockout_stage&action=raw"

# FIFA country codes used by Wikipedia -> spec team names
FIFA_CODE_MAP = {
    "RSA": "South Africa",
    "CAN": "Canada",
    "GER": "Germany",
    "PAR": "Paraguay",
    "NED": "Netherlands",
    "MAR": "Morocco",
    "BRA": "Brazil",
    "JPN": "Japan",
    "FRA": "France",
    "SWE": "Sweden",
    "CIV": "Ivory Coast",
    "NOR": "Norway",
    "MEX": "Mexico",
    "ECU": "Ecuador",
    "ENG": "England",
    "COD": "DR Congo",
    "USA": "United States",
    "BIH": "Bosnia and Herzegovina",
    "BEL": "Belgium",
    "SEN": "Senegal",
    "POR": "Portugal",
    "CRO": "Croatia",
    "ESP": "Spain",
    "AUT": "Austria",
    "SUI": "Switzerland",
    "ALG": "Algeria",
    "ARG": "Argentina",
    "CPV": "Cape Verde",
    "COL": "Colombia",
    "GHA": "Ghana",
    "AUS": "Australia",
    "EGY": "Egypt",
}

STAGE_MAP = {
    "Round of 32": "r32",
    "Round of 16": "r16",
    "Quarter-finals": "qf",
    "Semi-finals": "sf",
    "Third place": "third",
    "Final": "final",
}


def fetch_wiki_raw():
    req = urllib.request.Request(KNOCKOUT_URL, headers={"User-Agent": "WorldCup2026Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def parse_bracket_scores(text):
    """Parse the RoundN template from the Bracket section.

    Format:
    |Date � Location|{{#invoke:flag|fb|CODE}}|SCORE|{{#invoke:flag|fb|CODE}}|SCORE

    Score format:  '1' (simple), '1 (3)' (includes pens), '' (no score = TBD)
    """
    matches = []

    # Find the bracket section (RoundN template) - find the start and count braces to find end
    lines = text.split("\n")
    bracket_start = None
    for i, line in enumerate(lines):
        if "RoundN|N32" in line:
            bracket_start = i
            break
    if bracket_start is None:
        return matches

    # Collect lines until the RoundN template closes (counting {{ }} nesting)
    bracket_lines = []
    depth = 0
    started = False
    for line in lines[bracket_start:]:
        if not started:
            if "RoundN|N32" in line:
                started = True
        if started:
            bracket_lines.append(line)
            depth += line.count("{{") - line.count("}}")
            if depth <= 0 and started and len(bracket_lines) > 1:
                break

    bracket_text = "\n".join(bracket_lines)

    current_stage = None
    for line in lines:
        line = line.strip()

        # Detect stage from HTML comments: <!--Round of 32-->
        stage_comment = re.search(r"<!--\s*(Round of 32|Round of 16|Quarter-finals|Semi-finals|Final|Match for third place)\s*-->", line)
        if stage_comment:
            label = stage_comment.group(1)
            if label == "Match for third place":
                current_stage = "third"
            else:
                current_stage = STAGE_MAP.get(label)
            continue

        # Match lines with scores - match from the flag pattern (before which there's the date/venue which may contain |)
        # Format: ...|{{#invoke:flag|fb|CODE}}|SCORE|{{#invoke:flag|fb|CODE}}|SCORE
        match_line = re.search(
            r"\{\{#invoke:flag\|fb\|([A-Z]+)\}\}\|"  # home team code
            r"([0-9)( ]*?)\|"                         # home score (may be empty or "1 (3)")
            r"\{\{#invoke:flag\|fb\|([A-Z]+)\}\}"    # away team code
            r"(?:\s*\{\{pso\}\})?"                   # optional penalty marker
            r"\|([0-9)( ]*)",                         # away score (may be empty or "1 (4)")
            line
        )

        if match_line:
            home_code = match_line.group(1)
            home_score_raw = match_line.group(2).strip()
            away_code = match_line.group(3)
            away_score_raw = match_line.group(4).strip()

            # Parse score: get the 120-min score (before any pens)
            def parse_score(s):
                m = re.match(r"(\d+)", s)
                return int(m.group(1)) if m else None

            hg = parse_score(home_score_raw)
            ag = parse_score(away_score_raw)

            if hg is None or ag is None:
                continue  # Match not played yet

            home = FIFA_CODE_MAP.get(home_code, home_code)
            away = FIFA_CODE_MAP.get(away_code, away_code)

            if current_stage:
                matches.append((current_stage, home, away, hg, ag))

    return matches


def load_existing_results():
    existing = {}
    if not os.path.exists(RESULTS):
        return existing, []
    with open(RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        key = (row["stage"].strip(), row["home_team"].strip(), row["away_team"].strip())
        existing[key] = (int(row["home_score"]), int(row["away_score"]))
    return existing, rows


def save_results(rows):
    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "home_team", "away_team", "home_score", "away_score"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch match results from Wikipedia")
    parser.add_argument("--generate", action="store_true", help="Also re-run generate_outputs.py after fetching")
    args = parser.parse_args()

    print("Fetching Wikipedia data...")
    try:
        text = fetch_wiki_raw()
    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")
        sys.exit(1)

    print("Parsing knockout match scores...")
    new_matches = parse_bracket_scores(text)
    print(f"Found {len(new_matches)} knockout match(es) with scores")

    if not new_matches:
        print("No new match results found.")
        return

    existing, rows = load_existing_results()
    updated = 0
    added = 0

    for stage, home, away, hg, ag in new_matches:
        key = (stage, home, away)
        if key in existing and existing[key] == (hg, ag):
            continue
        if key in existing:
            for row in rows:
                if row["stage"].strip() == stage and row["home_team"].strip() == home and row["away_team"].strip() == away:
                    row["home_score"] = str(hg)
                    row["away_score"] = str(ag)
                    updated += 1
                    break
        else:
            rows.append({"stage": stage, "home_team": home, "away_team": away, "home_score": str(hg), "away_score": str(ag)})
            added += 1

    if updated or added:
        save_results(rows)
        print(f"Updated {updated} existing match(es), added {added} new match(es). Total: {len(rows)}")
    else:
        print("All results already up to date.")

    if args.generate:
        print("Re-running generate_outputs.py...")
        os.chdir(HERE)
        ret = os.system(f"{sys.executable} generate_outputs.py 10000")
        if ret == 0:
            print("Done! Dashboard and predictions regenerated.")
        else:
            print(f"generate_outputs.py exited with code {ret}.")


if __name__ == "__main__":
    main()
