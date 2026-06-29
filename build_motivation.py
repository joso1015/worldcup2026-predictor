"""
Build group-stage motivation factors for the dashboard.

This script is intentionally group-stage scoped. It does not try to model knockout
motivation, because knockout incentives are structurally different: every match is
an elimination match.

Inputs:
  data/qualified_teams.csv  explicit team statuses supplied by the user
  data/elo_ratings.csv      tournament teams

Output:
  data/motivation_status.csv

For another tournament, update the data files and qualified_teams.csv, then rerun:
  py build_motivation.py
  py generate_outputs.py 10000
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
QUALIFIED_PATH = os.path.join(DATA, "qualified_teams.csv")
OUTPUT_PATH = os.path.join(DATA, "motivation_status.csv")

STATUS_FACTORS = {
    "qualified": (0.94, 1.04, "Already qualified; rotation risk applied"),
    "likely_qualified": (0.97, 1.02, "Likely qualified; mild rotation risk applied"),
    "live": (1.00, 1.00, "Still live; no motivation adjustment"),
    "likely_eliminated": (0.99, 1.01, "Likely eliminated; slight openness applied"),
    "eliminated": (0.98, 1.00, "Eliminated; mild intensity reduction applied"),
}


def load_explicit_statuses() -> dict[str, dict[str, str]]:
    if not os.path.exists(QUALIFIED_PATH):
        return {}
    df = pd.read_csv(QUALIFIED_PATH).fillna("")
    statuses = {}
    for row in df.itertuples(index=False):
        team = str(getattr(row, "team", "")).strip()
        status = str(getattr(row, "status", "qualified")).strip().lower() or "qualified"
        note = str(getattr(row, "note", "")).strip()
        if team:
            statuses[team] = {"status": status, "note": note}
    return statuses


def build_status_rows() -> list[dict[str, object]]:
    teams = pd.read_csv(os.path.join(DATA, "elo_ratings.csv"))["team"].tolist()
    explicit = load_explicit_statuses()
    rows = []
    for team in teams:
        entry = explicit.get(team, {})
        status = entry.get("status", "live")
        if status not in STATUS_FACTORS:
            status = "live"
        attack_factor, defense_risk, default_note = STATUS_FACTORS[status]
        note = entry.get("note") or default_note
        rows.append(dict(
            team=team,
            scope="group",
            status=status,
            attack_factor=attack_factor,
            defense_risk=defense_risk,
            note=note,
        ))
    return rows


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    rows = build_status_rows()
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    qualified = [row["team"] for row in rows if row["status"] == "qualified"]
    print(f"Wrote motivation_status.csv - {len(rows)} teams, {len(qualified)} qualified")
    if qualified:
        print("Qualified teams: " + ", ".join(qualified))


if __name__ == "__main__":
    main()
