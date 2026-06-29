"""
Build real head-to-head + recent-form data for the dashboard.

Downloads the public international-results dataset (~47k matches, 1872-present)
and writes data/history.json containing, for the 48 World Cup teams:
  - "form": each team's most recent results (date, opponent, score, W/D/L)
  - "h2h":  every fixture pairing's past meetings and an aggregate W-D-L

Requires internet (run on your own machine or in Streamlit Cloud, same as
train_model.py). The dashboard shows this data when present and a clear
"how to populate" message when it is not. No data is fabricated.

Run:   python build_history.py
"""
import csv
import json
import os
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")

# Map the dataset's spellings to the names used in elo_ratings.csv.
ALIAS = {
    "Czech Republic": "Czechia", "Turkey": "Turkiye", "Curaçao": "Curacao",
    "Cape Verde Islands": "Cape Verde", "Republic of Ireland": "Ireland",
    "United States": "United States", "South Korea": "South Korea",
    "DR Congo": "DR Congo", "Ivory Coast": "Ivory Coast",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.strip()


def load_wc_teams():
    with open(os.path.join(DATA, "elo_ratings.csv"), newline="", encoding="utf-8") as f:
        return [r["team"] for r in csv.DictReader(f)]


def load_fixtures():
    with open(os.path.join(DATA, "fixtures.csv"), newline="", encoding="utf-8") as f:
        return [(r["home_team"], r["away_team"]) for r in csv.DictReader(f)]


def main():
    teams = set(load_wc_teams())
    norm_to_team = {norm(t): t for t in teams}

    def canon(name):
        name = ALIAS.get(name, name)
        return norm_to_team.get(norm(name))

    print("Downloading match history ...")
    with urllib.request.urlopen(RESULTS_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8", "ignore")
    rows = list(csv.DictReader(text.splitlines()))
    print(f"  {len(rows):,} historical matches")

    # collect matches that involve at least one WC team
    per_team = {t: [] for t in teams}
    pair_meetings = {}
    for r in rows:
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError):
            continue
        h, a = canon(r["home_team"]), canon(r["away_team"])
        date = r.get("date", "")
        if h:
            res = "W" if hs > as_ else ("L" if hs < as_ else "D")
            per_team[h].append((date, r["away_team"], hs, as_, res))
        if a:
            res = "W" if as_ > hs else ("L" if as_ < hs else "D")
            per_team[a].append((date, r["home_team"], as_, hs, res))
        if h and a:
            key = tuple(sorted((h, a)))
            pair_meetings.setdefault(key, []).append((date, h, hs, a, as_))

    form = {}
    for t, ms in per_team.items():
        ms.sort(key=lambda x: x[0])
        form[t] = [dict(date=d, opp=o, gf=gf, ga=ga, res=res)
                   for d, o, gf, ga, res in ms[-8:]][::-1]

    h2h = {}
    for (home, away) in load_fixtures():
        key = tuple(sorted((home, away)))
        meets = sorted(pair_meetings.get(key, []), key=lambda x: x[0])
        hw = dw = aw = 0
        recent = []
        for d, hteam, hs, ateam, as_ in meets:
            # normalise orientation to this fixture's home/away
            if hteam == home:
                gh, ga = hs, as_
            else:
                gh, ga = as_, hs
            if gh > ga:
                hw += 1
            elif gh < ga:
                aw += 1
            else:
                dw += 1
            recent.append(dict(date=d, gh=gh, ga=ga))
        h2h[f"{home}|{away}"] = dict(
            home_wins=hw, draws=dw, away_wins=aw, total=len(meets),
            meetings=recent[-8:][::-1])

    out = dict(form=form, h2h=h2h)
    with open(os.path.join(DATA, "history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote data/history.json — form for {len(form)} teams, "
          f"h2h for {len(h2h)} fixtures")


if __name__ == "__main__":
    main()
