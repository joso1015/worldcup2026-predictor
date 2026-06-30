"""
Run the prediction engine and write deliverables into outputs/:
  - predictions_groupstage.csv   both models + probabilities for all 72 games
  - tournament_odds.csv          per-team advancement % and title odds
  - dashboard.html               interactive master-detail single-page app

The dashboard is a self-contained HTML/CSS/JS app: a scrollable, sortable master
list of matches on the left; click any match to open a detail panel on the right
with tabs for Overview, the full prediction Math, model-implied Odds, Head-to-head
History, and Squads. Head-to-head/form data loads from data/history.json (run
build_history.py) and squads from data/squads.csv; both degrade gracefully.

Usage:  python generate_outputs.py [n_sims]
"""
import datetime as dt
import json
import os
import sys

import pandas as pd

import accuracy
import model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
AS_OF = ("Elo as of 2026-05-26 (eloratings.net) · official FIFA schedule · "
         "temperatures are seasonal estimates · odds are model-implied")

PROFILE_PRESETS = [
  dict(
    key="default",
    label="Default",
    description="Balanced, slightly more open tournament baseline.",
    params=model.Params(home_adv=80.0, elo_to_goals=1.65,
                        base_total=2.70, rho=-0.04),
  ),
  dict(
    key="conservative",
    label="Conservative",
    description="Tighter, lower-scoring matches.",
    params=model.Params(home_adv=80.0, elo_to_goals=1.45,
              base_total=2.35, rho=-0.08),
  ),
  dict(
    key="high_scoring",
    label="High scoring",
    description="More open games and bigger margins.",
    params=model.Params(home_adv=80.0, elo_to_goals=1.85,
              base_total=2.90, rho=-0.02),
  ),
]


def weekday(d):
    try:
        return dt.datetime.strptime(d, "%Y-%m-%d").strftime("%a %d %b")
    except Exception:
        return d


def load_history():
    p = os.path.join(DATA, "history.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"form": {}, "h2h": {}}


def load_squads():
    p = os.path.join(DATA, "squads.csv")
    squads = {}
    if os.path.exists(p):
        df = pd.read_csv(p).fillna("")
        for r in df.itertuples(index=False):
            squads.setdefault(r.team, []).append(
                dict(player=r.player, position=r.position, club=r.club))
    return {k: v for k, v in squads.items() if v}


def load_motivation():
    p = os.path.join(DATA, "motivation_status.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p).fillna("")
    out = {}
    for r in df.itertuples(index=False):
        team = str(getattr(r, "team", "")).strip()
        if not team:
            continue
        out[team] = dict(
            scope=str(getattr(r, "scope", "group") or "group"),
            status=str(getattr(r, "status", "live") or "live"),
            attack_factor=float(getattr(r, "attack_factor", 1.0) or 1.0),
            defense_risk=float(getattr(r, "defense_risk", 1.0) or 1.0),
            note=str(getattr(r, "note", "") or ""),
        )
    return out


def motivation_info(motivation, team):
    return motivation.get(team, dict(
        scope="group", status="live", attack_factor=1.0,
        defense_risk=1.0, note="Still live; no motivation adjustment"))


def score_from_lambdas(P, home, away, params, lh, la):
    lh = min(max(lh, params.min_lambda), params.max_lambda)
    la = min(max(la, params.min_lambda), params.max_lambda)
    matrix = P.score_matrix(lh, la, params)
    i, j = divmod(matrix.argmax(), matrix.shape[1])
    return dict(
        most_likely=f"{int(i)}-{int(j)}",
        expanded=f"{round(lh)}-{round(la)}",
        xg_home=round(float(lh), 3),
        xg_away=round(float(la), 3),
    )


def motivation_adjusted_match(P, fx, params, motivation, overrides):
    mno = int(fx.match_no)
    home_info = motivation_info(motivation, fx.home_team)
    away_info = motivation_info(motivation, fx.away_team)
    played = mno in P.played_group
    if played:
        return dict(
            active=False, scope="group", home_status=home_info["status"],
            away_status=away_info["status"], most_likely=None,
            expanded=None, xg_home=None, xg_away=None,
            note="Played match; motivation layer is only applied to unplayed group-stage fixtures.",
        )

    def adjusted(params_for_model):
        lh, la = P.expected_goals(fx.home_team, fx.away_team, params_for_model,
                                  venue_country=fx.country, venue=fx.venue,
                                  overrides=overrides)
        lh *= home_info["attack_factor"] * away_info["defense_risk"]
        la *= away_info["attack_factor"] * home_info["defense_risk"]
        return score_from_lambdas(P, fx.home_team, fx.away_team,
                                  params_for_model, lh, la)

    base = adjusted(params)
    exp = adjusted(model.expanded_params(params))
    parts = []
    for team, info in ((fx.home_team, home_info), (fx.away_team, away_info)):
        if info["status"] != "live":
            parts.append(f"{team}: {info['status'].replace('_', ' ')} - {info['note']}")
    note = "; ".join(parts) if parts else "No explicit motivation adjustment for this fixture."
    return dict(
        active=True, scope="group",
        home_status=home_info["status"], away_status=away_info["status"],
        home_attack_factor=round(home_info["attack_factor"], 3),
        home_defense_risk=round(home_info["defense_risk"], 3),
        away_attack_factor=round(away_info["attack_factor"], 3),
        away_defense_risk=round(away_info["defense_risk"], 3),
        most_likely=base["most_likely"], expanded=exp["expanded"],
        xg_home=base["xg_home"], xg_away=base["xg_away"], note=note,
    )


def add_motivation_columns(gp, P, params, motivation):
    rows = []
    for fx in P.fixtures.itertuples(index=False):
        mot = motivation_adjusted_match(P, fx, params, motivation,
                                        P.fixture_overrides(fx))
        rows.append(dict(
            match_no=int(fx.match_no),
            motivation_home_status=mot["home_status"],
            motivation_away_status=mot["away_status"],
            motivation_predicted_score=mot["most_likely"] or "",
            motivation_expanded_score=mot["expanded"] or "",
            motivation_xg_home=mot["xg_home"] or "",
            motivation_xg_away=mot["xg_away"] or "",
            motivation_note=mot["note"],
        ))
    extra = pd.DataFrame(rows)
    return gp.merge(extra, on="match_no", how="left")


def profile_summary(p):
    return dict(
        home_adv=round(float(p.home_adv), 2),
        elo_to_goals=round(float(p.elo_to_goals), 2),
        base_total=round(float(p.base_total), 2),
        rho=round(float(p.rho), 2),
        heat_tempo=round(float(p.heat_tempo), 2),
        altitude_penalty=round(float(p.altitude_penalty), 2),
        ml_weight=round(float(p.ml_weight), 2),
    )


def build_profile_data(P, n_sims, preset, motivation=None, export_csv=False):
    params = preset["params"]
    pe = model.expanded_params(params)
    motivation = motivation or {}
    odds = P.simulate_tournament(n_sims=n_sims, p=params)
    title = {r.team: r.win_title for r in odds.itertuples(index=False)}

    def cell(v):
        return v if (v != "" and v is not None) else None

    matches = []
    for fx in P.fixtures.itertuples(index=False):
        ov = P.fixture_overrides(fx)
        base = P.predict_match(fx.home_team, fx.away_team, params,
                               venue_country=fx.country, venue=fx.venue, overrides=ov)
        exp = P.predict_match(fx.home_team, fx.away_team, pe,
                              venue_country=fx.country, venue=fx.venue, overrides=ov)
        motivation_adjusted = motivation_adjusted_match(P, fx, params,
                    motivation, ov)
        math = P.explain_match(fx.home_team, fx.away_team, params,
                               venue_country=fx.country, venue=fx.venue, overrides=ov)
        th, ta = title.get(fx.home_team, 0), title.get(fx.away_team, 0)
        cs5 = [[s, pr, round(1 / pr, 2) if pr > 1e-6 else None]
               for s, pr in base["top_scores"][:5]]
        mno = int(fx.match_no)
        played = mno in P.played_group
        actual = ("%d-%d" % P.played_group[mno]) if played else None
        matches.append(dict(
            played=int(played), actual=actual,
            correct=int(bool(played and actual == base["most_likely_score"])),
            id=int(fx.match_no), group=fx.group, md=int(fx.match_day),
            date=fx.date_local, weekday=weekday(fx.date_local),
            kickoff=fx.kickoff_local, home=fx.home_team, away=fx.away_team,
            elo_home=int(self_elo(P, fx.home_team)),
            elo_away=int(self_elo(P, fx.away_team)),
            venue=fx.venue, city=fx.city, country=fx.country,
            temp=int(fx.expected_temp_c), indoor=int(fx.indoor),
            home_rest=cell(getattr(fx, "home_rest_h", "")),
            home_dist=cell(getattr(fx, "home_dist_km", "")),
            away_rest=cell(getattr(fx, "away_rest_h", "")),
            away_dist=cell(getattr(fx, "away_dist_km", "")),
            ms=base["most_likely_score"], exp=exp["expected_scoreline"],
            mot=motivation_adjusted,
            ph=base["p_home_win"], pd=base["p_draw"], pa=base["p_away_win"],
            xgh=base["lambda_home"], xga=base["lambda_away"],
            btts=exp["btts"], over=exp["over25"],
            topb=[[s, pr] for s, pr in base["top_scores"]],
            tope=[[s, pr] for s, pr in exp["top_scores"]],
            cs5=cs5, odds=math["odds"],
            th=round(1 / th, 1) if th else None, thp=round(th, 4),
            ta=round(1 / ta, 1) if ta else None, tap=round(ta, 4),
            math=math,
        ))

    # knockout match details for dashboard (full match-like objects)
    ko_rounds = P.resolve_knockout_bracket(params)
    ko_match_details = {}
    if ko_rounds:
        for rnd in ko_rounds:
            for idx, g in enumerate(rnd.get("games", [])):
                h, a = g.get("home", ""), g.get("away", "")
                if h and a:
                    pe_ko = model.expanded_params(params)
                    kp = P.predict_knockout_match(h, a, params)
                    kpe = P.predict_knockout_match(h, a, pe_ko)
                    base = P.predict_match(h, a, params)
                    exp = P.predict_match(h, a, pe_ko)
                    math = P.explain_match(h, a, params)
                    def ko_odds(prob):
                        return round(1.0 / prob, 2) if prob and prob > 1e-6 else None
                    # lookup venue from ko_bracket
                    ko_venue, ko_city, ko_country, ko_indoor, ko_temp = "", "", "", 0, 22
                    for kf in P.ko_bracket:
                        if kf.get("home_team") == h and kf.get("away_team") == a:
                            ko_venue = kf.get("venue", "")
                            ko_city = kf.get("city", "")
                            ko_country = kf.get("country", "")
                            ko_indoor = int(kf.get("indoor", 0))
                            ko_temp = float(kf.get("temp_c", 22))
                            break
                    th, ta = title.get(h, 0), title.get(a, 0)
                    cs5 = [[s, pr, round(1 / pr, 2) if pr > 1e-6 else None]
                           for s, pr in base["top_scores"][:5]]
                    ko_id = "%s-%d" % (rnd.get("label", ""), idx)
                    ko_match_details[ko_id] = dict(
                        id=ko_id, ko=True, round=rnd.get("label", ""),
                        home=h, away=a,
                        elo_home=int(self_elo(P, h)),
                        elo_away=int(self_elo(P, a)),
                        venue=ko_venue, city=ko_city, country=ko_country,
                        indoor=ko_indoor, temp=int(ko_temp),
                        home_rest=None, home_dist=None,
                        away_rest=None, away_dist=None,
                        group="R32", md=0, date="", weekday="", kickoff="",
                        ms=base["most_likely_score"],
                        exp=exp["expected_scoreline"],
                        ph=base["p_home_win"], pd=base["p_draw"], pa=base["p_away_win"],
                        xgh=base["lambda_home"], xga=base["lambda_away"],
                        btts=exp["btts"], over=exp["over25"],
                        topb=[[s, pr] for s, pr in base["top_scores"]],
                        tope=[[s, pr] for s, pr in exp["top_scores"]],
                        cs5=cs5, odds=math["odds"],
                        th=round(1 / th, 1) if th else None, thp=round(th, 4),
                        ta=round(1 / ta, 1) if ta else None, tap=round(ta, 4),
                        math=math,
                        mot=dict(active=False, home_status="qualified",
                                 away_status="qualified",
                                 most_likely=None, expanded=None,
                                 xg_home=None, xg_away=None,
                                 note="Knockout match — motivation layer is group-stage only."),
                        played=int(g.get("played", False)),
                        actual=("%d-%d" % P.played_ko.get(
                            int(ko_id.split("-")[-1]) if ko_id.split("-")[-1].isdigit() else 0,
                            (0, 0))) if g.get("played") else None,
                        correct=0, note=g.get("note", ""),
                        # KO-specific fields
                        p_home_adv=kp["p_home_adv"], p_away_adv=kp["p_away_adv"],
                        p_home_90=kp["p_home_90"], p_draw_90=kp["p_draw_90"],
                        p_away_90=kp["p_away_90"],
                        odds_home_adv=ko_odds(kp["p_home_adv"]),
                        odds_away_adv=ko_odds(kp["p_away_adv"]),
                    )

    if export_csv:
        gp = P.predict_all_group_matches(params)
        gp = add_motivation_columns(gp, P, params, motivation)
        gp.to_csv(os.path.join(OUT, "predictions_groupstage.csv"), index=False)
        odds.to_csv(os.path.join(OUT, "tournament_odds.csv"), index=False)
        ko_rows = []
        for rnd in ko_rounds:
            for g in rnd.get("games", []):
                ko_rows.append(dict(
                    stage=rnd.get("label", ""), home=g.get("home", ""),
                    away=g.get("away", ""), score_90min=g.get("score", ""),
                    winner=g.get("winner", ""), note=g.get("note", ""),
                    p_home_adv=g.get("p_home", 0), p_away_adv=g.get("p_away", 0),
                    played=int(g.get("played", False)),
                ))
        pd.DataFrame(ko_rows).to_csv(os.path.join(OUT, "predictions_knockout.csv"), index=False)

    champ = odds.iloc[0]
    played_n = len(P.played_group)
    hits = sum(1 for m in matches if m["played"] and m["correct"])
    result_hits = sum(1 for m in matches if m["played"] and
              accuracy.result_from_score(m["ms"]) == accuracy.result_from_score(m["actual"]))
    expanded_hits = sum(1 for m in matches if m["played"] and m["exp"] == m["actual"])
    expanded_result_hits = sum(1 for m in matches if m["played"] and
                   accuracy.result_from_score(m["exp"]) == accuracy.result_from_score(m["actual"]))
    return dict(
        label=preset["label"],
        description=preset["description"],
        params=profile_summary(params),
        n_sims=int(n_sims), champion=dict(team=champ.team, p=round(champ.win_title, 4)),
      progress=dict(played=played_n, total=72, exact_hits=hits,
              result_hits=result_hits, expanded_hits=expanded_hits,
              expanded_result_hits=expanded_result_hits),
        matches=matches,
        title_odds=[dict(team=r.team, p=round(r.win_title, 4))
                    for r in odds.head(12).itertuples(index=False)],
        knockout_bracket=ko_rounds,
        ko_matches=ko_match_details,
    )


def build_data(P, n_sims):
    hist = load_history()
    squads = load_squads()
    motivation = load_motivation()
    summary = accuracy.summary_for_params(P, PROFILE_PRESETS[0]["params"],
                                          include_sweep=True,
                                          include_ml=bool(P.model_ml))
    accuracy.write_outputs(summary, OUT)
    active_presets = list(PROFILE_PRESETS)
    winners = summary.get("winners") or {}

    def params_from_winner(row):
      return model.Params(
        home_adv=float(row["home_adv"]),
        elo_to_goals=float(row["elo_to_goals"]),
        base_total=float(row["base_total"]),
        rho=float(row["rho"]),
        heat_tempo=float(row["heat_tempo"]),
        altitude_penalty=float(row["altitude_penalty"]),
        ml_weight=float(row["ml_weight"]),
      )

    fitted_modes = [
      ("best_ml_score", "Best ML Exact Scores", "Current-results fitted mode, optimised for Most Likely exact scores.",
       winners.get("most_likely_score")),
      ("best_ml_result", "Best ML Results", "Current-results fitted mode, optimised for Most Likely win/draw/loss results.",
       winners.get("most_likely_result")),
      ("best_expanded_fit", "Best Expanded Fit", "Current-results fitted mode, optimised for Expanded results.",
       winners.get("expanded_result")),
    ]
    for key, label, description, row in fitted_modes:
      if row:
        active_presets.append(dict(
          key=key,
          label=label,
          description=description,
          params=params_from_winner(row),
        ))
    profiles = {}
    profile_meta = {}
    for preset in active_presets:
        key = preset["key"]
        profiles[key] = build_profile_data(P, n_sims, preset,
                                           motivation=motivation,
                                           export_csv=(key == "default"))
        profile_meta[key] = dict(
            label=profiles[key]["label"],
            description=profiles[key]["description"],
            params=profiles[key]["params"],
        )
    return dict(
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        as_of=AS_OF,
        active_profile="default",
        profile_order=[preset["key"] for preset in active_presets],
        profile_meta=profile_meta,
        profiles=profiles,
        form=hist.get("form", {}),
        h2h=hist.get("h2h", {}),
        squads=squads,
        motivation_available=bool(motivation),
        history_available=bool(hist.get("h2h")),
        squads_available=bool(squads),
        accuracy_summary={k: v for k, v in summary.items() if k != "rows"},
    )


def self_elo(P, team):
    return P.elo_of[team]


PAGE = r"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>World Cup 2026 Score Predictor</title><style>__CSS__</style></head>
<body>
<header>
  <div class='htop'>
    <div class='htitle'>World Cup 2026</div>
    <div class='seg'>
      <button id='tabSummary' class='on' onclick='showTab("summary")'>Summary</button>
      <button id='tabGroupStage' onclick='showTab("groupstage")'>Group Stage</button>
      <button id='tabBracket' onclick='showTab("bracket")'>Bracket</button>
    </div>
  </div>
  <div class='seg mode-seg'>__PROFILE_BTNS__</div>
</header>
<div class='toolbar' id='toolbar'>
    <input id='q' placeholder='Search…' oninput='onQuery(this.value)'>
    <div class='seg'>
      <button id='sg' class='on' onclick='setSort("group")'>Group</button>
      <button id='sd' onclick='setSort("date")'>Date</button>
    </div>
</div>
<div class='hsub' id='hsub'></div>
<main>
  <aside id='master'></aside>
  <section id='detail'><div class='empty'>Select a match to see the full breakdown.</div></section>
</main>
<script>var DATA=__DATA__;</script>
<script>__JS__</script>
</body></html>"""

CSS = r"""
:root{--bg:#0b1020;--panel:#121829;--card:#171f33;--card2:#1d2740;--ink:#e9edf5;
 --mut:#9aa6bd;--acc:#28d17c;--acc2:#3b82f6;--line:#26324c;--hot:#28d17c}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,
 Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.45}
header{display:flex;justify-content:space-between;align-items:center;gap:10px;
 padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.htop{display:flex;align-items:center;gap:8px;min-width:0}
.htitle{font-size:16px;font-weight:700;white-space:nowrap}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 16px;
 border-bottom:1px solid var(--line);background:rgba(18,24,41,.72)}
#q{background:var(--card);border:1px solid var(--line);color:var(--ink);
 border-radius:8px;padding:6px 10px;font-size:13px;min-width:160px;width:min(300px,40vw)}
.seg{display:flex;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-width:0}
.mode-seg{overflow-x:auto;max-width:100%;scrollbar-width:thin;flex:1;justify-content:flex-end}
.seg button{background:transparent;border:0;color:var(--mut);padding:6px 10px;font-size:12px;cursor:pointer}
.mode-seg button{white-space:nowrap}
.seg button.on{background:var(--acc2);color:#fff;font-weight:600}
.hsub{color:var(--mut);font-size:11px;padding:5px 16px;border-bottom:1px solid var(--line)}
.statusline{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.status-chip{background:rgba(59,130,246,.10);border:1px solid var(--line);border-radius:999px;padding:3px 8px;color:var(--mut)}
.status-chip b{color:var(--ink)}
main{display:grid;grid-template-columns:340px 1fr;height:calc(100vh - 96px)}
main.bracket-active,main.bracket-tree,main.summary-active{height:calc(100vh - 60px)}
main.summary-active{grid-template-columns:0 1fr !important}
main.summary-active #master{overflow:hidden;padding:0;border:0;max-width:0}
main.summary-active #toolbar{display:none !important}
main.summary-active~.hsub,main.summary-active .hsub{display:none !important}
main.bracket-active{grid-template-columns:340px 1fr !important}
main.bracket-active{grid-template-columns:1fr 1fr}
main.bracket-active #toolbar{display:none}
main.bracket-active .hsub{display:none}
main.bracket-tree{grid-template-columns:0 1fr}
main.bracket-tree #toolbar{display:none}
main.bracket-tree .hsub{display:none}
main.bracket-tree #master{overflow:hidden;padding:0;border:0}

#master{overflow-y:auto;border-right:1px solid var(--line);padding:10px}
.glabel{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;
 margin:12px 6px 6px;font-weight:700}
.mcard{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:9px 11px;margin-bottom:8px;cursor:pointer;transition:.12s}
.mcard:hover{border-color:var(--acc2);background:var(--card2)}
.mcard.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.mc-top{display:flex;justify-content:space-between;font-size:10.5px;color:var(--mut);margin-bottom:5px}
.mc-grp{color:var(--acc2);font-weight:700}
.mc-mid{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px}
.mc-team{font-weight:600;font-size:13px}.mc-team.r{text-align:right}
.mc-score{background:rgba(59,130,246,.16);border-radius:7px;padding:1px 9px;
 font-weight:800;font-variant-numeric:tabular-nums}
.mc-score.played{background:rgba(40,209,124,.18);color:var(--acc)}
.mc-grp .ft{color:var(--acc);font-weight:700}
.d-sc .s.played{background:rgba(40,209,124,.18);color:var(--acc)}
.resbanner{background:rgba(40,209,124,.10);border:1px solid var(--acc);border-radius:8px;
 padding:12px 15px;margin-bottom:13px;font-size:14px}.resbanner b{font-size:16px}
#detail{overflow-y:auto;padding:20px 24px}
.empty{color:var(--mut);padding:40px;text-align:center}
.d-head{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:6px}
.d-teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:14px}
.d-tm .nm{font-size:20px;font-weight:800}.d-tm.r{text-align:right}.d-tm .el{color:var(--mut);font-size:12px}
.d-sc{text-align:center}.d-sc .s{font-size:30px;font-weight:800;background:rgba(59,130,246,.16);
 border-radius:8px;padding:3px 16px;font-variant-numeric:tabular-nums}
.d-sc .lab{display:block;font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.d-meta{color:var(--mut);font-size:12.5px;margin-top:12px}
.tabs{display:flex;gap:6px;margin:16px 0 14px;flex-wrap:wrap}
.tabs button{background:var(--card);border:1px solid var(--line);color:var(--mut);
 padding:7px 14px;border-radius:8px;font-size:13px;cursor:pointer}
.tabs button.on{background:var(--acc);color:#06281a;font-weight:700;border-color:var(--acc)}
.sec{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:15px 17px;margin-bottom:13px}
.sec h4{margin:0 0 10px;font-size:13px;color:#f4f7fb}
.summary-grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin-bottom:13px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 13px}
.metric .label{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.metric .value{font-size:20px;font-weight:800;font-variant-numeric:tabular-nums;margin-top:4px}
.fit-grid{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:10px}
.fit-card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.fit-card .target{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em;font-weight:700}
.fit-card .primary{font-size:22px;font-weight:800;margin:3px 0 8px;font-variant-numeric:tabular-nums}
.fit-card .mini{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px}
.fit-card .mini div{background:var(--card2);border-radius:7px;padding:6px;text-align:center;font-size:11px;color:var(--mut)}
.fit-card .mini b{display:block;color:var(--ink);font-size:13px}
.settings-text{color:var(--mut);font-size:11.5px;line-height:1.35}
.fit-details{margin-top:12px}.fit-details summary{color:var(--mut);font-size:12px;cursor:pointer;margin-bottom:8px}
.table-wrap{overflow-x:auto;margin:-2px 0 0}.table-wrap table{min-width:760px}
.match-check-cards{display:none}
.check-card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.check-top{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px;font-size:12px;color:var(--mut)}
.check-match{color:var(--ink);font-weight:700}.check-score{font-variant-numeric:tabular-nums;color:var(--ink)}
.check-row{display:grid;grid-template-columns:84px 1fr auto auto;gap:8px;align-items:center;font-size:12px;padding:4px 0;border-top:1px solid rgba(36,48,73,.55)}
.check-row b{color:var(--mut);font-weight:600}.check-row .pred{font-variant-numeric:tabular-nums;color:var(--ink)}
.sec .step{color:var(--acc);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.kv{display:grid;grid-template-columns:1fr auto;gap:4px 10px;font-size:13px}
.kv .k{color:var(--mut)}.kv .v{font-variant-numeric:tabular-nums;text-align:right}
.eq{background:#0c1322;border:1px solid var(--line);border-radius:9px;padding:10px 12px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#cdd7ea;margin:8px 0;overflow-x:auto}
.eq b{color:var(--acc)}
.wdl{display:flex;gap:8px;margin:10px 0}
.wdl .b{flex:1;text-align:center;background:var(--card2);border-radius:8px;padding:8px 4px;font-size:12px}
.wdl .b .p{font-size:17px;font-weight:800;font-variant-numeric:tabular-nums}
.wdl .b.hot{background:var(--acc);color:#06281a}.wdl .b .l{color:var(--mut)}.wdl .b.hot .l{color:#06281a}
.readline{background:rgba(59,130,246,.08);border:1px solid var(--line);border-radius:8px;
 padding:9px 11px;color:#cdd7ea;font-size:12.5px;margin:9px 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:5px 10px;font-size:12px;font-variant-numeric:tabular-nums}
.chip b{color:var(--acc)}
.analysis-card{background:linear-gradient(180deg,rgba(59,130,246,.10),rgba(40,209,124,.06));
 border:1px solid var(--line);border-radius:8px;padding:13px 15px;margin-top:12px}
.analysis-card h4{margin:0 0 8px;font-size:13px}.analysis-card p{margin:8px 0 0;color:#cbd5e1;font-size:13px}
.analysis-pick{display:inline-flex;gap:8px;align-items:center;background:rgba(40,209,124,.12);
 border:1px solid rgba(40,209,124,.38);border-radius:999px;padding:5px 10px;font-size:12px;color:var(--mut)}
.analysis-pick b{color:var(--ink);font-size:14px}.analysis-note{color:var(--mut);font-size:11.5px;margin-top:8px}
table.t{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.t th,table.t td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:center}
table.t th{color:var(--mut);font-weight:600}
table.t td.l,table.t th.l{text-align:left}
.grid{border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums;margin-top:6px}
.grid td,.grid th{width:34px;height:30px;text-align:center;border:1px solid var(--line);color:#dfe6f2}
.grid th{color:var(--mut);font-weight:600;border:0}
.muted{color:var(--mut);font-size:12.5px}
.why{background:rgba(40,209,124,.07);border:1px solid var(--line);border-left:3px solid var(--acc);
 border-radius:8px;padding:9px 11px;margin-top:10px;font-size:12.5px;color:#b9c3d6}
.why span{display:block;color:var(--acc);font-weight:700;font-size:10px;text-transform:uppercase;
 letter-spacing:.05em;margin-bottom:3px}
.why i{color:#dfe6f2;font-style:italic}.why code{background:#0c1322;padding:1px 5px;border-radius:4px;font-size:11.5px}
td.live{color:var(--acc);font-weight:700}
.edge-pos{color:var(--acc);font-weight:800}.edge-neg{color:var(--mut)}
tr.best-live td{background:rgba(40,209,124,.08)}
.best-tag{display:inline-block;margin-left:5px;border:1px solid rgba(40,209,124,.45);border-radius:999px;
 padding:1px 5px;color:var(--acc);font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.04em}
code{background:#0c1322;padding:1px 5px;border-radius:4px;font-size:11.5px}
.note{background:rgba(59,130,246,.08);border:1px solid var(--line);border-radius:8px;
 padding:11px 13px;color:var(--mut);font-size:12.5px}
.formrow{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}
.fp{width:22px;height:22px;border-radius:5px;display:flex;align-items:center;justify-content:center;
 font-size:11px;font-weight:700}
.fp.W{background:#1e7d52;color:#eafff3}.fp.D{background:#4b5572;color:#e9edf5}.fp.L{background:#7d2b34;color:#ffeaec}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:640px){.cols2{grid-template-columns:1fr}}
.col h5{margin:0 0 4px;font-size:14px}.col h5 .ct{color:var(--mut);font-weight:400;font-size:11px}
.poslab{color:var(--acc);font-size:10px;text-transform:uppercase;letter-spacing:.05em;
 margin:10px 0 2px;font-weight:700}
.plist{list-style:none;margin:0;padding:0}
.prow{display:flex;justify-content:space-between;gap:8px;font-size:12.5px;padding:3px 0;
 border-bottom:1px solid rgba(36,48,73,.5)}
.prow .pn{font-weight:600}.prow .pc{color:var(--mut);font-size:11px;text-align:right;white-space:nowrap}
.frow{display:flex;align-items:center;gap:8px;font-size:12px;padding:5px 0;
 border-bottom:1px solid rgba(36,48,73,.5)}
.frow .fres{width:20px;height:20px;border-radius:5px;display:flex;align-items:center;
 justify-content:center;font-weight:700;font-size:10px;flex:0 0 auto}
.frow .fres.W{background:#1e7d52;color:#eafff3}.frow .fres.D{background:#4b5572}
.frow .fres.L{background:#7d2b34;color:#ffeaec}
.frow .fdate{color:var(--mut);flex:0 0 auto}.frow .fopp{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.frow .fsc{margin-left:auto;font-variant-numeric:tabular-nums;flex:0 0 auto}
@media(max-width:900px){
  html,body{height:auto;min-height:100%}
  header{align-items:flex-start;padding:12px 12px 8px}.htop{width:100%;justify-content:space-between}
  .htitle{font-size:16px}.mode{width:100%;justify-content:flex-start}.mode span{flex:0 0 auto}
  .toolbar{padding:8px 12px;align-items:stretch}.toolbar #q{width:100%;min-width:0}
  main{display:block;height:auto}#master{border-right:0;border-bottom:1px solid var(--line);max-height:34vh;overflow-y:auto;padding:8px 10px}
  #detail{overflow:visible;padding:14px 12px}.summary-grid{grid-template-columns:repeat(2,minmax(130px,1fr))}
  .fit-grid{grid-template-columns:1fr}.table-wrap{margin-left:-4px;margin-right:-4px}
}
@media(max-width:640px){
  header{gap:8px}.toolbar{flex-direction:column}.seg button{padding:8px 10px;font-size:12px}
  .hsub{padding:7px 12px}.status-chip{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #master{max-height:30vh}.glabel{margin-top:8px}.mcard{padding:8px}.mc-top span:last-child{display:none}
  #detail{padding:12px 10px}.d-head{padding-bottom:11px}.d-teams{grid-template-columns:1fr;gap:8px;text-align:left}
  .d-tm.r{text-align:left}.d-sc{text-align:left}.d-sc .s{font-size:24px;display:inline-block}.d-meta{font-size:11.5px}
  .tabs{overflow-x:auto;flex-wrap:nowrap;padding-bottom:2px}.tabs button{white-space:nowrap;flex:0 0 auto}
  .summary-grid{grid-template-columns:1fr}.metric .value{font-size:18px}.sec{padding:12px;margin-bottom:10px}
  .wdl{display:grid;grid-template-columns:1fr}.kv{grid-template-columns:1fr}.kv .v{text-align:left;margin-bottom:4px}
  .analysis-card{padding:12px}.analysis-pick{width:100%;justify-content:space-between;border-radius:8px}
  .played-table{display:none}.match-check-cards{display:block}.check-row{grid-template-columns:72px 1fr auto auto}
}
/* ---- bracket master panel (collapsible sections) ---- */
.br-master{padding:10px}
.brm-round{margin-bottom:8px}
.brm-header{display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--card);border:1px solid var(--line);border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;user-select:none}
.brm-header:hover{border-color:var(--acc2)}
.brm-header .arrow{color:var(--mut);font-size:10px;transition:transform .15s}
.brm-header.open .arrow{transform:rotate(90deg)}
.brm-body{padding:6px 0 2px 6px}
.brm-tbd{color:var(--mut);font-size:12px;padding:8px 10px;font-style:italic}
/* ---- bracket tree visual ---- */
.br-tree-wrap{padding:16px;display:flex;flex-direction:row;gap:20px;overflow-x:auto;overflow-y:auto;height:calc(100vh - 60px);align-items:flex-start}
.br-col{display:flex;flex-direction:column;gap:6px;min-width:190px;flex:1 0 auto}
.br-col-label{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.05em;font-weight:700;margin:0 0 6px;text-align:center;background:var(--bg);padding:4px 0;position:sticky;top:0;z-index:1}
.br-game{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:default}
.br-game.played{border-color:var(--acc)}
.br-game .teams{font-weight:600;font-size:11px;display:flex;justify-content:space-between;gap:6px}
.br-game .score{text-align:center;font-weight:800;margin:2px 0;color:var(--acc);font-size:12px}
.br-game .note{font-size:9px;color:var(--mut);text-align:center;font-style:italic}
.br-game .prob{font-size:9px;color:var(--mut);text-align:center}
/* ---- bracket toggle button ---- */
.br-toggle{margin:0 0 8px;display:flex;gap:6px}
.br-toggle button{background:var(--card);border:1px solid var(--line);color:var(--mut);padding:5px 12px;border-radius:6px;font-size:11px;cursor:pointer}
.br-toggle button.on{background:var(--acc2);color:#fff}
"""

JS = r"""
var state={sel:null,koSel:null,koTab:'Overview',sort:'group',q:'',tab:'Overview',currentTab:'summary',profile:DATA.active_profile,bracketView:'list'};
// ---- Live bookmaker odds (optional) -----------------------------------------
// To pull real correct-score prices from The Odds API, run odds_proxy.py locally.
// The proxy reads THE_ODDS_API_KEY from .env and returns JSON like
// {"correct_score":{"2-1":[{"book":"Book A","odds":4.8}]},"source":"The Odds API"}.
// {home}/{away}/{key} are substituted per match. The feed is fetched fresh EVERY
// time the Odds tab loads. Clear oddsEndpoint to show model fair odds only.
var CONFIG={oddsEndpoint:'http://127.0.0.1:8787/correct-score?home={home}&away={away}', oddsApiKey:''};
function liveOddsInfo(v){
  if(v==null)return null;
  if(Array.isArray(v)){
    var best=null;
    v.forEach(function(item){var info=liveOddsInfo(item);if(info&&(!best||info.odds>best.odds))best=info;});
    return best;
  }
  if(typeof v==='number')return {odds:v,label:v.toFixed(2)};
  if(typeof v==='string'){
    var parsed=parseFloat(v);return isNaN(parsed)?null:{odds:parsed,label:parsed.toFixed(2)};
  }
  if(typeof v==='object'){
    var raw=v.odds!=null?v.odds:(v.decimal!=null?v.decimal:(v.price!=null?v.price:null));
    var odds=parseFloat(raw);if(isNaN(odds))return null;
    var source=v.book||v.bookmaker||v.source||'';
    return {odds:odds,label:odds.toFixed(2)+(source?' · '+source:'')};
  }
  return null;
}
function loadLiveOdds(m){
  var st=document.getElementById('live_status'); if(!st)return;
  function setCell(score,txt,best){
    var el=document.getElementById('lo_'+score.replace('-','_'));if(!el)return;
    el.innerHTML=txt+(best?" <span class='best-tag'>best</span>":'');
    var row=el.parentNode;if(row)row.className=best?'best-live':'';
  }
  function setText(id,txt,cls){var el=document.getElementById(id);if(el){el.innerHTML=txt;el.className=cls||'';}}
  function setResult(key,info,modelOdds){
    if(!info){setText('lr_'+key,'—','live');setText('le_'+key,'—','muted');return false;}
    setText('lr_'+key,esc(info.label),'live');
    if(modelOdds){
      var edge=(info.odds/modelOdds-1)*100;
      setText('le_'+key,(edge>=0?'+':'')+edge.toFixed(0)+'%',edge>0?'edge-pos':'edge-neg');
    }
    return true;
  }
  if(!CONFIG.oddsEndpoint){
    m.cs5.forEach(function(c){setCell(c[0],'',false);});
    ['home','draw','away'].forEach(function(k){setResult(k,null,null);});
    st.innerHTML="Live bookmaker odds are <b>not configured</b>. Set <code>CONFIG.oddsEndpoint</code> "+
      "(top of the page script) to a correct-score feed returning "+
      "<code>{\"correct_score\":{\"2-1\":4.5,…}}</code> or "+
      "<code>{\"correct_score\":{\"2-1\":[{\"book\":\"Book A\",\"odds\":4.8}]}}</code>. "+
      "The highest returned live price in the top five is highlighted automatically. Until then, "+
      "the <b>Model odds</b> column above shows fair prices from this model.";
    return;
  }
  st.textContent='Fetching live odds…';
  var sel=state.sel;
  var url=CONFIG.oddsEndpoint.replace('{home}',encodeURIComponent(m.home))
    .replace('{away}',encodeURIComponent(m.away)).replace('{key}',encodeURIComponent(CONFIG.oddsApiKey||''));
  fetch(url).then(function(r){return r.json();}).then(function(d){
    if(state.sel!==sel)return;               // user moved on
    var cs=(d&&d.correct_score)||{}, priced=[], n=0;
    m.cs5.forEach(function(c){var info=liveOddsInfo(cs[c[0]]);if(info)priced.push({score:c[0],info:info});});
    var best=null;priced.forEach(function(p){if(!best||p.info.odds>best.info.odds)best=p;});
    m.cs5.forEach(function(c){var info=liveOddsInfo(cs[c[0]]);if(info){setCell(c[0],esc(info.label),best&&best.score===c[0]);n++;}else setCell(c[0],'',false);});
    var mr=(d&&d.match_result)||{}, rn=0;
    rn+=setResult('home',liveOddsInfo(mr.home),m.math.odds.home)?1:0;
    rn+=setResult('draw',liveOddsInfo(mr.draw),m.math.odds.draw)?1:0;
    rn+=setResult('away',liveOddsInfo(mr.away),m.math.odds.away)?1:0;
    st.innerHTML=n?("Live odds loaded"+(d.source?(" from "+esc(d.source)):"")+
      (best?(" · best top-five price: <b>"+esc(best.score)+" at "+esc(best.info.label)+"</b>"):"")+
      (rn?(" · live 1X2 loaded"):'')+" · "+new Date().toLocaleTimeString()):
      (rn?("Live 1X2 odds loaded"+(d.source?(" from "+esc(d.source)):"")+"; live correct-score prices are unavailable. · "+new Date().toLocaleTimeString()):
      (d&&d.message?esc(d.message):"The feed returned no correct-score prices for this match."));
  }).catch(function(){
    if(state.sel!==sel)return;
    m.cs5.forEach(function(c){setCell(c[0],'',false);});
    ['home','draw','away'].forEach(function(k){setResult(k,null,null);});
    st.textContent='Could not reach the live odds feed. Start odds_proxy.py locally or clear CONFIG.oddsEndpoint to show model odds only.';
  });
}
function pct(x){return Math.round(x*100)+'%';}
function esc(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function profileData(){return DATA.profiles[state.profile]||DATA.profiles[DATA.active_profile];}
function profileMeta(){return DATA.profile_meta[state.profile]||DATA.profile_meta[DATA.active_profile]||{};}
function activeMatches(){var p=profileData();return (p&&p.matches)||[];}

function filtered(){
  var ms=activeMatches().slice();
  if(state.q){var q=state.q.toLowerCase();
    ms=ms.filter(function(m){return m.home.toLowerCase().indexOf(q)>=0||m.away.toLowerCase().indexOf(q)>=0;});}
  if(state.sort==='date'){ms.sort(function(a,b){return (a.date+a.kickoff).localeCompare(b.date+b.kickoff)||a.id-b.id;});}
  else{ms.sort(function(a,b){return a.group.localeCompare(b.group)||a.md-b.md||a.id-b.id;});}
  return ms;
}
function renderMaster(){
  var ms=filtered(),h='',last=null;
  ms.forEach(function(m){
    var head=state.sort==='date'?m.weekday:('Group '+m.group);
    if(head!==last){h+="<div class='glabel'>"+esc(head)+"</div>";last=head;}
    var score=m.played?m.actual:m.ms;
    h+="<div class='mcard"+(state.sel===m.id?' sel':'')+"' onclick='sel("+m.id+")'>"+
       "<div class='mc-top'><span class='mc-grp'>"+m.group+" · "+(m.played?"<span class='ft'>FT</span>":("MD"+m.md))+"</span>"+
       "<span>"+esc(m.weekday)+" · "+m.kickoff+"</span></div>"+
       "<div class='mc-mid'><span class='mc-team'>"+esc(m.home)+"</span>"+
       "<span class='mc-score"+(m.played?" played":"")+"'>"+score+"</span>"+
       "<span class='mc-team r'>"+esc(m.away)+"</span></div></div>";
  });
  document.getElementById('master').innerHTML=h||"<div class='empty'>No matches.</div>";
}
function setSort(s){state.sort=s;document.getElementById('sg').className=s==='group'?'on':'';
  document.getElementById('sd').className=s==='date'?'on':'';renderMaster();}
function onQuery(v){state.q=v;renderMaster();}
function byId(id){return activeMatches().filter(function(m){return m.id===id;})[0];}
function sel(id){state.sel=id;syncProfileButtons();renderMaster();renderDetail(byId(id));}
function setTab(t){
  state.tab=t;
  if(state.currentTab==='bracket'&&state.koSel){
    renderKoDetail(state.koSel);
  }else{
    renderDetail(byId(state.sel));
  }
}

function showTab(tab){
  state.currentTab=tab;
  state.koSel=null;state.sel=null;
  var mainEl=document.querySelector('main');
  mainEl.classList.remove('bracket-active','bracket-tree','summary-active');
  var tb=document.getElementById('toolbar'); if(tb)tb.style.display='flex';
  var hs=document.getElementById('hsub'); if(hs)hs.style.display='block';
  if(tab==='summary'){
    mainEl.classList.add('summary-active');
    if(tb)tb.style.display='none';
    if(hs)hs.style.display='none';
    document.getElementById('master').innerHTML='';
    document.getElementById('master').style.maxWidth='0';
    renderSummary();
    // ensure master is truly hidden
    var mst=document.getElementById('master');
    if(mst)mst.style.display='none';
  }else if(tab==='groupstage'){
    var mst2=document.getElementById('master');
    if(mst2)mst2.style.display='block';
    if(mst2)mst2.style.maxWidth='';
    renderAll();
  }else if(tab==='bracket'){
    state.bracketView='list';
    mainEl.classList.add('bracket-active');
    var mst3=document.getElementById('master');
    if(mst3)mst3.style.display='block';
    if(mst3)mst3.style.maxWidth='';
    if(tb)tb.style.display='none';
    if(hs)hs.style.display='none';
    syncProfileButtons();renderHeader();renderBracketMaster();renderKoDetail(null);
  }
  syncProfileButtons();
}

function toggleBracketView(v){
  state.bracketView=v;
  if(v==='tree'){
    document.querySelector('main').classList.add('bracket-tree');
    document.querySelector('main').classList.remove('bracket-active');
    renderBracketTree();
  }else{
    document.querySelector('main').classList.remove('bracket-tree');
    document.querySelector('main').classList.add('bracket-active');
    renderBracketMaster();renderKoDetail(state.koSel);
  }
}

function renderBracketMaster(){
  var pdata=profileData(), ko=pdata.knockout_bracket||[], km=pdata.ko_matches||{};
  var html='<div class="br-master"><div class="br-toggle">'+
    '<button'+(state.bracketView==='list'?' class="on"':'')+' onclick="toggleBracketView(\'list\')">List</button>'+
    '<button'+(state.bracketView==='tree'?' class="on"':'')+' onclick="toggleBracketView(\'tree\')">Tree</button></div>';
  for(var ri=0;ri<ko.length;ri++){
    var rnd=ko[ri], rl=rnd.label||'Round';
    var hasGames=rnd.games&&rnd.games.some(function(g){return g.home&&g.away;});
    var isOpen=(ri===0||rl==='Round of 32')?' open':'';
    html+='<div class="brm-round"><div class="brm-header'+isOpen+'" onclick="toggleBrm(this)">'+
      '<span class="arrow">&#9654;</span> '+esc(rl)+' <span style="color:var(--mut);font-weight:400;font-size:11px">('+
      (rnd.games?rnd.games.length:0)+')</span></div><div class="brm-body"'+(isOpen?'':' style="display:none"')+'>';
    if(!hasGames){
      html+='<div class="brm-tbd">TBD — pairings depend on earlier rounds</div>';
    }else{
      for(var gi=0;gi<rnd.games.length;gi++){
        var         g=rnd.games[gi];
        if(!g.home||!g.away)continue;
        var kid=esc(rl)+'-'+gi;
        var kd=km[kid];
        var advPct=kd?Math.round(Math.max(kd.p_home_adv,kd.p_away_adv)*100):50;
        var advLabel=kd?('Adv: '+esc(kd.home)+' '+Math.round(kd.p_home_adv*100)+'% &mdash; '+esc(kd.away)+' '+Math.round(kd.p_away_adv*100)+'%'):'';
        var score=g.score||'?';
        html+="<div class='mcard"+(state.koSel===kid?' sel':'')+"' onclick='selKo(\""+kid+"\")'>"+
          "<div class='mc-top'><span class='mc-grp'>"+esc(rl)+"</span>"+
          "<span>"+advLabel+"</span></div>"+
          "<div class='mc-mid'><span class='mc-team'>"+esc(g.home)+"</span>"+
          "<span class='mc-score'>"+esc(score)+"</span>"+
          "<span class='mc-team r'>"+esc(g.away)+"</span></div></div>";
      }
    }
    html+='</div></div>';
  }
  html+='</div>';
  document.getElementById('master').innerHTML=html;
  document.getElementById('detail').innerHTML='<div class="empty">Select a Round of 32 match to see the prediction.</div>';
}

function toggleBrm(el){el.classList.toggle('open');
  var body=el.nextElementSibling;
  if(body)body.style.display=body.style.display==='none'?'':'none';}

function selKo(id){
  state.koSel=id;state.koTab='Overview';
  renderBracketMaster();
  renderKoDetail(id);
}

function renderKoDetail(id){
  if(!id){document.getElementById('detail').innerHTML='<div class="empty">Select a Round of 32 match to see the prediction.</div>'; return;}
  var pdata=profileData(), km=pdata.ko_matches||{}, kd=km[id];
  if(!kd){document.getElementById('detail').innerHTML='<div class="empty">Match not found.</div>'; return;}
  // Build a match-like object so we can reuse renderDetail
  var m=Object.assign({}, kd);
  m.home=kd.home; m.away=kd.away;
  renderDetail(m);
  // Add advancement banner on top of the detail
  var advBanner='<div class="resbanner" style="display:flex;justify-content:space-between;align-items:center">'+
    '<span>Advancement: <b>'+esc(kd.home)+'</b> '+Math.round(kd.p_home_adv*100)+'% &mdash; <b>'+esc(kd.away)+'</b> '+Math.round(kd.p_away_adv*100)+'%</span>'+
    (kd.note?'<span style="font-size:12px;color:var(--mut)">'+kd.note+'</span>':'')+
    '</div>';
  var detail=document.getElementById('detail');
  detail.innerHTML=advBanner+detail.innerHTML;
}

function renderBracketTree(){
  var pdata=profileData(), ko=pdata.knockout_bracket||[], km=pdata.ko_matches||{};
  var html='<div class="br-tree-wrap">';
  for(var ri=0;ri<ko.length;ri++){
    var rnd=ko[ri];
    html+='<div class="br-col"><div class="br-col-label">'+esc(rnd.label)+' ('+(rnd.games?rnd.games.length:0)+')</div>';
    for(var gi=0;gi<(rnd.games||[]).length;gi++){
      var g=rnd.games[gi];
      if(g.home&&g.away){
        var cls='br-game'+(g.played?' played':'');
        html+='<div class="'+cls+'"><div class="teams"><span>'+esc(g.home)+'</span><span>'+esc(g.away)+
          '</span></div><div class="score">'+esc(g.score)+'</div>';
        if(g.note)html+='<div class="note">'+esc(g.note)+'</div>';
        var kid=esc(rnd.label)+'-'+gi;
        var kd=km[kid];
        if(kd)html+='<div class="prob">Adv: '+Math.round(kd.p_home_adv*100)+'% — '+Math.round(kd.p_away_adv*100)+'%</div>';
        html+='</div>';
      }else{
        html+='<div class="br-game"><div class="teams"><span>'+(g.winner||'TBD')+'</span></div></div>';
      }
    }
    html+='</div>';
  }
  html+='</div>';
  document.getElementById('master').innerHTML='';
  document.getElementById('detail').innerHTML=html;
}
function setProfile(key){
  if(!DATA.profiles[key])return;
  state.profile=key;
  state.tab='Overview';
  state.sel=null;
  showTab(state.currentTab);
}
function syncProfileButtons(){
  var ts=document.getElementById('tabSummary'); if(ts)ts.className=state.currentTab==='summary'?'on':'';
  var tg=document.getElementById('tabGroupStage'); if(tg)tg.className=state.currentTab==='groupstage'?'on':'';
  var tb=document.getElementById('tabBracket'); if(tb)tb.className=state.currentTab==='bracket'?'on':'';
  DATA.profile_order.forEach(function(key){
    var el=document.getElementById('pf_'+key);
    if(el)el.className=state.profile===key?'on':'';
  });
}
function renderHeader(){
  var pdata=profileData(), meta=profileMeta(), params=meta.params||{}, prog='';
  if(pdata.progress&&pdata.progress.played){
    prog=pdata.progress.played+"/72 group games in — odds now condition on real results"+
      " · ML scores "+pdata.progress.exact_hits+"/"+pdata.progress.played+
      " · ML results "+pdata.progress.result_hits+"/"+pdata.progress.played+
      " · Expanded scores "+pdata.progress.expanded_hits+"/"+pdata.progress.played+
      " · Expanded results "+pdata.progress.expanded_result_hits+"/"+pdata.progress.played;
  }else prog="pre-tournament — no results entered yet";
  var tune="home +"+Number(params.home_adv||0).toFixed(0)+" · scale "+Number(params.elo_to_goals||0).toFixed(2)+
    " · goals "+Number(params.base_total||0).toFixed(2)+" · rho "+Number(params.rho||0).toFixed(2);
  document.getElementById('hsub').innerHTML="<div class='statusline'>"+
    "<span class='status-chip'>Mode <b>"+esc(meta.label||state.profile)+"</b></span>"+
    "<span class='status-chip'>Champion <b>"+esc(pdata.champion.team)+" "+Math.round(pdata.champion.p*100)+"%</b></span>"+
    "<span class='status-chip'><b>"+prog+"</b></span>"+
    "<span class='status-chip'>Settings <b>"+tune+"</b></span>"+
    "</div>";
}
function renderAll(){
  syncProfileButtons();
  renderHeader();
  renderMaster();
  var ms=filtered();
  if(!ms.length){
    state.sel=null;
    document.getElementById('detail').innerHTML="<div class='empty'>No matches.</div>";
    return;
  }
  if(state.sel==null){renderSummary();return;}
  if(!byId(state.sel)){state.sel=null;renderSummary();return;}
  renderDetail(byId(state.sel));
}

function countCell(done,total){return done+"/"+total+" ("+(total?Math.round(done*100/total):0)+"%)";}
function resultOfScore(score){var p=(''+score).split('-').map(Number);return p[0]>p[1]?'H':(p[0]<p[1]?'A':'D');}
function hitCell(hit){return "<span class='"+(hit?'live':'muted')+"'>"+(hit?'Hit':'Miss')+"</span>";}
function favoriteInfo(m){
  var opts=[{side:'H',team:m.home,label:esc(m.home)+' win',p:m.ph},{side:'D',team:'Draw',label:'Draw',p:m.pd},
    {side:'A',team:m.away,label:esc(m.away)+' win',p:m.pa}];
  opts.sort(function(a,b){return b.p-a.p;});return opts[0];
}
function topScoreText(m){
  var top=(m.topb&&m.topb[0])||[m.ms,0], second=(m.topb&&m.topb[1])||null;
  var tight=second && Math.abs(top[1]-second[1])<0.018;
  return tight?('Top exact scores are tightly grouped around '+top[0]+' and '+second[0]+'.'):
    ('The leading exact-score cell is '+top[0]+' at '+pct(top[1])+'.');
}
function matchReadLine(m){
  var fav=favoriteInfo(m), edge=Math.abs(m.ph-m.pa), favTxt=fav.side==='D'?'Draw':fav.team;
  var lean=fav.side==='D'?'No clear winner':(favTxt+' '+pct(fav.p));
  var xg='xG '+m.xgh.toFixed(2)+'-'+m.xga.toFixed(2);
  var caveat=edge<0.18?'Outcome is fairly balanced.':topScoreText(m);
  return lean+' · '+xg+' · '+caveat;
}
function analysisPick(m){
  var fav=favoriteInfo(m), expRes=resultOfScore(m.exp), msRes=resultOfScore(m.ms);
  if(fav.side==='D')return msRes==='D'?m.ms:(m.pd>0.25?'1-1':m.ms);
  if(fav.side===expRes)return m.exp;
  if(fav.side===msRes && Math.abs(m.ph-m.pa)<0.22)return m.ms;
  if(fav.side==='H'){
    if(m.xgh>=2.15 && m.xga<0.85)return '2-0';
    if(m.xgh>=1.75 && (m.btts>=0.48 || m.xga>=0.85))return '2-1';
    return '1-0';
  }
  if(m.xga>=2.15 && m.xgh<0.85)return '0-2';
  if(m.xga>=1.75 && (m.btts>=0.48 || m.xgh>=0.85))return '1-2';
  return '0-1';
}
function matchAnalysis(m){
  var fav=favoriteInfo(m), pick=analysisPick(m), eloGap=Math.abs(m.elo_home-m.elo_away);
  var stronger=m.elo_home>=m.elo_away?m.home:m.away;
  var xgDiff=Math.abs(m.xgh-m.xga), total=m.xgh+m.xga;
  var tone=fav.p>=0.65?'strong favourite':(fav.p>=0.55?'clear favourite':(fav.p>=0.45?'narrow favourite':'coin-flip range'));
  var para1=fav.side==='D'?
    'The model sees this as a tight matchup rather than a strong lean either way. The draw probability is '+pct(m.pd)+
    ', and the expected-goals gap is only '+xgDiff.toFixed(2)+'.':
    fav.team+' are the '+tone+' at '+pct(fav.p)+', with an expected-goals edge of '+xgDiff.toFixed(2)+
    '. '+stronger+' also hold the Elo advantage, but exact scores remain spread across several nearby cells.';
  var tempo=total>=3.0?'The total-goals profile is open, so I am willing to shade toward a livelier score.':
    (total<=2.25?'The total-goals profile is restrained, so I would keep the scoreline conservative.':
    'The total-goals profile is moderate, which keeps both one-goal and two-goal outcomes live.');
  var context=[];
  if(m.indoor)context.push('The indoor venue reduces weather uncertainty.');
  if(m.home_rest!=null&&m.away_rest!=null){
    var restEdge=Number(m.home_rest)-Number(m.away_rest);
    if(Math.abs(restEdge)>=12)context.push((restEdge>0?m.home:m.away)+' have the cleaner rest profile.');
  }
  if(m.home_dist!=null&&m.away_dist!=null){
    var distEdge=Number(m.home_dist)-Number(m.away_dist);
    if(Math.abs(distEdge)>=1200)context.push((distEdge<0?m.home:m.away)+' have the lighter travel load.');
  }
  var para2=tempo+(context.length?' '+context.join(' '):'')+' My score pick leans on the result probabilities, xG balance, and the expanded scoreline rather than only the modal exact-score cell.';
  return "<div class='analysis-card'><h4>Copilot match analysis</h4><div class='analysis-pick'><span>My predicted score</span><b>"+
    esc(pick)+"</b></div><p>"+esc(para1)+"</p><p>"+esc(para2)+"</p><div class='analysis-note'>Generated from this fixture's model probabilities, Elo gap, xG, venue, rest and travel inputs.</div></div>";
}
function motivationCard(m){
  var mot=m.mot;
  if(!mot||!mot.active)return "";
  function stat(s){return (s||'live').replace(/_/g,' ');}
  return "<div class='analysis-card'><h4>Motivation-adjusted read</h4>"+
    "<div class='analysis-pick'><span>Most likely</span><b>"+esc(mot.most_likely||'—')+"</b></div> "+
    "<div class='analysis-pick'><span>Expanded</span><b>"+esc(mot.expanded||'—')+"</b></div>"+
    "<p>"+esc(mot.note||'No explicit motivation adjustment for this fixture.')+"</p>"+
    "<div class='analysis-note'>Group-stage-only scenario layer. Baseline predictions, probabilities, backtest checks and tournament odds are unchanged. "+
    esc(m.home)+": "+esc(stat(mot.home_status))+" · "+esc(m.away)+": "+esc(stat(mot.away_status))+" · adjusted xG "+
    Number(mot.xg_home||0).toFixed(2)+"-"+Number(mot.xg_away||0).toFixed(2)+"</div></div>";
}
function paramText(r){return "home +"+Number(r.home_adv).toFixed(0)+" Elo · scale "+Number(r.elo_to_goals).toFixed(2)+
  " · goals "+Number(r.base_total).toFixed(2)+" · rho "+Number(r.rho).toFixed(2)+
  " · heat "+Number(r.heat_tempo).toFixed(2)+" · altitude "+Number(r.altitude_penalty).toFixed(2)+
  " · ML "+Number(r.ml_weight).toFixed(2);}
function renderSummary(){
  var pdata=profileData(), p=pdata.progress||{}, s=DATA.accuracy_summary||{}, current=(s.current&&s.current.metrics)||{};
  var played=p.played||s.played||current.played||0;
  var mlScore=p.exact_hits!=null?p.exact_hits:(current.ml_score_correct||0);
  var mlResult=p.result_hits!=null?p.result_hits:(current.ml_result_correct||0);
  var expScore=p.expanded_hits!=null?p.expanded_hits:(current.exp_score_correct||0);
  var expResult=p.expanded_result_hits!=null?p.expanded_result_hits:(current.exp_result_correct||0);
  var h="<div class='d-head'><div class='d-teams'><div class='d-tm'><div class='nm'>Current mode performance</div>"+
    "<div class='el'>Fitted to results entered in data/results.csv</div></div></div></div>"+
    "<div class='summary-grid'>"+
    "<div class='metric'><div class='label'>Matches played</div><div class='value'>"+played+"</div></div>"+
    "<div class='metric'><div class='label'>ML exact scores</div><div class='value'>"+countCell(mlScore,played)+"</div></div>"+
    "<div class='metric'><div class='label'>ML results</div><div class='value'>"+countCell(mlResult,played)+"</div></div>"+
    "<div class='metric'><div class='label'>Expanded scores</div><div class='value'>"+countCell(expScore,played)+"</div></div>"+
    "<div class='metric'><div class='label'>Expanded results</div><div class='value'>"+countCell(expResult,played)+"</div></div>"+
    "</div><div class='note'>"+esc(s.note||"These settings are fitted to currently entered results only and can overfit a small sample.")+"</div>";
  var labels={most_likely_score:'Most Likely: exact scores',most_likely_result:'Most Likely: results',
    expanded_score:'Expanded: exact scores',expanded_result:'Expanded: results'};
  var w=s.winners||{}, rows='', cards='';
  Object.keys(labels).forEach(function(k){var r=w[k]; if(!r)return;
    var primary=k==='most_likely_score'?countCell(r.ml_score_correct,r.played):
      (k==='most_likely_result'?countCell(r.ml_result_correct,r.played):
      (k==='expanded_score'?countCell(r.exp_score_correct,r.played):countCell(r.exp_result_correct,r.played)));
    cards+="<div class='fit-card'><div class='target'>"+labels[k]+"</div><div class='primary'>"+primary+"</div>"+
      "<div class='mini'><div><b>"+countCell(r.ml_score_correct,r.played)+"</b>ML score</div>"+
      "<div><b>"+countCell(r.ml_result_correct,r.played)+"</b>ML result</div>"+
      "<div><b>"+countCell(r.exp_result_correct,r.played)+"</b>Expanded result</div></div>"+
      "<div class='settings-text'>"+esc(paramText(r))+"</div></div>";
    rows+="<tr><td class='l'>"+labels[k]+"</td><td>"+countCell(r.ml_score_correct,r.played)+"</td>"+
      "<td>"+countCell(r.ml_result_correct,r.played)+"</td><td>"+countCell(r.exp_score_correct,r.played)+"</td>"+
      "<td>"+countCell(r.exp_result_correct,r.played)+"</td><td class='l'>"+esc(paramText(r))+"</td></tr>";});
  h+="<div class='sec'><h4>Best-fitting engine settings so far</h4><div class='fit-grid'>"+cards+"</div>"+
    "<details class='fit-details'><summary>Detailed parameter table</summary><div class='table-wrap'><table class='t'><tr><th class='l'>Target</th>"+
    "<th>ML score</th><th>ML result</th><th>Expanded score</th><th>Expanded result</th><th class='l'>Settings</th></tr>"+
    rows+"</table></div></details></div>";
  var playedMatches=(pdata.matches||[]).filter(function(m){return m.played;});
  var matchRows='', matchCards='';
  playedMatches.forEach(function(m){
    var actualResult=resultOfScore(m.actual), mlResult=resultOfScore(m.ms), expResult=resultOfScore(m.exp);
    matchRows+="<tr><td class='l'>"+esc(m.group)+"</td><td class='l'>"+esc(m.home)+" v "+esc(m.away)+"</td>"+
      "<td>"+esc(m.actual)+"</td><td>"+esc(m.ms)+"</td><td>"+hitCell(m.actual===m.ms)+"</td>"+
      "<td>"+hitCell(actualResult===mlResult)+"</td><td>"+esc(m.exp)+"</td>"+
      "<td>"+hitCell(m.actual===m.exp)+"</td><td>"+hitCell(actualResult===expResult)+"</td></tr>";
    matchCards+="<div class='check-card'><div class='check-top'><span class='check-match'>"+esc(m.group)+" · "+
      esc(m.home)+" v "+esc(m.away)+"</span><span class='check-score'>Actual "+esc(m.actual)+"</span></div>"+
      "<div class='check-row'><b>ML</b><span class='pred'>"+esc(m.ms)+"</span>"+hitCell(m.actual===m.ms)+hitCell(actualResult===mlResult)+"</div>"+
      "<div class='check-row'><b>Expanded</b><span class='pred'>"+esc(m.exp)+"</span>"+hitCell(m.actual===m.exp)+hitCell(actualResult===expResult)+"</div></div>";
  });
  h+="<div class='sec'><h4>Played match checks</h4><div class='table-wrap played-table'><table class='t'><tr><th class='l'>Group</th>"+
    "<th class='l'>Match</th><th>Actual</th><th>ML</th><th>ML score</th><th>ML result</th>"+
    "<th>Expanded</th><th>Expanded score</th><th>Expanded result</th></tr>"+matchRows+"</table></div>"+
    "<div class='match-check-cards'>"+matchCards+"</div></div>";
  document.getElementById('detail').innerHTML=h;
}

function renderDetail(m){
  if(!m){
    document.getElementById('detail').innerHTML="<div class='empty'>No matches.</div>";
    return;
  }
  var tabs=['Overview','Model Math','Odds','History','Squads'];
  var ds=m.played?("<span class='s played'>"+m.actual+"</span><span class='lab'>final result</span>")
                 :("<span class='s'>"+m.ms+"</span><span class='lab'>most likely</span>");
  var h="<div class='d-head'><div class='d-teams'>"+
    "<div class='d-tm'><div class='nm'>"+esc(m.home)+"</div><div class='el'>Elo "+m.elo_home+"</div></div>"+
    "<div class='d-sc'>"+ds+"</div>"+
    "<div class='d-tm r'><div class='nm'>"+esc(m.away)+"</div><div class='el'>Elo "+m.elo_away+"</div></div></div>"+
    "<div class='d-meta'>Group "+m.group+" · Match day "+m.md+" · "+esc(m.weekday)+" "+m.kickoff+
    " · "+esc(m.venue)+", "+esc(m.city)+" · "+m.temp+"&deg;C"+(m.indoor?" (indoor)":"")+"</div></div>";
  h+="<div class='tabs'>";
  tabs.forEach(function(t){h+="<button class='"+(state.tab===t?'on':'')+"' onclick=\"setTab('"+t+"')\">"+t+"</button>";});
  h+="</div><div id='tb'></div>";
  document.getElementById('detail').innerHTML=h;
  var body={'Overview':tabOverview,'Model Math':tabMath,'Odds':tabOdds,
            'History':tabHistory,'Squads':tabSquads}[state.tab](m);
  document.getElementById('tb').innerHTML=body;
}

function fmtRT(rest,dist){
  if(rest==null||rest==='')return "First match — no prior game";
  return rest+" h rest · "+Math.round(dist).toLocaleString()+" km travelled";
}
function tabOverview(m){
  var fav=Math.max(m.ph,m.pd,m.pa);
  function b(p,lab,hot){return "<div class='b"+(p===fav&&hot?' hot':'')+"'><div class='p'>"+pct(p)+
    "</div><div class='l'>"+lab+"</div></div>";}
  var top=m.tope.slice(0,4).map(function(t){return "<span class='chip'><b>"+t[0]+"</b> "+pct(t[1])+"</span>";}).join('');
  var banner=m.played?("<div class='resbanner'>Final result: <b>"+m.actual+"</b> · "+
    (m.correct?"the model called the exact score":"model's most-likely call was "+m.ms)+
    "</div>"):"";
  return banner+"<div class='sec'><h4>"+(m.played?"Pre-match prediction":"Predicted result")+"</h4>"+
    "<div class='wdl'>"+b(m.ph,esc(m.home)+' win',true)+b(m.pd,'Draw',true)+b(m.pa,esc(m.away)+' win',true)+"</div>"+
    "<div class='readline'>"+esc(matchReadLine(m))+"</div>"+
    "<div class='kv'><div class='k'>Most likely exact score</div><div class='v'>"+m.ms+"</div>"+
    "<div class='k'>Expanded model (expected)</div><div class='v'>"+m.exp+"</div>"+
    "<div class='k'>Expected goals</div><div class='v'>"+m.xgh.toFixed(2)+" – "+m.xga.toFixed(2)+"</div>"+
    "<div class='k'>Both teams to score</div><div class='v'>"+pct(m.btts)+"</div>"+
    "<div class='k'>Over 2.5 goals</div><div class='v'>"+pct(m.over)+"</div></div></div>"+
    "<div class='sec'><h4>Rest &amp; travel (since previous match)</h4>"+
    "<div class='kv'><div class='k'>"+esc(m.home)+"</div><div class='v'>"+fmtRT(m.home_rest,m.home_dist)+"</div>"+
    "<div class='k'>"+esc(m.away)+"</div><div class='v'>"+fmtRT(m.away_rest,m.away_dist)+"</div></div>"+
    "<div class='muted' style='margin-top:6px'>Short rest (&lt;96 h) and long travel each shave a little off "+
    "a team's expected goals — already baked into the odds below. Group games are well spaced, so the effect "+
    "is small here; it bites harder in quick knockout turnarounds.</div></div>"+
    motivationCard(m)+
    "<div class='sec'><h4>Likely scorelines (expanded model)</h4><div class='chips'>"+top+"</div>"+
    matchAnalysis(m)+"</div>";
}

function why(t){return "<div class='why'><span>Why this step</span>"+t+"</div>";}
function tabMath(m){
  var x=m.math, i=x.inputs, s=x.supremacy, ba=x.base, w=x.weather, f=x.final, o=x.outcomes;
  var adv=(i.adv_home>0?(esc(m.home)+" +"+i.adv_home+" host"):"")+(i.adv_away>0?(esc(m.away)+" +"+i.adv_away+" host"):"");
  var h="<div class='note'>This is the exact arithmetic behind the headline "+m.ms+
    " prediction — every number below is from this specific match, and each step says <i>why</i> it is done.</div>";
  // Step 1
  h+="<div class='sec'><div class='step'>Step 1 · Team strength (Elo)</div><h4>Adjusted ratings</h4>"+
    "<div class='kv'><div class='k'>"+esc(m.home)+" Elo</div><div class='v'>"+i.elo_home+(i.adv_home>0?" + "+i.adv_home+" (host)":"")+" = "+i.elo_home_adj+"</div>"+
    "<div class='k'>"+esc(m.away)+" Elo</div><div class='v'>"+i.elo_away+(i.adv_away>0?" + "+i.adv_away+" (host)":"")+" = "+i.elo_away_adj+"</div></div>"+
    why("Elo condenses a team's entire results history into one number: you gain points for beating strong "+
    "teams and lose them for losing to weak ones, scaled by the margin. It is the single most predictive "+
    "rating for football and the cleanest starting point for a match model. The host gets +"+(i.adv_home||i.adv_away||80)+
    " Elo because home advantage — crowd, familiarity, no travel — is empirically worth roughly that much.")+"</div>";
  // Step 2
  h+="<div class='sec'><div class='step'>Step 2 · Rating gap → goal supremacy</div>"+
    "<div class='eq'>dr = "+i.elo_home_adj+" − "+i.elo_away_adj+" = <b>"+s.dr+"</b><br>"+
    "supremacy = "+s.elo_to_goals+" × dr ÷ 400 = <b>"+s.supremacy+"</b> goals</div>"+
    why("Elo natively predicts <i>win probability</i>, but we want a <i>scoreline</i>. So we convert the rating "+
    "gap into an expected goal margin (\"supremacy\"). The divisor 400 is the standard Elo scale, and the "+
    "multiplier "+s.elo_to_goals+" is calibrated so a 400-point edge ≈ "+s.elo_to_goals+" goals — which reproduces "+
    "historically realistic margins between mismatched sides.")+"</div>";
  // Step 3
  h+="<div class='sec'><div class='step'>Step 3 · Split into expected goals</div>"+
    "<div class='eq'>total goals T = "+ba.base_total+(ba.goal_spread?(" + "+ba.goal_spread+"·|s|"):"")+" = "+ba.total+"<br>"+
    "λ("+esc(m.home)+") = (T + s) ÷ 2 = <b>"+ba.lh_base+"</b><br>"+
    "λ("+esc(m.away)+") = (T − s) ÷ 2 = <b>"+ba.la_base+"</b></div>"+
    why("Each team needs its own scoring rate λ. We start from a baseline total (≈"+ba.base_total+" goals, the "+
    "long-run average for this kind of match) and tilt it by the supremacy: the stronger side's λ goes up, the "+
    "weaker side's down. The floor of "+ba.attack_floor+" stops a heavy underdog's rate hitting zero — even "+
    "minnows score sometimes.")+"</div>";
  // Step 4
  var fr="";
  x.factors.forEach(function(ff){if(ff.value!==1){fr+="<div class='k'>"+ff.name+" ("+esc(ff.side)+")</div><div class='v'>×"+ff.value+"</div>";}});
  var rt="";
  if(m.home_rest!=null&&m.home_rest!=='')rt+=esc(m.home)+": "+m.home_rest+" h rest, "+Math.round(m.home_dist)+" km. ";
  if(m.away_rest!=null&&m.away_rest!=='')rt+=esc(m.away)+": "+m.away_rest+" h rest, "+Math.round(m.away_dist)+" km.";
  h+="<div class='sec'><div class='step'>Step 4 · Context adjustments</div>"+
    "<div class='kv'>"+(fr||"<div class='k'>Injury / travel / weather</div><div class='v'>all neutral (×1.00)</div>")+
    "<div class='k'>Venue heat tempo</div><div class='v'>1 − "+w.heat_tempo+"×"+(w.heat_index)+" = ×"+w.tempo+"</div>"+
    (w.altitude_side?("<div class='k'>Altitude penalty</div><div class='v'>"+esc(w.altitude_side)+" ×"+(1-w.altitude_penalty).toFixed(2)+" ("+w.altitude_m+" m)</div>"):"")+
    "<div class='k'>Final λ "+esc(m.home)+"</div><div class='v'><b>"+f.lh+"</b></div>"+
    "<div class='k'>Final λ "+esc(m.away)+"</div><div class='v'><b>"+f.la+"</b></div></div>"+
    why("Pure ratings ignore match context. Injuries cut a side's attack; heat and humidity slow the game's "+
    "tempo (fewer goals); high altitude tires the non-acclimatised visitor; and fatigue from short rest or long "+
    "travel dulls a team. "+(rt?("Here — "+rt+" "):"")+"Each is a multiplier on λ, so effects compound "+
    "proportionally rather than adding raw goals.")+"</div>";
  // Step 5 Poisson
  function prow(name,arr){var c="";for(var k=0;k<arr.length;k++)c+="<td>"+(arr[k]*100).toFixed(1)+"</td>";
    return "<tr><td class='l'>"+esc(name)+"</td>"+c+"</tr>";}
  var hd="";for(var k=0;k<7;k++)hd+="<th>"+k+"</th>";
  h+="<div class='sec'><div class='step'>Step 5 · Goal distributions (Poisson)</div>"+
    "<div class='eq'>P(k goals) = e<sup>−λ</sup> · λ<sup>k</sup> ÷ k!</div>"+
    "<table class='t'><tr><th class='l'>P(goals) %</th>"+hd+"</tr>"+
    prow(m.home,x.poisson.home)+prow(m.away,x.poisson.away)+"</table>"+
    why("Goals are rare, roughly independent events spread across 90 minutes — exactly the situation the "+
    "Poisson distribution describes, and it fits football goal counts well in practice. Feeding each team's λ "+
    "into the formula gives the probability of them scoring 0, 1, 2, … goals.")+"</div>";
  // Step 6 grid
  var mx=0;x.grid6.forEach(function(r){r.forEach(function(v){if(v>mx)mx=v;});});
  var gr="<tr><th></th>";for(var b=0;b<6;b++)gr+="<th>"+b+"</th>";gr+="</tr>";
  for(var a=0;a<6;a++){gr+="<tr><th>"+a+"</th>";
    for(var b2=0;b2<6;b2++){var v=x.grid6[a][b2];var al=(v/mx);
      gr+="<td style='background:rgba(40,209,124,"+al.toFixed(2)+")'>"+(v*100).toFixed(0)+"</td>";}
    gr+="</tr>";}
  h+="<div class='sec'><div class='step'>Step 6 · Joint score grid + Dixon-Coles</div>"+
    "<table class='grid'>"+gr+"</table>"+
    why("Multiplying the two teams' goal distributions gives every exact scoreline's probability ("+esc(m.home)+
    " down the rows, "+esc(m.away)+" across the columns). Plain independent Poisson slightly under-counts the "+
    "0-0, 1-0, 0-1 and 1-1 results, so we apply the Dixon-Coles (1997) correction (ρ="+x.dixon_coles.rho+") that "+
    "nudges exactly those four cells — the fix that makes correct-score and draw probabilities accurate. The "+
    "brightest cell is the single most-likely score.")+"</div>";
  // Step 7 outcomes
  h+="<div class='sec'><div class='step'>Step 7 · From grid to predictions</div>"+
    "<div class='kv'><div class='k'>Win = Σ cells below diagonal</div><div class='v'>"+pct(o.p_home)+"</div>"+
    "<div class='k'>Draw = Σ diagonal</div><div class='v'>"+pct(o.p_draw)+"</div>"+
    "<div class='k'>Loss = Σ cells above diagonal</div><div class='v'>"+pct(o.p_away)+"</div>"+
    "<div class='k'>Most likely (brightest cell)</div><div class='v'><b>"+x.scores.most_likely+"</b></div>"+
    "<div class='k'>Expanded (rounded λ)</div><div class='v'><b>"+x.scores.expected+"</b></div></div>"+
    why("Every prediction is just a sum over the grid. Add the cells where the home team scores more for the "+
    "win probability, the diagonal for the draw, the rest for the loss. The single brightest cell is the "+
    "\"most likely\" exact score; rounding the two expected-goal values gives the livelier \"expanded\" "+
    "scoreline. Inverting these probabilities (1 ÷ p) gives the model-implied odds on the Odds tab.")+"</div>";
  return h;
}

function tabOdds(m){
  function row(key,lab,dec,p){return "<tr><td class='l'>"+lab+"</td><td>"+(dec?dec.toFixed(2):'—')+
    "</td><td>"+pct(p)+"</td><td id='lr_"+key+"' class='live'>…</td><td id='le_"+key+"' class='muted'>…</td></tr>";}
  // top-5 correct scores: model prob, model fair odds, and a live cell per score
  var cs=m.cs5.map(function(c){var id="lo_"+c[0].replace('-','_');
    return "<tr><td class='l'><b>"+c[0]+"</b></td><td>"+pct(c[1])+"</td><td>"+
      (c[2]?c[2].toFixed(2):'—')+"</td><td id='"+id+"' class='live'>…</td></tr>";}).join('');
  var t="<div class='sec'><h4>Correct score — top 5 (model vs live bookmakers)</h4>"+
    "<table class='t'><tr><th class='l'>Score</th><th>Model prob</th><th>Model odds</th>"+
    "<th>Live odds</th></tr>"+cs+"</table>"+
    "<div id='live_status' class='muted' style='margin-top:9px'>Loading…</div></div>";
  t+="<div class='sec'><h4>Match result — model vs live bookmakers (1X2)</h4>"+
    "<table class='t'><tr><th class='l'>Outcome</th><th>Model odds</th><th>Model implied</th><th>Live best</th><th>Edge</th></tr>"+
    row('home',esc(m.home)+" win",m.math.odds.home,m.ph)+row('draw',"Draw",m.math.odds.draw,m.pd)+
    row('away',esc(m.away)+" win",m.math.odds.away,m.pa)+"</table>"+
    "<div class='muted' style='margin-top:8px'>Edge compares the best returned live decimal price with this model's fair price; positive means the live price is longer than the model-implied fair odds.</div></div>";
  t+="<div class='sec'><h4>To win the tournament</h4><table class='t'>"+
    "<tr><th class='l'>Team</th><th>Decimal</th><th>Implied</th></tr>"+
    "<tr><td class='l'>"+esc(m.home)+"</td><td>"+(m.th?m.th.toFixed(1):'—')+"</td><td>"+pct(m.thp)+"</td></tr>"+
    "<tr><td class='l'>"+esc(m.away)+"</td><td>"+(m.ta?m.ta.toFixed(1):'—')+"</td><td>"+pct(m.tap)+"</td></tr></table>"+
    "<div class='note' style='margin-top:10px'>“Model” columns are <b>fair</b> odds (1 ÷ probability) with no "+
    "margin. The <b>Live odds</b> column pulls real correct-score prices from a configured bookmaker feed and "+
    "refreshes on every load — see the note under the table to switch it on.</div></div>";
  setTimeout(function(){loadLiveOdds(m);},0);   // fetch live prices on every render
  return t;
}

function formColumn(team){
  var fm=DATA.form[team];
  var h="<div class='col'><h5>"+esc(team)+"</h5>";
  if(!fm||!fm.length)return h+"<span class='muted'>no recent data</span></div>";
  fm.forEach(function(g){h+="<div class='frow'><span class='fres "+g.res+"'>"+g.res+"</span>"+
    "<span class='fdate'>"+esc(g.date)+"</span><span class='fopp'>vs "+esc(g.opp)+"</span>"+
    "<span class='fsc'>"+g.gf+"-"+g.ga+"</span></div>";});
  return h+"</div>";
}
function tabHistory(m){
  if(!DATA.history_available){
    return "<div class='sec'><h4>Head-to-head &amp; form</h4><div class='note'>No history loaded yet. "+
      "Run <b>python build_history.py</b> on a machine with internet to download ~47k international "+
      "results and generate <b>data/history.json</b>; reload this page and real head-to-head records "+
      "and recent form appear here. (No results are ever invented.)</div></div>";
  }
  var hh=DATA.h2h[m.home+"|"+m.away]||{home_wins:0,draws:0,away_wins:0,total:0,meetings:[]};
  var mt=hh.meetings.map(function(g){var r=g.gh>g.ga?'W':(g.gh<g.ga?'L':'D');
    return "<div class='frow'><span class='fres "+r+"'>"+(g.gh)+"-"+(g.ga)+"</span>"+
      "<span class='fdate'>"+esc(g.date)+"</span><span class='fopp'>"+esc(m.home)+" v "+esc(m.away)+"</span></div>";}).join('')||
    "<span class='muted'>No previous meetings on record.</span>";
  return "<div class='sec'><h4>Head-to-head — "+esc(m.home)+" vs "+esc(m.away)+"</h4>"+
    "<div class='chips'><span class='chip'><b>"+hh.home_wins+"</b> "+esc(m.home)+" wins</span>"+
    "<span class='chip'><b>"+hh.draws+"</b> draws</span>"+
    "<span class='chip'><b>"+hh.away_wins+"</b> "+esc(m.away)+" wins</span>"+
    "<span class='chip'>"+hh.total+" total meetings</span></div>"+
    "<div style='margin-top:10px'>"+mt+"</div></div>"+
    "<div class='sec'><h4>Recent form (most recent first)</h4>"+
    "<div class='cols2'>"+formColumn(m.home)+formColumn(m.away)+"</div></div>";
}

function squadColumn(team){
  var sq=DATA.squads[team];
  if(!sq||!sq.length)
    return "<div class='col'><h5>"+esc(team)+"</h5><span class='muted'>not in squads.csv</span></div>";
  var order=['Goalkeeper','Defender','Midfielder','Forward'];
  var byPos={};sq.forEach(function(p){(byPos[p.position]=byPos[p.position]||[]).push(p);});
  var h="<div class='col'><h5>"+esc(team)+" <span class='ct'>("+sq.length+" players)</span></h5>";
  order.forEach(function(pos){ if(byPos[pos]){
    h+="<div class='poslab'>"+pos+"s</div><ul class='plist'>";
    byPos[pos].forEach(function(p){h+="<li class='prow'><span class='pn'>"+esc(p.player)+
      "</span><span class='pc'>"+esc(p.club)+"</span></li>";});
    h+="</ul>";
  }});
  return h+"</div>";
}
function tabSquads(m){
  if(!DATA.squads_available){
    return "<div class='sec'><h4>Squads</h4><div class='note'>No squad data loaded. Run "+
      "<b>python build_squads.py</b> (or add rows to <b>data/squads.csv</b>) and reload — each side's "+
      "squad appears here as a two-column list by position. Left empty on purpose: squad lists are real "+
      "data, not invented.</div></div>";
  }
  return "<div class='sec'><div class='cols2'>"+squadColumn(m.home)+squadColumn(m.away)+"</div></div>";
}

renderAll();
"""


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    os.makedirs(OUT, exist_ok=True)
    P = model.load_predictor()
    data = build_data(P, n_sims)
    profile_btns = "".join(
      f'<button id="pf_{key}" '
      f'onclick="setProfile(\'{key}\')">{data["profile_meta"][key]["label"]}</button>'
      for key in data["profile_order"]
    )
    html = (PAGE.replace("__CSS__", CSS).replace("__JS__", JS)
            .replace("__PROFILE_BTNS__", profile_btns)
            .replace("__DATA__", json.dumps(data)))
    with open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(os.path.join(OUT, "dashboard.html")) / 1024
    print(f"Wrote dashboard.html ({size:.0f} KB), predictions_groupstage.csv, "
          f"tournament_odds.csv — {len(data['profiles']['default']['matches'])} matches, {n_sims:,} sims")
    print(f"History loaded: {data['history_available']} · Squads loaded: {data['squads_available']}")


if __name__ == "__main__":
    main()
