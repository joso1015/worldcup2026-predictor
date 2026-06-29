"""
World Cup 2026 Correct-Score Predictor -- Streamlit front end.

Run locally:   streamlit run app.py
Deploy free:   push this folder to GitHub, then connect it at share.streamlit.io
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import accuracy
import model

try:
    from train_model import MLModel
except Exception:
    MLModel = None

DATA_AS_OF = "Elo ratings as of 2026-05-26 (eloratings.net) · FIFA Final Draw"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

st.set_page_config(page_title="World Cup 2026 Score Predictor",
                   page_icon="⚽", layout="wide")


@st.cache_resource
def get_predictor():
    return model.load_predictor()


@st.cache_data(show_spinner="Running tournament simulations…")
def run_sim(param_tuple, n_sims, ml_active):
    p = model.Params(*param_tuple)
    P = get_predictor()
    P.model_ml = MLModel.load() if (ml_active and MLModel) else None
    return P.simulate_tournament(n_sims=n_sims, p=p)


@st.cache_data(show_spinner="Fitting engine settings to entered results…")
def run_accuracy_summary(param_tuple, ml_active):
    p = model.Params(*param_tuple)
    P = get_predictor()
    P.model_ml = MLModel.load() if (ml_active and MLModel) else None
    current = accuracy.evaluate_params(P, p)
    saved = accuracy.load_summary(OUT)
    if saved:
        saved["current"] = current
        saved["played"] = current["metrics"]["played"]
        return saved
    return accuracy.summary_for_params(P, p, include_sweep=True,
                                       include_ml=ml_active)


@st.cache_data
def load_history():
    p = os.path.join(DATA, "history.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"form": {}, "h2h": {}}


@st.cache_data
def load_squads():
    p = os.path.join(DATA, "squads.csv")
    squads = {}
    if os.path.exists(p):
        df = pd.read_csv(p).fillna("")
        for r in df.itertuples(index=False):
            squads.setdefault(r.team, []).append(
                dict(player=r.player, position=r.position, club=r.club))
    return {k: v for k, v in squads.items() if v}


P = get_predictor()
ml_available = bool(MLModel and MLModel.load())
HISTORY = load_history()
SQUADS = load_squads()

# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("⚽ Model controls")
st.sidebar.caption(DATA_AS_OF)

st.sidebar.subheader("Engine parameters")
home_adv = st.sidebar.slider("Host home advantage (Elo)", 0, 150, 80, 5,
                             help="Elo boost for Mexico / USA / Canada when playing in their own country.")
elo_to_goals = st.sidebar.slider("Elo → goal supremacy (per 400 Elo)",
                                 1.0, 2.5, 1.65, 0.05,
                                 help="How strongly rating gaps become expected goal margins.")
base_total = st.sidebar.slider("Baseline goals per match", 2.0, 3.2, 2.7, 0.05,
                               help="Expected total goals in an even match.")
rho = st.sidebar.slider("Dixon-Coles low-score correction (ρ)",
                        -0.20, 0.05, -0.04, 0.01,
                        help="Adjusts the 0-0, 1-0, 0-1 and 1-1 cells in the score grid.")

with st.sidebar.expander("What do these controls do?"):
    st.markdown("""
    **Host home advantage** adds Elo points to a host nation at home.

    **Elo → goal supremacy** turns rating gaps into goal margin.

    **Baseline goals** sets the match's starting goal total.

    **Dixon-Coles ρ** tweaks the lowest scorelines after the Poisson grid is built.
    """)

st.sidebar.subheader("ML calibration")
if ml_available:
    ml_weight = st.sidebar.slider("ML blend weight", 0.0, 1.0, 0.0, 0.05,
                                  help="0 = pure statistical, 1 = pure ML.")
else:
    ml_weight = 0.0
    st.sidebar.info("ML model not trained yet. Run `python train_model.py`.")

st.sidebar.subheader("Simulation")
n_sims = st.sidebar.select_slider("Monte Carlo runs",
                                  options=[1000, 2000, 5000, 10000], value=5000)

params = model.Params(home_adv=home_adv, elo_to_goals=elo_to_goals,
                      base_total=base_total, rho=rho, ml_weight=ml_weight)
P.model_ml = MLModel.load() if (ml_weight > 0 and MLModel) else None
ptuple = (params.home_adv, params.elo_to_goals, params.base_total, params.rho,
          params.max_goals, params.min_lambda, params.max_lambda,
          params.heat_tempo, params.altitude_penalty, params.ml_weight)

# ------------------------------------------------------------------ header ---
st.title("World Cup 2026 — Correct-Score Predictor")
st.caption("Elo → expected goals · Dixon-Coles · injury / travel / weather "
           "adjustments · Monte Carlo tournament simulation")

tab_summary, tab_groups, tab_ko, tab_odds, tab_explore = st.tabs(
    ["📈 Summary", "📋 Group stage", "🏆 Knockouts", "📊 Tournament outlook", "🔬 Match explorer"])

# --------------------------------------------------------------- summary -----
with tab_summary:
    summary = run_accuracy_summary(ptuple, ml_weight > 0)
    current = summary["current"]
    metrics = current["metrics"]
    played = metrics["played"]

    st.subheader("Current results accuracy")
    st.caption("Fitted to the real results currently entered in data/results.csv.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches played", played)
    c2.metric("Most Likely score", f"{metrics['ml_score_correct']}/{played}")
    c3.metric("Most Likely result", f"{metrics['ml_result_correct']}/{played}")
    c4.metric("Expanded score", f"{metrics['exp_score_correct']}/{played}")
    c5.metric("Expanded result", f"{metrics['exp_result_correct']}/{played}")

    st.markdown("**Best-fitting engine settings so far**")
    winners = summary.get("winners", {})
    labels = {
        "most_likely_score": "Most Likely: exact scores",
        "most_likely_result": "Most Likely: results",
        "expanded_score": "Expanded: exact scores",
        "expanded_result": "Expanded: results",
    }
    rows = []
    for key, label in labels.items():
        row = winners.get(key)
        if not row:
            continue
        rows.append(dict(
            Target=label,
            **{"ML score": f"{int(row['ml_score_correct'])}/{int(row['played'])}",
               "ML result": f"{int(row['ml_result_correct'])}/{int(row['played'])}",
               "Expanded score": f"{int(row['exp_score_correct'])}/{int(row['played'])}",
               "Expanded result": f"{int(row['exp_result_correct'])}/{int(row['played'])}",
               "Home adv": row["home_adv"],
               "Elo scale": row["elo_to_goals"],
               "Base goals": row["base_total"],
               "rho": row["rho"],
               "Heat": row["heat_tempo"],
               "Altitude": row["altitude_penalty"],
               "ML weight": row["ml_weight"]}
        ))
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("**Played match checks**")
    match_rows = pd.DataFrame(current["matches"])
    if not match_rows.empty:
        show_matches = match_rows[["match_no", "home", "away", "actual",
                                   "most_likely", "most_likely_score_correct",
                                   "most_likely_result_correct", "expanded",
                                   "expanded_score_correct", "expanded_result_correct"]]
        st.dataframe(show_matches.rename(columns={
            "match_no": "No", "home": "Home", "away": "Away",
            "actual": "Actual", "most_likely": "Most Likely",
            "most_likely_score_correct": "ML score hit",
            "most_likely_result_correct": "ML result hit",
            "expanded": "Expanded",
            "expanded_score_correct": "Exp score hit",
            "expanded_result_correct": "Exp result hit",
        }), width='stretch', hide_index=True, height=400)


# ----------------------------------------------------------- group stage -----
def render_match_detail(P, home, away, params, venue_country="", venue="",
                        overrides=None, is_ko=False):
    """Full match breakdown with Overview, Model Math, Odds, History, Squads."""
    pe = model.expanded_params(params)
    base = P.predict_match(home, away, params, venue_country=venue_country,
                           venue=venue, overrides=overrides)
    exp = P.predict_match(home, away, pe, venue_country=venue_country,
                          venue=venue, overrides=overrides)
    math_detail = P.explain_match(home, away, params,
                                  venue_country=venue_country, venue=venue,
                                  overrides=overrides)

    # --- Header ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Most likely score", base["most_likely_score"])
    m2.metric("Expanded score", exp["expected_scoreline"])
    m3.metric(f"{home} xG", f"{base['lambda_home']:.2f}")
    m4.metric(f"{away} xG", f"{base['lambda_away']:.2f}")

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(f"{home} win", f"{base['p_home_win']*100:.0f}%")
    p2.metric("Draw", f"{base['p_draw']*100:.0f}%")
    p3.metric(f"{away} win", f"{base['p_away_win']*100:.0f}%")
    p4.metric("BTTS", f"{exp['btts']*100:.0f}%")
    p5.metric("Over 2.5", f"{exp['over25']*100:.0f}%")

    if is_ko:
        ko = P.predict_knockout_match(home, away, params,
                                       venue_country=venue_country,
                                       venue=venue, overrides=overrides)
        st.info(f"**Advancement:** {home} {ko['p_home_adv']*100:.0f}% — "
                f"{away} {ko['p_away_adv']*100:.0f}% "
                f"(90-min: {home} {ko['p_home_90']*100:.0f}% · Draw "
                f"{ko['p_draw_90']*100:.0f}% · {away} {ko['p_away_90']*100:.0f}%)")

    # --- Sub-tabs ---
    dt1, dt2, dt3, dt4, dt5 = st.tabs(
        ["Overview", "Model Math", "Score Grid", "History", "Squads"])

    with dt1:
        st.markdown("#### Predicted result")
        st.write(f"Most likely exact score: **{base['most_likely_score']}**")
        st.write(f"Expanded (expected): **{exp['expected_scoreline']}**")
        st.write(f"Expected goals: {home} {base['lambda_home']:.2f} — "
                 f"{away} {base['lambda_away']:.2f}")
        st.write(f"Both teams to score: {exp['btts']*100:.0f}%")
        st.write(f"Over 2.5 goals: {exp['over25']*100:.0f}%")

        st.markdown("#### Top correct scores")
        for score, prob in base["top_scores"][:8]:
            odds = 1 / prob if prob > 1e-6 else 0
            st.write(f"  {score} — {prob*100:.1f}% (fair odds: {odds:.2f})")

        st.markdown("#### Fair model odds")
        odds_df = pd.DataFrame([
            {"Outcome": f"{home} win", "Probability": f"{base['p_home_win']*100:.1f}%",
             "Fair odds": f"{1/base['p_home_win']:.2f}" if base['p_home_win'] > 1e-6 else "—"},
            {"Outcome": "Draw", "Probability": f"{base['p_draw']*100:.1f}%",
             "Fair odds": f"{1/base['p_draw']:.2f}" if base['p_draw'] > 1e-6 else "—"},
            {"Outcome": f"{away} win", "Probability": f"{base['p_away_win']*100:.1f}%",
             "Fair odds": f"{1/base['p_away_win']:.2f}" if base['p_away_win'] > 1e-6 else "—"},
        ])
        st.dataframe(odds_df, width='stretch', hide_index=True)

        if is_ko:
            ko = P.predict_knockout_match(home, away, params,
                                          venue_country=venue_country,
                                          venue=venue, overrides=overrides)
            ko_odds_df = pd.DataFrame([
                {"Outcome": f"{home} advances", "Probability": f"{ko['p_home_adv']*100:.1f}%",
                 "Fair odds": f"{1/ko['p_home_adv']:.2f}" if ko['p_home_adv'] > 1e-6 else "—"},
                {"Outcome": f"{away} advances", "Probability": f"{ko['p_away_adv']*100:.1f}%",
                 "Fair odds": f"{1/ko['p_away_adv']:.2f}" if ko['p_away_adv'] > 1e-6 else "—"},
            ])
            st.markdown("**Advancement odds (KO)**")
            st.dataframe(ko_odds_df, width='stretch', hide_index=True)

    with dt2:
        x = math_detail
        i = x["inputs"]
        s = x["supremacy"]
        ba = x["base"]
        w = x["weather"]
        f = x["final"]
        o = x["outcomes"]

        st.markdown("#### Step 1 · Team strength (Elo)")
        st.write(f"Elo {home}: {i['elo_home']}" +
                 (f" (+ {i['adv_home']} host)" if i.get('adv_home', 0) > 0 else "") +
                 f" → {i['elo_home_adj']}")
        st.write(f"Elo {away}: {i['elo_away']}" +
                 (f" (+ {i['adv_away']} host)" if i.get('adv_away', 0) > 0 else "") +
                 f" → {i['elo_away_adj']}")

        st.markdown("#### Step 2 · Rating gap → goal supremacy")
        st.write(f"dr = {i['elo_home_adj']} − {i['elo_away_adj']} = **{s['dr']}**")
        st.write(f"supremacy = {s['elo_to_goals']} × {s['dr']} ÷ 400 = **{s['supremacy']}** goals")

        st.markdown("#### Step 3 · Split into expected goals")
        st.write(f"Total goals T = {ba['base_total']}" +
                 (f" + {ba['goal_spread']}·|s|" if ba.get('goal_spread', 0) else "") +
                 f" = {ba['total']}")
        st.write(f"λ({home}) = (T + s) ÷ 2 = **{ba['lh_base']}**")
        st.write(f"λ({away}) = (T − s) ÷ 2 = **{ba['la_base']}**")

        st.markdown("#### Step 4 · Context adjustments")
        for fac in x.get("factors", []):
            if fac["value"] != 1.0:
                st.write(f"  {fac['name']} ({fac['side']}): ×{fac['value']}")
        if w.get("heat_index") is not None:
            st.write(f"  Venue heat tempo: ×{w['tempo']}")
        if w.get("altitude_side"):
            st.write(f"  Altitude penalty ({w['altitude_side']}): "
                     f"×{1-w['altitude_penalty']:.2f} ({w['altitude_m']} m)")
        st.write(f"  Final λ {home}: **{f['lh']}**")
        st.write(f"  Final λ {away}: **{f['la']}**")

        st.markdown("#### Step 5 · Goal distributions (Poisson)")
        poisson_df = pd.DataFrame({
            "Goals": list(range(len(x["poisson"]["home"]))),
            f"{home} %": [p * 100 for p in x["poisson"]["home"]],
            f"{away} %": [p * 100 for p in x["poisson"]["away"]],
        })
        st.dataframe(poisson_df, width='stretch', hide_index=True)

        st.markdown("#### Step 6 · Joint score grid + Dixon-Coles")
        st.write(f"Dixon-Coles ρ = {x['dixon_coles']['rho']}")
        grid = x.get("grid6", [])
        if grid:
            grid_df = pd.DataFrame(grid)
            grid_df.columns = [f"{j}" for j in range(len(grid_df.columns))]
            grid_df.index = [f"{home} {i}" for i in range(len(grid))]
            st.write(f"Rows = {home} goals, Columns = {away} goals "
                     f"(values ×100)")
            st.dataframe(grid_df.style.format("{:.1f}"), width='stretch')

        st.markdown("#### Step 7 · From grid to predictions")
        st.write(f"Win ({home}): {o['p_home']*100:.1f}%")
        st.write(f"Draw: {o['p_draw']*100:.1f}%")
        st.write(f"Loss ({away}): {o['p_away']*100:.1f}%")
        st.write(f"Most likely score: **{x['scores']['most_likely']}**")
        st.write(f"Expanded (rounded λ): **{x['scores']['expected']}**")

    with dt3:
        cap = 8
        mat = base["matrix"][:cap, :cap]
        hm = px.imshow(mat, labels=dict(x=f"{away} goals", y=f"{home} goals",
                       color="Prob."), x=list(range(cap)), y=list(range(cap)),
                       color_continuous_scale="Greens", text_auto=".3f")
        hm.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(hm, width='stretch')
        st.markdown(f"**Most likely scorelines:** " +
                    " · ".join(f"{s} ({pr*100:.0f}%)" for s, pr in base["top_scores"]))

    with dt4:
        form = HISTORY.get("form", {})
        h2h = HISTORY.get("h2h", {})
        h2h_key = f"{home}|{away}"
        hh = h2h.get(h2h_key, {})
        if hh:
            st.markdown(f"#### Head-to-head — {home} vs {away}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Home wins", hh.get("home_wins", 0))
            c2.metric("Draws", hh.get("draws", 0))
            c3.metric("Away wins", hh.get("away_wins", 0))
            c4.metric("Total meetings", hh.get("total", 0))
            meetings = hh.get("meetings", [])
            if meetings:
                st.markdown("Recent meetings:")
                for mt in meetings[:8]:
                    st.write(f"  {mt['date']}: {home} {mt['gh']} - {mt['ga']} {away}")
        else:
            st.info(f"No head-to-head data for {home} vs {away}")

        st.markdown(f"#### Recent form")
        c1, c2 = st.columns(2)
        for col, team in [(c1, home), (c2, away)]:
            tf = form.get(team, [])
            with col:
                st.markdown(f"**{team}** (last {len(tf)})")
                if tf:
                    for g in tf[:8]:
                        res = g.get("res", "?")
                        emoji = "🟢" if res == "W" else ("🟡" if res == "D" else "🔴")
                        st.write(f"  {emoji} {g['date']} vs {g['opp']}: {g['gf']}-{g['ga']} ({res})")
                else:
                    st.write("  No recent data")

    with dt5:
        for col, team in [(st, home), (st, away)]:
            sq = SQUADS.get(team, [])
            st.markdown(f"#### {team} ({len(sq)} players)")
            if not sq:
                st.info(f"No squad data for {team}")
                continue
            by_pos = {}
            for pl in sq:
                by_pos.setdefault(pl["position"], []).append(pl)
            for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                players = by_pos.get(pos, [])
                if players:
                    st.markdown(f"**{pos}s** ({len(players)})")
                    for p in players:
                        st.write(f"  {p['player']} — {p['club']}")


with tab_groups:
    gp = P.predict_all_group_matches(params)
    c1, c2 = st.columns([1, 2])
    groups = sorted(gp.group.unique())
    sel = c1.multiselect("Filter groups", groups, default=groups)
    team_q = c2.text_input("Search a team", "")
    view = gp[gp.group.isin(sel)]
    if team_q:
        m = view.home.str.contains(team_q, case=False) | \
            view.away.str.contains(team_q, case=False)
        view = view[m]

    show = view.assign(
        Match=view.home + "  vs  " + view.away,
        **{"Most likely": view.predicted_score,
           "Expanded": view.expanded_score,
           "Win %": (view.p_home_win * 100).round(0),
           "Draw %": (view.p_draw * 100).round(0),
           "Loss %": (view.p_away_win * 100).round(0),
           "BTTS %": (view.btts * 100).round(0),
           "O2.5 %": (view.over25 * 100).round(0),
           "xG": view.xg_home.round(2).astype(str) + "–" + view.xg_away.round(2).astype(str),
           "Temp": view.temp_c.astype(str) + "°C" +
                   view.indoor.map({1: " (in)", 0: ""}),
           "Venue": view.venue + ", " + view.city}
    )[["group", "match_day", "date", "kickoff", "Match", "Most likely",
       "Expanded", "Win %", "Draw %", "Loss %", "BTTS %", "O2.5 %", "xG",
       "Temp", "Venue"]].rename(columns={"group": "Grp", "match_day": "MD",
                                          "date": "Date", "kickoff": "KO"})
    st.dataframe(show, width='stretch', hide_index=True, height=400)
    st.caption("“Most likely” = single most-probable exact score. “Expanded” = "
               "the livelier model’s expected scoreline.")

    # --- Match detail ---
    st.markdown("---")
    st.subheader("📋 Match detail")
    match_labels = [f"{r.home} vs {r.away} ({r.group} MD{r.match_day})"
                    for r in view.itertuples(index=False)]
    match_idx = st.selectbox("Select a match for full breakdown", match_labels)
    if match_idx is not None and match_labels:
        sel_row = view.iloc[match_labels.index(match_idx)]
        h, a = sel_row.home, sel_row.away
        render_match_detail(P, h, a, params,
                            venue_country=sel_row.country,
                            venue=sel_row.venue,
                            overrides=P.fixture_overrides(sel_row))


# ------------------------------------------------------------- knockouts -----
with tab_ko:
    st.subheader("Knockout bracket — full tournament prediction")
    st.caption("Real R32 pairings with played results locked. "
               "Draws → extra time → penalties modeled.")
    ko = P.resolve_knockout_bracket(params)
    champion = ko[-1]["games"][0]["winner"] if ko else "TBD"
    st.success(f"🏆 Projected champion: **{champion}**")

    # Build flat list of KO matches for detail selection
    ko_match_list = []
    for rnd in ko[:-1]:
        for g in rnd.get("games", []):
            if g.get("home") and g.get("away"):
                ko_match_list.append((rnd["label"], g))

    for rnd in ko[:-1]:
        with st.expander(rnd["label"], expanded=(rnd["label"] in
                         ("Semi-finals", "Final", "Round of 32"))):
            for g in rnd["games"]:
                note = g.get("note", "")
                if g.get("played"):
                    st.markdown(
                        f"**{g['home']}**  {g['score']}  **{g['away']}**  *(FT)* → "
                        f"**{g['winner']}**")
                else:
                    win_pct = max(g.get("p_home", 0), g.get("p_away", 0)) * 100
                    winner_note = f" {note}" if note else ""
                    st.markdown(
                        f"**{g['home']}**  {g['score']}  **{g['away']}**{winner_note} → "
                        f"**{g['winner']}** ({win_pct:.0f}%)")

    # --- KO Match detail ---
    st.markdown("---")
    st.subheader("📋 KO match detail")
    if ko_match_list:
        ko_labels = [f"{lbl}: {g['home']} vs {g['away']}"
                     for lbl, g in ko_match_list]
        ko_sel = st.selectbox("Select a KO match for full breakdown", ko_labels)
        if ko_sel is not None:
            lbl, g = ko_match_list[ko_labels.index(ko_sel)]
            h, a = g["home"], g["away"]
            vcountry, vname = "", ""
            for kf in P.ko_bracket:
                if kf.get("home_team") == h and kf.get("away_team") == a:
                    vcountry = kf.get("country", "")
                    vname = kf.get("venue", "")
                    break
            render_match_detail(P, h, a, params,
                                venue_country=vcountry, venue=vname,
                                is_ko=True)


# ------------------------------------------------------- tournament odds -----
with tab_odds:
    odds = run_sim(ptuple, n_sims, ml_weight > 0)
    top = odds.head(20)
    fig = px.bar(top, x="win_title", y="team", orientation="h",
                 labels={"win_title": "Title probability", "team": ""},
                 text=(top.win_title * 100).round(1).astype(str) + "%")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=560,
                      xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
    fig.update_traces(marker_color="#28d17c")
    st.plotly_chart(fig, width='stretch')

    disp = odds.copy()
    for c in ["reach_r32", "reach_r16", "reach_qf", "reach_sf",
              "reach_final", "win_title"]:
        disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
    disp = disp.rename(columns={"reach_r32": "R32", "reach_r16": "R16",
                                "reach_qf": "QF", "reach_sf": "SF",
                                "reach_final": "Final", "win_title": "Champion"})
    st.dataframe(disp[["team", "group", "elo", "R32", "R16", "QF", "SF",
                       "Final", "Champion"]], width='stretch',
                 hide_index=True, height=420)
    st.caption(f"From {n_sims:,} Monte Carlo simulations of the full bracket.")


# --------------------------------------------------------- match explorer ----
with tab_explore:
    st.subheader("Predict any match — with live injury / travel / weather")
    teams = sorted(P.teams)
    c1, c2, c3 = st.columns(3)
    home = c1.selectbox("Home / Team A", teams, index=teams.index("Spain"))
    away = c2.selectbox("Away / Team B", teams, index=teams.index("England"))
    venue_opts = ["Neutral"] + list(P.venues.venue)
    venue = c3.selectbox("Venue", venue_opts, index=0)

    st.markdown("**Adjustments** — drag to model injuries, fatigue and heat.")
    a1, a2, a3, a4 = st.columns(4)
    hi = a1.slider(f"{home} injury impact", 0.7, 1.1, 1.0, 0.01,
                   help="<1 = key players out")
    ai = a2.slider(f"{away} injury impact", 0.7, 1.1, 1.0, 0.01)
    ht = a3.slider(f"{home} travel/rest", 0.85, 1.1, 1.0, 0.01,
                   help="<1 = tired / long trip")
    at = a4.slider(f"{away} travel/rest", 0.85, 1.1, 1.0, 0.01)

    overrides = dict(home_injury=hi, away_injury=ai,
                     home_travel=ht, away_travel=at)
    vcountry = ""
    vname = "" if venue == "Neutral" else venue
    temp_txt = "—"
    if vname:
        vrow = P.venue_row.get(vname)
        if vrow:
            vcountry = vrow.country
            if int(vrow.indoor):
                temp_txt = "22°C (indoor)"
            else:
                temp_txt = f"~{int(vrow.afternoon_high_c)}°C day / {int(vrow.evening_c)}°C eve"

    render_match_detail(P, home, away, params,
                        venue_country=vcountry, venue=vname,
                        overrides=overrides, is_ko=True)
    st.caption(f"Venue temperature: {temp_txt}")

st.divider()
st.caption("Probabilistic estimates, not certainties. The pure statistical "
           "engine is the default; ML and live injury/weather feeds are optional "
           "enhancements layered on top.")