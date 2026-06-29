"""
World Cup 2026 correct-score prediction engine.

Layered model:
  1. Elo  -> expected goals (supremacy + total-goals decomposition)
  2. Dixon-Coles low-score correction -> full score-probability matrix
  3. Adjustment layer: injuries, travel/rest, weather (heat + altitude), host
  4. Optional ML calibration (blended in if a trained model is present)
  5. Monte Carlo tournament simulation -> advancement % and title odds

Pure-numpy / pandas. No scipy required (Poisson implemented directly), so the
statistical engine and the simulation run anywhere. The ML layer (scikit-learn)
is optional and only activates when model_ml is supplied with ml_weight > 0.

All tunable parameters live in DEFAULTS and can be overridden per call.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    home_adv: float = 80.0        # Elo points added to a co-host on home soil
    elo_to_goals: float = 1.60    # goal supremacy produced by a 400-Elo edge
    base_total: float = 2.60      # baseline total goals in an even match
    rho: float = -0.05            # Dixon-Coles low-score dependence
    max_goals: int = 8            # score-grid cap per team
    min_lambda: float = 0.15      # floor on expected goals
    max_lambda: float = 6.00      # ceiling on expected goals
    heat_tempo: float = 0.12      # max goal dampening from a hot venue
    altitude_penalty: float = 0.08  # away-team penalty at high-altitude venues
    ml_weight: float = 0.0        # 0 = pure statistical; 1 = pure ML
    # "Expanded" scoreline model (defaults keep the base model unchanged):
    goal_spread: float = 0.0      # extra total goals per unit of supremacy
    attack_floor: float = 0.15    # minimum expected goals (underdog still scores)


DEFAULTS = Params()

# Preset for the livelier, more realistic scoreline model: totals grow with the
# mismatch and underdogs keep a real attacking threat, so 2-1 / 3-1 / 2-2 appear
# instead of everything collapsing to 1-1 or a win-to-nil.
def expanded_params(p: Params = None) -> "Params":
    p = p or DEFAULTS
    return replace(p, base_total=max(p.base_total, 2.7),
                   goal_spread=0.45, attack_floor=0.55)


# ----------------------------------------------------------------------------
# Poisson helpers (no scipy)
# ----------------------------------------------------------------------------
def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction for the four lowest-scoring cells."""
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


# ----------------------------------------------------------------------------
# Reusable prediction math (used by the engine AND the backtester so both score
# the identical model).
# ----------------------------------------------------------------------------
def lambdas_from_elo(elo_h, elo_a, p=None, home_adv=0.0):
    """Elo gap -> (lambda_home, lambda_away) via the supremacy + total split."""
    p = p or DEFAULTS
    s = p.elo_to_goals * ((elo_h + home_adv) - elo_a) / 400.0
    total = p.base_total + p.goal_spread * abs(s)
    return max((total + s) / 2.0, p.attack_floor), max((total - s) / 2.0, p.attack_floor)


def dc_score_matrix(lh, la, p=None):
    """Normalised Dixon-Coles score-probability grid for two scoring rates."""
    p = p or DEFAULTS
    n = p.max_goals + 1
    ph = np.array([poisson_pmf(i, lh) for i in range(n)])
    pa = np.array([poisson_pmf(j, la) for j in range(n)])
    m = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            m[i, j] *= _dc_tau(i, j, lh, la, p.rho)
    s = m.sum()
    return m / s if s > 0 else m


def match_probs(elo_h, elo_a, p=None, home_adv=0.0):
    """(p_home_win, p_draw, p_away_win, lambda_home, lambda_away) from Elo."""
    p = p or DEFAULTS
    lh, la = lambdas_from_elo(elo_h, elo_a, p, home_adv)
    m = dc_score_matrix(lh, la, p)
    return (float(np.tril(m, -1).sum()), float(np.trace(m)),
            float(np.triu(m, 1).sum()), lh, la)


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------
class Predictor:
    def __init__(self, data_dir: str, model_ml=None):
        self.data_dir = data_dir
        self.elo = pd.read_csv(os.path.join(data_dir, "elo_ratings.csv"))
        self.fixtures = pd.read_csv(os.path.join(data_dir, "fixtures.csv")).fillna("")
        self.venues = pd.read_csv(os.path.join(data_dir, "venues.csv"))
        try:
            adj = pd.read_csv(os.path.join(data_dir, "adjustments.csv")).fillna("")
        except FileNotFoundError:
            adj = pd.DataFrame(columns=["team", "injury_factor",
                                        "travel_rest_factor", "weather_factor"])
        self.model_ml = model_ml

        self.elo_of = dict(zip(self.elo.team, self.elo.elo))
        self.elo_base = dict(self.elo_of)   # pre-tournament ratings (reference)
        self.group_of = dict(zip(self.elo.team, self.elo.group))
        self.host_of = dict(zip(self.elo.team, self.elo.host_country.fillna("")))
        self.conf_of = dict(zip(self.elo.team, self.elo.confederation))
        self.venue_row = {r.venue: r for r in self.venues.itertuples(index=False)}

        self.adj = {}
        for r in adj.itertuples(index=False):
            self.adj[r.team] = dict(
                injury=float(r.injury_factor or 1.0),
                travel=float(r.travel_rest_factor or 1.0),
                weather=float(r.weather_factor or 1.0),
            )

        # index group fixtures by the (home, away) pairing
        self._fx_no = {(fx.home_team, fx.away_team): int(fx.match_no)
                       for fx in self.fixtures.itertuples(index=False)}
        self.results = []            # ordered list of recorded results
        self.played_group = {}       # match_no -> (home_goals, away_goals)
        # knockout bracket data
        self.ko_bracket = []
        try:
            kdf = pd.read_csv(os.path.join(data_dir, "knockout_bracket.csv")).fillna("")
            for kr in kdf.itertuples(index=False):
                self.ko_bracket.append(dict(
                    match_no=int(kr.match_no), stage=str(kr.stage),
                    home_team=str(kr.home_team or ""), away_team=str(kr.away_team or ""),
                    home_slot=str(getattr(kr, "home_slot", "") or ""),
                    away_slot=str(getattr(kr, "away_slot", "") or ""),
                    date=str(kr.date_local), kickoff=str(kr.kickoff_local),
                    venue=str(kr.venue), city=str(kr.city), country=str(kr.country),
                    indoor=int(getattr(kr, "indoor", 0)),
                    temp_c=float(getattr(kr, "expected_temp_c", 22)),
                ))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        self.played_ko = {}          # match_no -> (home_goals, away_goals)
        self.ko_winners = {}         # match_no -> winning team
        self._load_results()         # also rolls Elo forward off real results

    def _load_results(self):
        """Read recorded tournament results, fix played group games, and roll
        each team's Elo forward off the actual scores so upcoming predictions
        reflect tournament form. Add rows to data/results.csv as games finish."""
        path = os.path.join(self.data_dir, "results.csv")
        if not os.path.exists(path):
            return
        df = pd.read_csv(path).fillna("")
        K = 40.0  # World Cup matches carry high weight
        for r in df.itertuples(index=False):
            h, a = str(r.home_team).strip(), str(r.away_team).strip()
            if h not in self.elo_of or a not in self.elo_of:
                continue
            try:
                hs, as_ = int(r.home_score), int(r.away_score)
            except (TypeError, ValueError):
                continue
            stage = str(getattr(r, "stage", "group")).strip().lower()
            self.results.append(dict(stage=stage, home=h, away=a,
                                     hs=hs, as_=as_))
            if stage in ("", "group") and (h, a) in self._fx_no:
                self.played_group[self._fx_no[(h, a)]] = (hs, as_)
            elif stage in ("r32", "r16", "qf", "sf", "final", "third"):
                for kf in self.ko_bracket:
                    if kf["stage"] == stage and kf.get("home_team", "") == h and kf.get("away_team", "") == a:
                        self.played_ko[kf["match_no"]] = (hs, as_)
                        self.ko_winners[kf["match_no"]] = h if hs > as_ else (a if as_ > hs else None)
                        break
            # roll Elo forward (home edge only when the home side is a host nation)
            eh, ea = self.elo_of[h], self.elo_of[a]
            adv = DEFAULTS.home_adv if self.host_of.get(h) else 0.0
            we = 1.0 / (10 ** (-((eh + adv) - ea) / 400.0) + 1)
            sh = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            margin = 1.0 + math.log1p(abs(hs - as_))
            delta = K * margin * (sh - we)
            self.elo_of[h] = eh + delta
            self.elo_of[a] = ea - delta

    @property
    def teams(self):
        return list(self.elo.team)

    # -- Layer 1: Elo -> expected goals --------------------------------------
    def base_lambdas(self, home, away, p: Params, venue_country="", venue=""):
        eh = self.elo_of[home]
        ea = self.elo_of[away]
        # Host advantage goes to whichever side is the host nation playing in its
        # own country -- regardless of which team is listed "home" (e.g. Czechia
        # vs Mexico at the Azteca: Mexico is the away side but on home soil).
        adv_h = p.home_adv if (venue_country and self.host_of.get(home) == venue_country) else 0.0
        adv_a = p.home_adv if (venue_country and self.host_of.get(away) == venue_country) else 0.0
        return lambdas_from_elo(eh, ea, p, home_adv=(adv_h - adv_a))

    # -- Layer 3: adjustments -------------------------------------------------
    def apply_adjustments(self, home, away, lh, la, p: Params, venue="",
                          venue_country="", overrides=None):
        ah = self.adj.get(home, {})
        aa = self.adj.get(away, {})
        lh *= ah.get("injury", 1.0) * ah.get("travel", 1.0) * ah.get("weather", 1.0)
        la *= aa.get("injury", 1.0) * aa.get("travel", 1.0) * aa.get("weather", 1.0)

        # Live slider overrides (Match Explorer): dict with home_/away_ factors
        if overrides:
            lh *= overrides.get("home_injury", 1.0) * overrides.get("home_travel", 1.0)
            la *= overrides.get("away_injury", 1.0) * overrides.get("away_travel", 1.0)

        # Venue weather: heat dampens tempo for both; altitude tires whichever
        # side is NOT acclimatised (i.e. not the host nation at a high venue).
        row = self.venue_row.get(venue)
        heat = overrides.get("heat", row.heat_index if row else None) if overrides else (
            row.heat_index if row else None)
        alt = row.altitude_m if row else 0
        if heat is not None:
            tempo = 1.0 - p.heat_tempo * float(heat)
            lh *= tempo
            la *= tempo
        if alt and alt > 1500 and venue_country:
            home_local = self.host_of.get(home) == venue_country
            away_local = self.host_of.get(away) == venue_country
            if home_local and not away_local:
                la *= (1.0 - p.altitude_penalty)
            elif away_local and not home_local:
                lh *= (1.0 - p.altitude_penalty)

        lh = min(max(lh, p.min_lambda), p.max_lambda)
        la = min(max(la, p.min_lambda), p.max_lambda)
        return lh, la

    # -- Layer 4: optional ML blend ------------------------------------------
    def blend_ml(self, home, away, lh, la, p: Params, venue_country=""):
        if not self.model_ml or p.ml_weight <= 0:
            return lh, la
        feats = self._ml_features(home, away, venue_country)
        try:
            lh_ml, la_ml = self.model_ml.predict_goals(feats)
        except Exception:
            return lh, la
        w = p.ml_weight
        return (1 - w) * lh + w * lh_ml, (1 - w) * la + w * la_ml

    def _ml_features(self, home, away, venue_country):
        return dict(
            elo_diff=self.elo_of[home] - self.elo_of[away],
            home_host=int(bool(venue_country and self.host_of.get(home) == venue_country)),
            same_conf=int(self.conf_of[home] == self.conf_of[away]),
            elo_home=self.elo_of[home],
            elo_away=self.elo_of[away],
        )

    # -- Layers combined: expected goals for a match -------------------------
    def expected_goals(self, home, away, p: Params = None, venue_country="",
                       venue="", overrides=None):
        p = p or DEFAULTS
        lh, la = self.base_lambdas(home, away, p, venue_country, venue)
        lh, la = self.apply_adjustments(home, away, lh, la, p, venue,
                                        venue_country, overrides)
        lh, la = self.blend_ml(home, away, lh, la, p, venue_country)
        return lh, la

    # -- Layer 2: Dixon-Coles score matrix -----------------------------------
    def score_matrix(self, lh, la, p: Params = None):
        return dc_score_matrix(lh, la, p)

    def predict_match(self, home, away, p: Params = None, venue_country="",
                      venue="", overrides=None, top_n=5):
        p = p or DEFAULTS
        lh, la = self.expected_goals(home, away, p, venue_country, venue, overrides)
        m = self.score_matrix(lh, la, p)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        p_home = float(np.tril(m, -1).sum())   # home goals > away goals
        p_away = float(np.triu(m, 1).sum())    # away goals > home goals
        p_draw = float(np.trace(m))
        # both teams to score and over/under 2.5 goals
        btts = float(m[1:, 1:].sum())
        idx = np.indices(m.shape).sum(axis=0)
        over25 = float(m[idx >= 3].sum())
        # top correct scores
        flat = [((a, b), float(m[a, b])) for a in range(m.shape[0])
                for b in range(m.shape[1])]
        flat.sort(key=lambda x: -x[1]        )
        return dict(
            home=home, away=away,
            lambda_home=round(lh, 3), lambda_away=round(la, 3),
            most_likely_score=f"{i}-{j}",          # modal exact score (base lens)
            expected_scoreline=f"{round(lh)}-{round(la)}",  # rounded xG (expanded lens)
            score_home=int(i), score_away=int(j),
            p_home_win=round(p_home, 4), p_draw=round(p_draw, 4),
            p_away_win=round(p_away, 4),
            btts=round(btts, 4), over25=round(over25, 4),
            top_scores=[(f"{a}-{b}", round(pr, 4)) for (a, b), pr in flat[:top_n]],
            matrix=m,
        )

    def predict_knockout_match(self, home, away, p: Params = None, venue_country="",
                                venue="", overrides=None, top_n=5):
        """90-min exact-score prediction + advancement probability for a
        knockout tie. Extra time (30 min at ~1/3 goal rate) and penalties
        (weighted by λ share) resolve draws."""
        p = p or DEFAULTS
        lh, la = self.expected_goals(home, away, p, venue_country, venue, overrides)
        m = self.score_matrix(lh, la, p)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        p_home_90 = float(np.tril(m, -1).sum())
        p_draw_90 = float(np.trace(m))
        p_away_90 = float(np.triu(m, 1).sum())
        btts = float(m[1:, 1:].sum())
        idx_sum = np.indices(m.shape).sum(axis=0)
        over25 = float(m[idx_sum >= 3].sum())
        flat = [((a, b), float(m[a, b])) for a in range(m.shape[0])
                for b in range(m.shape[1])]
        flat.sort(key=lambda x: -x[1])
        # Extra time: 30 min ~ 1/3 of 90-min λ
        et_factor = 0.33
        lh_et = lh * et_factor
        la_et = la * et_factor
        et_n = min(p.max_goals, 4) + 1
        ph_et = np.array([poisson_pmf(k, lh_et) for k in range(et_n)])
        pa_et = np.array([poisson_pmf(k, la_et) for k in range(et_n)])
        pem = np.outer(ph_et, pa_et)
        p_home_et = float(np.tril(pem, -1).sum())
        p_draw_et = float(np.trace(pem))
        p_away_et = float(np.triu(pem, 1).sum())
        # Penalties weighted by λ share
        total_l = lh + la
        p_home_pens = lh / total_l if total_l > 0 else 0.5
        # Advancement probability: home wins in 90, or draws then wins in ET, or draws ET then pens
        p_home_adv = (p_home_90 + p_draw_90 * p_home_et +
                      p_draw_90 * p_draw_et * p_home_pens)
        p_away_adv = 1.0 - p_home_adv
        return dict(
            home=home, away=away,
            lambda_home=round(lh, 3), lambda_away=round(la, 3),
            most_likely_score=f"{i}-{j}",
            expected_scoreline=f"{round(lh)}-{round(la)}",
            score_home=int(i), score_away=int(j),
            p_home_90=round(p_home_90, 4), p_draw_90=round(p_draw_90, 4),
            p_away_90=round(p_away_90, 4),
            p_home_adv=round(p_home_adv, 4), p_away_adv=round(p_away_adv, 4),
            p_home_et=round(p_home_et, 4), p_draw_et=round(p_draw_et, 4),
            p_away_et=round(p_away_et, 4),
            p_home_pens=round(p_home_pens, 4),
            btts=round(btts, 4), over25=round(over25, 4),
            top_scores=[(f"{a}-{b}", round(pr, 4)) for (a, b), pr in flat[:top_n]],
        )

    def resolve_knockout_bracket(self, p: Params = None):
        """Resolve the real knockout bracket: lock played KO ties, predict
        unplayed ones, propagate winners through R16/QF/SF/Final/Third.
        Returns list of round dicts matching projected_knockouts format."""
        p = p or DEFAULTS
        if not self.ko_bracket:
            return self.projected_knockouts(p)
        bracket = dict((k["match_no"], dict(k)) for k in self.ko_bracket)
        resolved_teams = {}
        resolved_winners = {}
        stage_order = ["r32", "r16", "qf", "sf", "third", "final"]
        rounds_data = []
        round_labels = {"r32": "Round of 32", "r16": "Round of 16",
                        "qf": "Quarter-finals", "sf": "Semi-finals",
                        "third": "Third place", "final": "Final"}
        for stg in stage_order:
            matches_in_round = sorted(
                [k for k, v in bracket.items() if v["stage"] == stg],
                key=lambda x: x)
            if not matches_in_round:
                continue
            games = []
            for mn in matches_in_round:
                v = bracket[mn]
                h = v.get("home_team", "") or resolved_teams.get(v.get("home_slot", ""), "")
                a = v.get("away_team", "") or resolved_teams.get(v.get("away_slot", ""), "")
                if not h or not a:
                    continue
                played = mn in self.played_ko
                if played:
                    hs, as_ = self.played_ko[mn]
                    score = f"{hs}-{as_}"
                    w = h if hs > as_ else (a if as_ > hs else None)
                    note = ""
                    if hs == as_:
                        note = "pens"
                        w = resolved_winners.get(mn, h)
                else:
                    r = self.predict_knockout_match(h, a, p)
                    score = r["most_likely_score"]
                    w = h if r.get("p_home_adv", 0) >= r.get("p_away_adv", 0) else a
                    if r["score_home"] == r["score_away"]:
                        p_pens = r.get("p_draw_90", 0) * r.get("p_draw_et", 0)
                        if p_pens > 0.03:
                            note = "pens"
                        else:
                            note = "et"
                    else:
                        note = ""
                if w:
                    resolved_winners[mn] = w
                for slot_key in [f"{mn}W", f"{mn}w"]:
                    resolved_teams[slot_key] = w or ""
                resolved_teams[f"{mn}L"] = a if w == h else (h if w else "")
                note_display = {"pens": " (aet/pens)", "et": " (aet)"}.get(note, "")
                p_home_val = 0.5
                p_away_val = 0.5
                if not played:
                    p_home_val = r.get("p_home_adv", 0)
                    p_away_val = r.get("p_away_adv", 0)
                games.append(dict(
                    home=h, away=a, score=score, winner=w or "TBD",
                    note=note_display, played=played,
                    p_home=round(p_home_val, 4), p_away=round(p_away_val, 4),
                ))
            if games:
                rounds_data.append(dict(label=round_labels.get(stg, stg), games=games))
        if resolved_winners:
            champ = resolved_winners.get(max(k for k in resolved_winners if bracket[k]["stage"] in ("final", "sf")), "TBD")
            rounds_data.append(dict(label="Champion", games=[dict(winner=champ)]))
        return rounds_data

    @staticmethod
    def _odds(prob):
        """Fair (model-implied) decimal odds from a probability."""
        return round(1.0 / prob, 2) if prob > 1e-6 else None

    def explain_match(self, home, away, p: Params = None, venue_country="",
                      venue="", overrides=None):
        """Full step-by-step derivation of a match prediction, with every
        intermediate number, for the 'how the score was predicted' panel."""
        p = p or DEFAULTS
        eh, ea = self.elo_of[home], self.elo_of[away]
        adv_h = p.home_adv if (venue_country and self.host_of.get(home) == venue_country) else 0.0
        adv_a = p.home_adv if (venue_country and self.host_of.get(away) == venue_country) else 0.0
        eh_adj, ea_adj = eh + adv_h, ea + adv_a
        dr = eh_adj - ea_adj
        supremacy = p.elo_to_goals * dr / 400.0
        total = p.base_total + p.goal_spread * abs(supremacy)
        lh_base = max((total + supremacy) / 2.0, p.attack_floor)
        la_base = max((total - supremacy) / 2.0, p.attack_floor)

        # adjustment factors actually in play
        ah, aa = self.adj.get(home, {}), self.adj.get(away, {})
        factors = []
        for nm in ("injury", "travel", "weather"):
            factors.append(dict(name=nm, side=home, value=round(ah.get(nm, 1.0), 3)))
            factors.append(dict(name=nm, side=away, value=round(aa.get(nm, 1.0), 3)))
        if overrides:
            ht, at = overrides.get("home_travel", 1.0), overrides.get("away_travel", 1.0)
            if ht != 1.0:
                factors.append(dict(name="rest/travel", side=home, value=round(ht, 3)))
            if at != 1.0:
                factors.append(dict(name="rest/travel", side=away, value=round(at, 3)))
        row = self.venue_row.get(venue)
        heat = float(row.heat_index) if row is not None else None
        tempo = round(1.0 - p.heat_tempo * heat, 3) if heat is not None else 1.0
        alt = float(row.altitude_m) if row is not None else 0.0
        alt_side = None
        if alt > 1500 and venue_country:
            if self.host_of.get(home) == venue_country and self.host_of.get(away) != venue_country:
                alt_side = away
            elif self.host_of.get(away) == venue_country and self.host_of.get(home) != venue_country:
                alt_side = home

        lh, la = self.expected_goals(home, away, p, venue_country, venue, overrides)
        m = self.score_matrix(lh, la, p)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        p_home = float(np.tril(m, -1).sum())
        p_away = float(np.triu(m, 1).sum())
        p_draw = float(np.trace(m))
        btts = float(m[1:, 1:].sum())
        idx = np.indices(m.shape).sum(axis=0)
        over25 = float(m[idx >= 3].sum())
        poisson_home = [round(poisson_pmf(k, lh), 4) for k in range(7)]
        poisson_away = [round(poisson_pmf(k, la), 4) for k in range(7)]
        flat = sorted(((a, b, float(m[a, b])) for a in range(7) for b in range(7)),
                      key=lambda x: -x[2])
        return dict(
            home=home, away=away,
            inputs=dict(elo_home=eh, elo_away=ea, adv_home=adv_h, adv_away=adv_a,
                        elo_home_adj=eh_adj, elo_away_adj=ea_adj),
            supremacy=dict(dr=round(dr, 1), elo_to_goals=p.elo_to_goals,
                           supremacy=round(supremacy, 3)),
            base=dict(base_total=p.base_total, goal_spread=p.goal_spread,
                      total=round(total, 3), lh_base=round(lh_base, 3),
                      la_base=round(la_base, 3), attack_floor=p.attack_floor),
            factors=factors,
            weather=dict(venue=venue or "Neutral", heat_index=heat, tempo=tempo,
                         heat_tempo=p.heat_tempo, altitude_m=alt,
                         altitude_side=alt_side, altitude_penalty=p.altitude_penalty),
            final=dict(lh=round(lh, 3), la=round(la, 3)),
            dixon_coles=dict(rho=p.rho),
            poisson=dict(home=poisson_home, away=poisson_away),
            grid6=[[round(float(m[a, b]), 4) for b in range(6)] for a in range(6)],
            grid_top=[(a, b, round(pr, 4)) for a, b, pr in flat[:8]],
            outcomes=dict(p_home=round(p_home, 4), p_draw=round(p_draw, 4),
                          p_away=round(p_away, 4), btts=round(btts, 4),
                          over25=round(over25, 4)),
            odds=dict(home=self._odds(p_home), draw=self._odds(p_draw),
                      away=self._odds(p_away)),
            scores=dict(most_likely=f"{i}-{j}", expected=f"{round(lh)}-{round(la)}"),
        )

    def fixture_overrides(self, fx):
        """Schedule-derived rest/travel multipliers for a fixture, fed into the
        model as home_/away_travel so they move the expected goals and odds."""
        def num(v, d=1.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return d
        return dict(home_travel=num(getattr(fx, "home_travel_factor", 1.0)),
                    away_travel=num(getattr(fx, "away_travel_factor", 1.0)))

    def predict_all_group_matches(self, p: Params = None):
        p = p or DEFAULTS
        pe = expanded_params(p)
        rows = []
        for fx in self.fixtures.itertuples(index=False):
            ov = self.fixture_overrides(fx)
            base = self.predict_match(fx.home_team, fx.away_team, p,
                                      venue_country=fx.country, venue=fx.venue,
                                      overrides=ov)
            exp = self.predict_match(fx.home_team, fx.away_team, pe,
                                     venue_country=fx.country, venue=fx.venue,
                                     overrides=ov)
            played = int(fx.match_no) in self.played_group
            actual = ""
            if played:
                hs, ag = self.played_group[int(fx.match_no)]
                actual = f"{hs}-{ag}"
            rows.append(dict(
                played=int(played), actual=actual,
                correct=int(played and actual == base["most_likely_score"]),
                match_no=fx.match_no, group=fx.group, match_day=fx.match_day,
                date=fx.date_local, kickoff=fx.kickoff_local,
                home=fx.home_team, away=fx.away_team,
                elo_home=self.elo_of[fx.home_team], elo_away=self.elo_of[fx.away_team],
                city=fx.city, venue=fx.venue, country=fx.country,
                indoor=int(fx.indoor), temp_c=fx.expected_temp_c,
                predicted_score=base["most_likely_score"],
                expanded_score=exp["expected_scoreline"],
                xg_home=base["lambda_home"], xg_away=base["lambda_away"],
                xg_home_exp=exp["lambda_home"], xg_away_exp=exp["lambda_away"],
                p_home_win=base["p_home_win"], p_draw=base["p_draw"],
                p_away_win=base["p_away_win"],
                btts=exp["btts"], over25=exp["over25"],
                p_top_score=base["top_scores"][0][1],
                top_scores_exp="; ".join(f"{s} {pr*100:.0f}%"
                                         for s, pr in exp["top_scores"][:3]),
            ))
        return pd.DataFrame(rows)

    # -- Deterministic "most likely" projections (for display) ---------------
    def projected_standings(self, p: Params = None):
        """Rank each group. For played matches use actual points/GD/GF; for
        unplayed matches use expected points. Return projected winners,
        runners-up and the 8 best third-placed teams."""
        p = p or DEFAULTS
        xpts = {t: 0.0 for t in self.teams}
        xgd = {t: 0.0 for t in self.teams}
        xgf = {t: 0.0 for t in self.teams}
        for fx in self.fixtures.itertuples(index=False):
            h, a = fx.home_team, fx.away_team
            mno = int(fx.match_no)
            fixed = self.played_group.get(mno)
            if fixed is not None:
                # Use actual result
                hg, ag = fixed
                xpts[h] += 3 if hg > ag else (1 if hg == ag else 0)
                xpts[a] += 3 if ag > hg else (1 if hg == ag else 0)
                xgd[h] += hg - ag
                xgd[a] += ag - hg
                xgf[h] += hg
                xgf[a] += ag
            else:
                # Use predicted expected points
                r = self.predict_match(h, a, p,
                                       venue_country=fx.country, venue=fx.venue)
                xpts[h] += 3 * r["p_home_win"] + r["p_draw"]
                xpts[a] += 3 * r["p_away_win"] + r["p_draw"]
                xgd[h] += r["lambda_home"] - r["lambda_away"]
                xgd[a] += r["lambda_away"] - r["lambda_home"]
                xgf[h] += r["lambda_home"]
                xgf[a] += r["lambda_away"]

        groups = sorted(self.elo.group.unique())
        ordered = {}
        winners, runners, thirds = [], [], []
        for g in groups:
            gt = list(self.elo[self.elo.group == g].team)
            gt.sort(key=lambda t: (xpts[t], xgd[t], xgf[t], self.elo_of[t]),
                    reverse=True)
            ordered[g] = [(t, round(xpts[t], 1)) for t in gt]
            winners.append(gt[0]); runners.append(gt[1])
            thirds.append((gt[2], xpts[gt[2]], xgd[gt[2]], xgf[gt[2]]))
        best_thirds = [t for t, *_ in sorted(thirds, key=lambda x: (x[1], x[2], x[3]),
                                             reverse=True)[:8]]
        return dict(ordered=ordered, winners=winners, runners=runners,
                    best_thirds=best_thirds,
                    qualifiers=winners + runners + best_thirds)

    def projected_knockouts(self, p: Params = None):
        """A single representative bracket using the projected qualifiers,
        with the most-likely scoreline for each tie. Pairings depend on real
        results, so this is illustrative, not a fixed forecast."""
        p = p or DEFAULTS
        st = self.projected_standings(p)
        seeds = sorted(st["qualifiers"], key=lambda t: self.elo_of[t], reverse=True)
        bracket = self._seed_bracket(seeds)
        labels = ["Round of 32", "Round of 16", "Quarter-finals",
                  "Semi-finals", "Final"]
        rounds = []
        for lab in labels:
            games, nxt = [], []
            for x in range(0, len(bracket), 2):
                h, a = bracket[x], bracket[x + 1]
                r = self.predict_match(h, a, p)
                if r["p_home_win"] >= r["p_away_win"]:
                    w = h
                else:
                    w = a
                note = ""
                # if the modal score is level, mark it as decided on penalties
                if r["score_home"] == r["score_away"]:
                    note = "pens"
                games.append(dict(home=h, away=a, score=r["most_likely_score"],
                                  winner=w, note=note,
                                  p_home=r["p_home_win"], p_away=r["p_away_win"]))
                nxt.append(w)
            rounds.append(dict(label=lab, games=games))
            bracket = nxt
        rounds.append(dict(label="Champion", games=[dict(winner=bracket[0])]))
        return rounds

    # -- Layer 5: Monte Carlo tournament -------------------------------------
    def simulate_tournament(self, n_sims=10000, p: Params = None, seed=42):
        p = p or DEFAULTS
        rng = np.random.default_rng(seed)
        teams = self.teams
        idx = {t: k for k, t in enumerate(teams)}

        # Pre-compute each group fixture: a fixed real score if already played,
        # otherwise the expected goals to simulate from.
        gfix = []
        for fx in self.fixtures.itertuples(index=False):
            fixed = self.played_group.get(int(fx.match_no))
            if fixed is not None:
                gfix.append((fx.group, fx.home_team, fx.away_team, None, None, fixed))
            else:
                lh, la = self.expected_goals(fx.home_team, fx.away_team, p,
                                             venue_country=fx.country, venue=fx.venue,
                                             overrides=self.fixture_overrides(fx))
                gfix.append((fx.group, fx.home_team, fx.away_team, lh, la, None))

        groups = sorted(self.elo.group.unique())
        group_teams = {g: list(self.elo[self.elo.group == g].team) for g in groups}

        reached = {t: dict(r32=0, won_r32=0, won_r16=0, won_qf=0, won_sf=0, won_final=0, champ=0) for t in teams}

        for _ in range(n_sims):
            # ---- group stage ----
            pts = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            ga = {t: 0 for t in teams}
            for g, h, a, lh, la, fixed in gfix:
                if fixed is not None:
                    hg, ag = fixed          # already played — use the real score
                else:
                    hg = rng.poisson(lh)
                    ag = rng.poisson(la)
                gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
                if hg > ag:
                    pts[h] += 3
                elif hg < ag:
                    pts[a] += 3
                else:
                    pts[h] += 1; pts[a] += 1

            def rank_key(t):
                return (pts[t], gf[t] - ga[t], gf[t], rng.random())

            winners, runners, thirds = [], [], []
            for g in groups:
                order = sorted(group_teams[g], key=rank_key, reverse=True)
                winners.append(order[0])
                runners.append(order[1])
                thirds.append((order[2], pts[order[2]], gf[order[2]] - ga[order[2]],
                               gf[order[2]]))

            best_thirds = [t for t, *_ in sorted(
                thirds, key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)[:8]]

            qualifiers = winners + runners + best_thirds
            for t in qualifiers:
                reached[t]["r32"] += 1

            # ---- knockouts (single-elimination bracket) ----
            # Use the real FIFA bracket from knockout_bracket.csv if available;
            # otherwise fall back to Elo-strength seeding.
            bracket = self._build_mc_bracket(qualifiers, p)

            # 32 -> 16 (won_r32) -> 8 (won_r16) -> 4 (won_qf) -> 2 (won_sf) -> 1 (won_final + champ).
            ko_rounds = [("won_r32", "r16"), ("won_r16", "qf"), ("won_qf", "sf"),
                         ("won_sf", "final"), ("won_final", "champ")]
            for wkey, ckey in ko_rounds:
                nxt = []
                for x in range(0, len(bracket), 2):
                    w = self._sim_ko(bracket[x], bracket[x + 1], p, rng)
                    reached[w][wkey] += 1
                    nxt.append(w)
                bracket = nxt
            if bracket:
                reached[bracket[0]]["champ"] += 1

        rows = []
        for t in teams:
            d = reached[t]
            rows.append(dict(
                team=t, group=self.group_of[t], elo=self.elo_of[t],
                reach_r32=d["r32"] / n_sims, reach_r16=d["won_r32"] / n_sims,
                reach_qf=d["won_r16"] / n_sims, reach_sf=d["won_qf"] / n_sims,
                reach_final=d["won_sf"] / n_sims, win_title=d["champ"] / n_sims,
            ))

        df = pd.DataFrame(rows).sort_values("win_title", ascending=False)
        return df.reset_index(drop=True)

    def _build_mc_bracket(self, qualifiers, p=None):
        """Build the R32 bracket as a flat list of 32 teams, ordered by the
        real FIFA bracket structure (from knockout_bracket.csv) when available,
        falling back to Elo-strength seeding otherwise.

        Returns a flat list of 32 team names where bracket[0] vs bracket[1],
        bracket[2] vs bracket[3], etc. are the R32 ties."""
        if not self.ko_bracket:
            seeds = sorted(qualifiers, key=lambda t: self.elo_of[t], reverse=True)
            return self._seed_bracket(seeds)
        # Extract R32 matches from the real bracket, in match-number order
        r32_matches = sorted(
            [kf for kf in self.ko_bracket if kf["stage"] == "r32"],
            key=lambda x: x["match_no"])
        bracket = []
        for kf in r32_matches:
            h = kf.get("home_team", "")
            a = kf.get("away_team", "")
            if h and a and h in qualifiers and a in qualifiers:
                bracket.extend([h, a])
            else:
                # Fallback: shouldn't happen if bracket data is correct
                bracket.extend([h or "TBD", a or "TBD"])
        # If bracket is incomplete (missing teams), fall back to Elo seeding
        if len(bracket) < 32 or "TBD" in bracket:
            seeds = sorted(qualifiers, key=lambda t: self.elo_of[t], reverse=True)
            return self._seed_bracket(seeds)
        return bracket

    def _seed_bracket(self, seeds):
        """Proper tournament seeding so the strongest sides are spread across
        the bracket (1 and 2 can only meet in the final), not clustered."""
        order = [0]
        while len(order) < len(seeds):
            size = len(order) * 2
            order = [v for s in order for v in (s, size - 1 - s)]
        return [seeds[i] for i in order]

    def _sim_ko(self, home, away, p: Params, rng):
        lh, la = self.expected_goals(home, away, p)
        hg = rng.poisson(lh)
        ag = rng.poisson(la)
        if hg > ag:
            return home
        if ag > hg:
            return away
        # draw -> extra time (30 min ~ 1/3 of 90-min λ)
        et_factor = 0.33
        hg_et = rng.poisson(lh * et_factor)
        ag_et = rng.poisson(la * et_factor)
        hg_total, ag_total = hg + hg_et, ag + ag_et
        if hg_total > ag_total:
            return home
        if ag_total > hg_total:
            return away
        # still draw -> penalties, weighted by relative strength
        ph = lh / (lh + la)
        return home if rng.random() < ph else away


def load_predictor(data_dir=None, model_ml=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return Predictor(data_dir, model_ml=model_ml)
