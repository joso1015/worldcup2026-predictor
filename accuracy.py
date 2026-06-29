"""
Current-results accuracy and parameter sweep helpers.

This is not a historical backtest. It fits/checks the model against the real
World Cup 2026 results currently entered in data/results.csv.
"""
from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Tuple

import pandas as pd

import model


DEFAULT_REFERENCE = model.Params(home_adv=80.0, elo_to_goals=1.65,
                                 base_total=2.70, rho=-0.04)


PARAM_FIELDS = [
    "home_adv", "elo_to_goals", "base_total", "rho",
    "heat_tempo", "altitude_penalty", "ml_weight",
]


def score_tuple(score: str) -> Tuple[int, int]:
    home, away = score.split("-", 1)
    return int(home), int(away)


def result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def result_from_score(score: str) -> str:
    home_goals, away_goals = score_tuple(score)
    return result_from_goals(home_goals, away_goals)


def params_dict(p: model.Params) -> Dict[str, float]:
    return {field: round(float(getattr(p, field)), 4) for field in PARAM_FIELDS}


def params_label(p: model.Params) -> str:
    d = params_dict(p)
    return (f"home +{d['home_adv']:.0f} Elo, scale {d['elo_to_goals']:.2f}, "
            f"goals {d['base_total']:.2f}, rho {d['rho']:.2f}, "
            f"heat {d['heat_tempo']:.2f}, altitude {d['altitude_penalty']:.2f}, "
            f"ML {d['ml_weight']:.2f}")


def played_fixtures(P: model.Predictor):
    fixtures = {int(fx.match_no): fx for fx in P.fixtures.itertuples(index=False)}
    return [(match_no, fixtures[match_no], score)
            for match_no, score in sorted(P.played_group.items())
            if match_no in fixtures]


def evaluate_params(P: model.Predictor, p: model.Params) -> Dict:
    rows = []
    metrics = dict(
        played=0,
        ml_score_correct=0,
        ml_result_correct=0,
        exp_score_correct=0,
        exp_result_correct=0,
        ml_goal_error=0,
        exp_goal_error=0,
    )
    expanded = model.expanded_params(p)

    for match_no, fx, (actual_home, actual_away) in played_fixtures(P):
        overrides = P.fixture_overrides(fx)
        base = P.predict_match(fx.home_team, fx.away_team, p,
                               venue_country=fx.country, venue=fx.venue,
                               overrides=overrides)
        exp = P.predict_match(fx.home_team, fx.away_team, expanded,
                              venue_country=fx.country, venue=fx.venue,
                              overrides=overrides)
        actual_score = f"{actual_home}-{actual_away}"
        actual_result = result_from_goals(actual_home, actual_away)
        ml_score = base["most_likely_score"]
        exp_score = exp["expected_scoreline"]
        ml_home, ml_away = score_tuple(ml_score)
        exp_home, exp_away = score_tuple(exp_score)
        ml_score_hit = ml_score == actual_score
        exp_score_hit = exp_score == actual_score
        ml_result_hit = result_from_score(ml_score) == actual_result
        exp_result_hit = result_from_score(exp_score) == actual_result
        ml_error = abs(ml_home - actual_home) + abs(ml_away - actual_away)
        exp_error = abs(exp_home - actual_home) + abs(exp_away - actual_away)

        metrics["played"] += 1
        metrics["ml_score_correct"] += int(ml_score_hit)
        metrics["ml_result_correct"] += int(ml_result_hit)
        metrics["exp_score_correct"] += int(exp_score_hit)
        metrics["exp_result_correct"] += int(exp_result_hit)
        metrics["ml_goal_error"] += ml_error
        metrics["exp_goal_error"] += exp_error

        rows.append(dict(
            match_no=match_no,
            group=fx.group,
            home=fx.home_team,
            away=fx.away_team,
            actual=actual_score,
            actual_result=actual_result,
            most_likely=ml_score,
            most_likely_result=result_from_score(ml_score),
            most_likely_score_correct=ml_score_hit,
            most_likely_result_correct=ml_result_hit,
            expanded=exp_score,
            expanded_result=result_from_score(exp_score),
            expanded_score_correct=exp_score_hit,
            expanded_result_correct=exp_result_hit,
        ))

    played = max(metrics["played"], 1)
    metrics["ml_avg_goal_error"] = round(metrics["ml_goal_error"] / played, 4)
    metrics["exp_avg_goal_error"] = round(metrics["exp_goal_error"] / played, 4)
    return dict(params=params_dict(p), label=params_label(p), metrics=metrics,
                matches=rows)


def param_distance(p: model.Params, ref: model.Params = DEFAULT_REFERENCE) -> float:
    return round(
        abs(p.home_adv - ref.home_adv) / 80.0 +
        abs(p.elo_to_goals - ref.elo_to_goals) / 0.5 +
        abs(p.base_total - ref.base_total) / 0.5 +
        abs(p.rho - ref.rho) / 0.1 +
        abs(p.heat_tempo - ref.heat_tempo) / 0.1 +
        abs(p.altitude_penalty - ref.altitude_penalty) / 0.1 +
        abs(p.ml_weight - ref.ml_weight),
        6,
    )


def practical_grid(include_ml: bool = False) -> List[model.Params]:
    stage_a = []
    for home_adv, elo_to_goals, base_total, rho in itertools.product(
        [40.0, 60.0, 80.0, 100.0, 120.0],
        [1.30, 1.45, 1.60, 1.65, 1.75, 1.90, 2.05],
        [2.30, 2.50, 2.70, 2.90, 3.10],
        [-0.12, -0.08, -0.04, 0.00],
    ):
        stage_a.append(model.Params(home_adv=home_adv, elo_to_goals=elo_to_goals,
                                    base_total=base_total, rho=rho))

    ml_weights = [0.0, 0.25, 0.50] if include_ml else [0.0]
    refined = []
    for base in stage_a:
        if base.home_adv not in (60.0, 80.0, 100.0):
            continue
        if base.elo_to_goals not in (1.45, 1.60, 1.65, 1.75, 1.90):
            continue
        if base.base_total not in (2.50, 2.70, 2.90):
            continue
        if base.rho not in (-0.08, -0.04, 0.00):
            continue
        for heat_tempo, altitude_penalty, ml_weight in itertools.product(
            [0.06, 0.10, 0.12, 0.16],
            [0.00, 0.04, 0.08, 0.12],
            ml_weights,
        ):
            refined.append(model.Params(
                home_adv=base.home_adv,
                elo_to_goals=base.elo_to_goals,
                base_total=base.base_total,
                rho=base.rho,
                heat_tempo=heat_tempo,
                altitude_penalty=altitude_penalty,
                ml_weight=ml_weight,
            ))

    seen = set()
    out = []
    for p in stage_a + refined:
        key = tuple(params_dict(p).items())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def row_from_evaluation(evaluation: Dict, p: model.Params) -> Dict:
    row = dict(params_dict(p))
    row.update(evaluation["metrics"])
    row["param_distance"] = param_distance(p)
    row["label"] = evaluation["label"]
    return row


def rank_rows(rows: List[Dict], target: str) -> List[Dict]:
    if target == "ml_score":
        return sorted(rows, key=lambda r: (
            -r["ml_score_correct"], -r["ml_result_correct"],
            r["ml_avg_goal_error"], r["param_distance"], r["label"]
        ))
    if target == "ml_result":
        return sorted(rows, key=lambda r: (
            -r["ml_result_correct"], -r["ml_score_correct"],
            r["ml_avg_goal_error"], r["param_distance"], r["label"]
        ))
    if target == "exp_score":
        return sorted(rows, key=lambda r: (
            -r["exp_score_correct"], -r["exp_result_correct"],
            r["exp_avg_goal_error"], r["param_distance"], r["label"]
        ))
    if target == "exp_result":
        return sorted(rows, key=lambda r: (
            -r["exp_result_correct"], -r["exp_score_correct"],
            r["exp_avg_goal_error"], r["param_distance"], r["label"]
        ))
    raise ValueError(f"Unknown target: {target}")


def sweep_current_results(P: model.Predictor, include_ml: bool = False) -> Dict:
    rows = []
    for p in practical_grid(include_ml=include_ml):
        rows.append(row_from_evaluation(evaluate_params(P, p), p))

    winners = {
        "most_likely_score": rank_rows(rows, "ml_score")[0] if rows else None,
        "most_likely_result": rank_rows(rows, "ml_result")[0] if rows else None,
        "expanded_score": rank_rows(rows, "exp_score")[0] if rows else None,
        "expanded_result": rank_rows(rows, "exp_result")[0] if rows else None,
    }
    return dict(rows=rows, winners=winners)


def summary_for_params(P: model.Predictor, p: model.Params,
                       include_sweep: bool = True,
                       include_ml: bool = False) -> Dict:
    current = evaluate_params(P, p)
    sweep = sweep_current_results(P, include_ml=include_ml) if include_sweep else {
        "rows": [], "winners": {}
    }
    return dict(
        played=current["metrics"]["played"],
        current=current,
        winners=sweep["winners"],
        rows=sweep["rows"],
        note="Fitted to currently entered results only; this can overfit a small sample.",
    )


def write_outputs(summary: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = summary.get("rows", [])
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "parameter_sweep_current.csv"),
                              index=False)
    payload = {k: v for k, v in summary.items() if k != "rows"}
    with open(os.path.join(out_dir, "parameter_sweep_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_summary(out_dir: str) -> Dict | None:
    path = os.path.join(out_dir, "parameter_sweep_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
