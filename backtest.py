"""
Backtest & calibration harness for the prediction model.

Walks ~47k historical international matches in date order, maintains a running
Elo for every nation, and for each match (after a burn-in) produces the SAME
Dixon-Coles match probabilities the live model uses (model.match_probs). It then
scores those probabilities against what actually happened with the metrics the
football-forecasting literature uses:

  * RPS (Ranked Probability Score) — the standard ordered-outcome metric
    (Constantinou & Fenton, 2012). Lower is better.
  * Log-loss and multiclass Brier score.
  * Accuracy (most-likely outcome correct).

It compares the model to a naive base-rate baseline, draws a reliability
(calibration) diagram, and runs a small train/test parameter search over the
hand-set constants (elo_to_goals, base_total, rho, home advantage) to show
whether better-calibrated values exist — a lightweight stand-in for full
maximum-likelihood fitting.

Run (needs internet, like build_history.py):
    python backtest.py
Validate the metric maths offline:
    python backtest.py --selftest

Outputs: outputs/backtest_report.txt and outputs/reliability.png
"""
import csv
import math
import os
import sys
import urllib.request
from dataclasses import replace

import numpy as np

import model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")

BURN_IN_YEAR = 2008      # only evaluate matches from this year on
MIN_PRIOR_GAMES = 10     # both teams need this many prior games to be eligible
ELO_BASE, ELO_K, HOME_FIELD = 1500.0, 40.0, 65.0


# --------------------------------------------------------------------------- #
# Metrics (outcome: 0 = home win, 1 = draw, 2 = away win)
# --------------------------------------------------------------------------- #
def rps(probs, outcome):
    """Ranked Probability Score for an ordered 3-outcome forecast."""
    o = [0.0, 0.0, 0.0]
    o[outcome] = 1.0
    cp = co = s = 0.0
    for i in range(2):                 # r - 1 cumulative steps
        cp += probs[i]
        co += o[i]
        s += (cp - co) ** 2
    return s / 2.0


def log_loss(probs, outcome):
    return -math.log(max(probs[outcome], 1e-12))


def brier(probs, outcome):
    o = [0.0, 0.0, 0.0]
    o[outcome] = 1.0
    return sum((probs[i] - o[i]) ** 2 for i in range(3))


def accuracy(probs, outcome):
    return 1.0 if max(range(3), key=lambda k: probs[k]) == outcome else 0.0


# --------------------------------------------------------------------------- #
# Data + Elo reconstruction
# --------------------------------------------------------------------------- #
def download_matches():
    with urllib.request.urlopen(RESULTS_URL, timeout=60) as r:
        text = r.read().decode("utf-8", "ignore")
    rows = []
    for r in csv.DictReader(text.splitlines()):
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError):
            continue
        rows.append((r.get("date", ""), r["home_team"], r["away_team"], hs, as_,
                     str(r.get("neutral", "")).strip().upper() == "TRUE"))
    rows.sort(key=lambda x: x[0])
    return rows


def build_eval_records(rows):
    """Replay history, returning (elo_home, elo_away, home_adv, outcome) for
    every eligible match (pre-match ratings, so no leakage)."""
    elo, games = {}, {}
    records = []
    for date, h, a, hs, as_, neutral in rows:
        eh = elo.get(h, ELO_BASE)
        ea = elo.get(a, ELO_BASE)
        adv = 0.0 if neutral else HOME_FIELD
        year = int(date[:4]) if date[:4].isdigit() else 0
        eligible = (year >= BURN_IN_YEAR and games.get(h, 0) >= MIN_PRIOR_GAMES
                    and games.get(a, 0) >= MIN_PRIOR_GAMES)
        if eligible:
            outcome = 0 if hs > as_ else (1 if hs == as_ else 2)
            records.append((eh, ea, adv, outcome))
        # update Elo (goal-difference weighted)
        we = 1.0 / (10 ** (-((eh + adv) - ea) / 400.0) + 1)
        sh = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = ELO_K * (1.0 + math.log1p(abs(hs - as_))) * (sh - we)
        elo[h] = eh + delta
        elo[a] = ea - delta
        games[h] = games.get(h, 0) + 1
        games[a] = games.get(a, 0) + 1
    return records


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(records, p):
    n = len(records)
    tr = tl = tb = ta = 0.0
    for eh, ea, adv, outcome in records:
        pH, pD, pA, _, _ = model.match_probs(eh, ea, p, home_adv=adv)
        pr = (pH, pD, pA)
        tr += rps(pr, outcome)
        tl += log_loss(pr, outcome)
        tb += brier(pr, outcome)
        ta += accuracy(pr, outcome)
    return dict(rps=tr / n, logloss=tl / n, brier=tb / n, acc=ta / n, n=n)


def base_rate_scores(records):
    c = [0, 0, 0]
    for *_, o in records:
        c[o] += 1
    n = len(records)
    base = (c[0] / n, c[1] / n, c[2] / n)
    tr = tl = tb = 0.0
    for *_, o in records:
        tr += rps(base, o); tl += log_loss(base, o); tb += brier(base, o)
    return dict(rps=tr / n, logloss=tl / n, brier=tb / n,
                freq=base), base


def reliability(records, p, bins=10):
    """Calibration of P(home win): predicted vs observed per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    sp = np.zeros(bins); so = np.zeros(bins); cnt = np.zeros(bins)
    for eh, ea, adv, outcome in records:
        pH, _, _, _, _ = model.match_probs(eh, ea, p, home_adv=adv)
        b = min(int(pH * bins), bins - 1)
        sp[b] += pH; so[b] += (1.0 if outcome == 0 else 0.0); cnt[b] += 1
    rows = []
    for b in range(bins):
        if cnt[b]:
            rows.append((sp[b] / cnt[b], so[b] / cnt[b], int(cnt[b])))
    return rows


def save_reliability_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    pred = [r[0] for r in rows]; obs = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    ax.plot(pred, obs, "o-", color="#28d17c", label="model")
    ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram — home-win probability")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def param_search(train, test):
    """Grid-search the hand-set constants on a train split, score on test."""
    grid_e = [1.2, 1.4, 1.6, 1.8, 2.0]
    grid_t = [2.4, 2.5, 2.6, 2.7, 2.8]
    grid_r = [-0.10, -0.05, 0.0]
    best, best_rps = None, 1e9
    for e in grid_e:
        for t in grid_t:
            for r in grid_r:
                p = replace(model.DEFAULTS, elo_to_goals=e, base_total=t, rho=r)
                s = evaluate(train, p)["rps"]
                if s < best_rps:
                    best_rps, best = s, (e, t, r)
    p_best = replace(model.DEFAULTS, elo_to_goals=best[0],
                     base_total=best[1], rho=best[2])
    return best, evaluate(test, p_best), evaluate(test, model.DEFAULTS)


# --------------------------------------------------------------------------- #
def run_full():
    os.makedirs(OUT, exist_ok=True)
    print("Downloading match history ...")
    rows = download_matches()
    print(f"  {len(rows):,} matches")
    records = build_eval_records(rows)
    print(f"  {len(records):,} eligible test matches (from {BURN_IN_YEAR})")

    cur = evaluate(records, model.DEFAULTS)
    base, _ = base_rate_scores(records)
    rel = reliability(records, model.DEFAULTS)
    have_png = save_reliability_plot(rel, os.path.join(OUT, "reliability.png"))

    # train/test split by chronological halves of the eligible set
    cut = len(records) // 2
    best, test_best, test_cur = param_search(records[:cut], records[cut:])

    lines = []
    lines.append("WORLD CUP 2026 PREDICTOR — BACKTEST REPORT")
    lines.append("=" * 48)
    lines.append(f"Eligible matches scored: {cur['n']:,} "
                 f"(>= {BURN_IN_YEAR}, both teams >= {MIN_PRIOR_GAMES} prior games)\n")
    lines.append("Model vs naive base-rate (lower RPS / log-loss / Brier = better):")
    lines.append(f"  {'metric':10}{'model':>12}{'base-rate':>12}")
    for k in ("rps", "logloss", "brier"):
        lines.append(f"  {k:10}{cur[k]:>12.4f}{base[k]:>12.4f}")
    lines.append(f"  {'accuracy':10}{cur['acc']:>12.3f}{'-':>12}")
    skill = (base["rps"] - cur["rps"]) / base["rps"] * 100
    lines.append(f"\n  RPS skill vs base-rate: {skill:+.1f}%  "
                 f"({'adds skill' if skill > 0 else 'no skill'})")
    lines.append("  (Good football models sit ~0.18-0.19 RPS; an uninformed "
                 "forecast ~0.22-0.23.)\n")

    lines.append("Calibration (reliability of P(home win)):")
    lines.append(f"  {'pred':>8}{'observed':>10}{'n':>8}")
    for pred, obs, n in rel:
        lines.append(f"  {pred:>8.2f}{obs:>10.2f}{n:>8}")
    lines.append("  Well-calibrated = predicted ≈ observed down the column.\n")

    lines.append("Parameter search (train on first half, score on second half):")
    lines.append(f"  current defaults  -> test RPS {test_cur['rps']:.4f}")
    lines.append(f"  best grid params  -> test RPS {test_best['rps']:.4f}")
    lines.append(f"     elo_to_goals={best[0]}, base_total={best[1]}, rho={best[2]}")
    gain = (test_cur["rps"] - test_best["rps"]) / test_cur["rps"] * 100
    lines.append(f"  out-of-sample gain from re-tuning: {gain:+.2f}%")
    lines.append("  (A small gain means the hand-set constants are already "
                 "reasonable; a large gain argues for full MLE fitting.)")
    if have_png:
        lines.append("\nSaved reliability diagram -> outputs/reliability.png")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT, "backtest_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


def selftest():
    """Validate the metric maths against hand-computed values (no network)."""
    ok = True

    def approx(a, b, t=1e-6):
        return abs(a - b) < t

    # perfect forecast
    ok &= approx(rps((1, 0, 0), 0), 0.0)
    ok &= approx(log_loss((1, 0, 0), 0), 0.0)
    ok &= approx(brier((1, 0, 0), 0), 0.0)
    # uniform forecast, home win actual
    u = (1 / 3, 1 / 3, 1 / 3)
    ok &= approx(rps(u, 0), 5 / 18)
    ok &= approx(log_loss(u, 0), math.log(3))
    ok &= approx(brier(u, 0), 6 / 9)
    # RPS rewards ordering: a near-miss (away) costs more than a draw miss
    ok &= rps((0.2, 0.2, 0.6), 0) > rps((0.2, 0.6, 0.2), 0)
    # synthetic eligible records run through evaluate + match_probs
    recs = [(1800, 1500, 65.0, 0), (1500, 1800, 0.0, 2), (1600, 1600, 65.0, 1)]
    res = evaluate(recs, model.DEFAULTS)
    ok &= 0.0 < res["rps"] < 0.4 and res["n"] == 3
    base, freq = base_rate_scores(recs)
    ok &= approx(sum(freq), 1.0)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    run_full()
