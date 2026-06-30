"""
OPTIONAL machine-learning calibration layer.

Trains two gradient-boosting regressors to predict home and away goals from an
Elo difference, using ~150 years of international results. It reconstructs a
running Elo from match history so every historical match has a pre-match Elo
gap as its feature, then learns the goals-from-Elo relationship empirically.

This is a *blend partner* for the statistical engine, not a replacement. It is
OFF by default (ml_weight = 0). After training, raise the blend weight in the
Streamlit sidebar to mix it in.

Requires internet (available on Streamlit Cloud) and scikit-learn.

Run:   python train_model.py
Output: model.pkl  (loaded automatically by app.py if present)

Caveat: the reconstructed Elo scale is similar to, but not identical to,
eloratings.net. The blend is therefore an approximation; treat the pure
statistical engine as the trustworthy default.
"""
import os
import pickle

import numpy as np
import pandas as pd

RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")


def build_elo_features(df, base=1500.0, k=30.0, home_field=60.0):
    """Replay history chronologically, maintaining an Elo per team, and emit
    pre-match (elo_diff, neutral) features with (home_goals, away_goals) targets."""
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    elo = {}
    rows = []
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        eh = elo.get(h, base)
        ea = elo.get(a, base)
        adv = 0.0 if getattr(r, "neutral", False) else home_field
        dr = (eh + adv) - ea
        we = 1.0 / (10 ** (-dr / 400.0) + 1)

        rows.append((dr, int(bool(getattr(r, "neutral", False))),
                     r.home_score, r.away_score))

        # Elo update on actual result
        if r.home_score > r.away_score:
            sh = 1.0
        elif r.home_score < r.away_score:
            sh = 0.0
        else:
            sh = 0.5
        margin = 1.0 + np.log1p(abs(r.home_score - r.away_score))
        delta = k * margin * (sh - we)
        elo[h] = eh + delta
        elo[a] = ea - delta

    feat = pd.DataFrame(rows, columns=["elo_diff", "neutral",
                                       "home_goals", "away_goals"])
    return feat, elo


class MLModel:
    """Wraps two regressors and exposes predict_goals(feats)->(lh, la)."""

    def __init__(self, reg_home, reg_away):
        self.reg_home = reg_home
        self.reg_away = reg_away

    def predict_goals(self, feats):
        elo_diff = feats["elo_diff"]
        neutral = 0 if feats.get("home_host") else 1
        x = np.array([[elo_diff, neutral]])
        lh = float(self.reg_home.predict(x)[0])
        la = float(self.reg_away.predict(x)[0])
        return max(lh, 0.1), max(la, 0.1)

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"home": self.reg_home, "away": self.reg_away}, f)

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            return cls(d["home"], d["away"])
        except Exception:
            return None


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor

    print("Downloading international match history ...")
    df = pd.read_csv(RESULTS_URL)
    print(f"  {len(df):,} matches")

    feat, _ = build_elo_features(df)
    print(f"Built {len(feat):,} training rows")

    X = feat[["elo_diff", "neutral"]].values
    reg_home = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_depth=4).fit(X, feat.home_goals)
    reg_away = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_depth=4).fit(X, feat.away_goals)

    MLModel(reg_home, reg_away).save()
    print(f"Saved model -> {MODEL_PATH}")
    # quick sanity print
    m = MLModel(reg_home, reg_away)
    for d in (0, 200, 500, -300):
        lh, la = m.predict_goals({"elo_diff": d, "home_host": False})
        print(f"  elo_diff {d:+5d} -> {lh:.2f} - {la:.2f}")


if __name__ == "__main__":
    main()
