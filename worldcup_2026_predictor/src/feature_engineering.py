"""
Feature engineering for the machine-learning 1X2 model.

Every feature for a given match is computed using **only information available
before that match** (rolling/expanding windows over the chronologically sorted
history).  This is what prevents data leakage and makes the later temporal
train/test split meaningful.

Features produced per match (home perspective):
* ``elo_diff``         – pre-match Elo difference (home - away).
* ``elo_home/away``    – pre-match ratings.
* ``form_home/away``   – avg points/game over the team's last N matches.
* ``gf_home/away``     – avg goals scored over last N matches.
* ``ga_home/away``     – avg goals conceded over last N matches.
* ``rest_*``           – (proxy) matches played recently.
* ``is_home_host``     – host-nation flag.
* ``neutral``          – neutral-venue flag.
"""
from __future__ import annotations

from collections import deque, defaultdict

import pandas as pd

from . import config
from . import elo_model

FORM_WINDOW = 5  # number of recent matches used for rolling form features

FEATURE_COLUMNS = [
    "elo_diff", "elo_home", "elo_away",
    "form_home", "form_away",
    "gf_home", "gf_away", "ga_home", "ga_away",
    "is_home_host", "is_away_host", "neutral",
]


def _points(team_score: int, opp_score: int) -> int:
    if team_score > opp_score:
        return config.WIN_POINTS
    if team_score == opp_score:
        return config.DRAW_POINTS
    return config.LOSS_POINTS


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix (one row per match) without look-ahead leakage."""
    matches = matches.sort_values("date").reset_index(drop=True)

    # Replay Elo so we can store the *pre-match* rating for each game.
    elo = elo_model.EloModel()

    # Rolling history per team.
    recent_points: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    recent_gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    recent_ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    def avg(dq, default):
        return sum(dq) / len(dq) if len(dq) else default

    rows = []
    for r in matches.itertuples(index=False):
        home, away = r.home_team, r.away_team
        rh, ra = elo.rating(home), elo.rating(away)

        rows.append({
            "date": r.date,
            "home_team": home,
            "away_team": away,
            "elo_home": rh,
            "elo_away": ra,
            "elo_diff": rh - ra,
            "form_home": avg(recent_points[home], 1.0),
            "form_away": avg(recent_points[away], 1.0),
            "gf_home": avg(recent_gf[home], 1.2),
            "gf_away": avg(recent_gf[away], 1.2),
            "ga_home": avg(recent_ga[home], 1.2),
            "ga_away": avg(recent_ga[away], 1.2),
            "is_home_host": int(home in config.HOST_TEAMS),
            "is_away_host": int(away in config.HOST_TEAMS),
            "neutral": int(getattr(r, "neutral", 0)),
            "result": r.result,
        })

        # ---- AFTER recording features, update state with this match's result.
        ha = 0.0 if getattr(r, "neutral", 0) else config.ELO_HOME_ADVANTAGE
        exp_home = elo_model.expected_score(rh + ha, ra)
        if r.home_score > r.away_score:
            sc = 1.0
        elif r.home_score < r.away_score:
            sc = 0.0
        else:
            sc = 0.5
        k = elo_model._k_factor(r.home_score - r.away_score)
        delta = k * (sc - exp_home)
        elo.ratings[home] = rh + delta
        elo.ratings[away] = ra - delta

        recent_points[home].append(_points(r.home_score, r.away_score))
        recent_points[away].append(_points(r.away_score, r.home_score))
        recent_gf[home].append(r.home_score)
        recent_gf[away].append(r.away_score)
        recent_ga[home].append(r.away_score)
        recent_ga[away].append(r.home_score)

    return pd.DataFrame(rows)


def run(matches: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(matches)
    feats.to_csv(config.FEATURES_CSV, index=False)
    print(f"[feature_engineering] wrote {config.FEATURES_CSV} ({len(feats)} rows)")
    return feats
