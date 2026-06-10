"""
Match predictor / ensemble model.

Combines the three base models (Elo, Poisson, ML) into a single set of 1X2
probabilities via a weighted average (weights in ``config.ENSEMBLE_WEIGHTS``).

It also exposes the **single-match prediction** entry point used by both the
Streamlit app and the tournament simulator:

    predictor.predict_match("Brazil", "Argentina")

Returns a dict with each model's probabilities, the ensemble probabilities, the
expected goals and the most likely scoreline.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from . import feature_engineering as fe
from .elo_model import EloModel
from .poisson_model import PoissonModel
from .ml_model import MLModel


def build_team_state(clean_matches: pd.DataFrame) -> dict[str, dict]:
    """Compute each team's latest rolling form / goals (for ML features)."""
    clean_matches = clean_matches.sort_values("date")
    pts = defaultdict(lambda: deque(maxlen=fe.FORM_WINDOW))
    gf = defaultdict(lambda: deque(maxlen=fe.FORM_WINDOW))
    ga = defaultdict(lambda: deque(maxlen=fe.FORM_WINDOW))

    def _pts(a, b):
        return config.WIN_POINTS if a > b else (config.DRAW_POINTS if a == b else config.LOSS_POINTS)

    for r in clean_matches.itertuples(index=False):
        pts[r.home_team].append(_pts(r.home_score, r.away_score))
        pts[r.away_team].append(_pts(r.away_score, r.home_score))
        gf[r.home_team].append(r.home_score)
        gf[r.away_team].append(r.away_score)
        ga[r.home_team].append(r.away_score)
        ga[r.away_team].append(r.home_score)

    def avg(dq, d):
        return sum(dq) / len(dq) if len(dq) else d

    state = {}
    teams = set(clean_matches["home_team"]).union(clean_matches["away_team"])
    for t in teams:
        state[t] = {
            "form": avg(pts[t], 1.0),
            "gf": avg(gf[t], 1.2),
            "ga": avg(ga[t], 1.2),
        }
    return state


@dataclass
class MatchPredictor:
    elo: EloModel
    poisson: PoissonModel
    ml: MLModel
    team_state: dict[str, dict]
    weights: dict = None

    def __post_init__(self):
        self.weights = self.weights or dict(config.ENSEMBLE_WEIGHTS)

    # ---------------------------------------------------------- ML features
    def _feature_row(self, home: str, away: str, neutral: bool) -> dict:
        sh = self.team_state.get(home, {"form": 1.0, "gf": 1.2, "ga": 1.2})
        sa = self.team_state.get(away, {"form": 1.0, "gf": 1.2, "ga": 1.2})
        rh, ra = self.elo.rating(home), self.elo.rating(away)
        return {
            "elo_home": rh, "elo_away": ra, "elo_diff": rh - ra,
            "form_home": sh["form"], "form_away": sa["form"],
            "gf_home": sh["gf"], "gf_away": sa["gf"],
            "ga_home": sh["ga"], "ga_away": sa["ga"],
            "is_home_host": int(home in config.HOST_TEAMS),
            "is_away_host": int(away in config.HOST_TEAMS),
            "neutral": int(neutral),
        }

    # ----------------------------------------------------------- prediction
    def predict_match(self, home: str, away: str, neutral: bool = True) -> dict:
        """Predict a single match. World-Cup matches are neutral by default."""
        elo_p = self.elo.predict_proba(home, away, neutral=neutral)
        poi_p = self.poisson.predict_proba(home, away, neutral=neutral)
        ml_p = self.ml.predict_proba_match(self._feature_row(home, away, neutral))

        w = self.weights
        ens = tuple(
            w["elo"] * e + w["poisson"] * p + w["ml"] * m
            for e, p, m in zip(elo_p, poi_p, ml_p)
        )
        ens = tuple(x / sum(ens) for x in ens)  # normalise

        eg_home, eg_away = self.poisson.expected_goals(home, away, neutral=neutral)
        score = self.poisson.most_likely_score(home, away, neutral=neutral)

        return {
            "home": home, "away": away,
            "elo": elo_p, "poisson": poi_p, "ml": ml_p,
            "ensemble": ens,
            "p_home": ens[0], "p_draw": ens[1], "p_away": ens[2],
            "exp_goals_home": eg_home, "exp_goals_away": eg_away,
            "likely_score": score,
        }

    def predict_proba(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        return self.predict_match(home, away, neutral)["ensemble"]

    def knockout_win_prob(self, home: str, away: str) -> float:
        """P(home advances) for a knockout match (draws resolved on penalties).

        A drawn 90'+ET is modelled as a near-coin-flip slightly favouring the
        stronger side, so the draw mass is split using the no-draw odds.
        """
        p_home, p_draw, p_away = self.predict_proba(home, away, neutral=True)
        denom = p_home + p_away
        share_home = 0.5 if denom == 0 else p_home / denom
        # Penalties: dampen toward 50/50.
        pen_home = 0.5 + (share_home - 0.5) * 0.6
        return p_home + p_draw * pen_home


def build_predictor(elo, poisson, ml, clean_matches) -> MatchPredictor:
    return MatchPredictor(
        elo=elo, poisson=poisson, ml=ml,
        team_state=build_team_state(clean_matches),
    )
