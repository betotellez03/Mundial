"""
Elo rating model.

Implements a World-Football-Elo-style rating system:

* ratings updated chronologically over the match history,
* K-factor scaled by margin of victory,
* home-field / host advantage applied to the expected score,
* 1X2 probabilities derived from the rating difference with an explicit draw
  model (draws are most likely when two teams are evenly matched).

The model exposes:
* ``fit(matches)``            – learn ratings from history (no leakage; ratings
                                only ever use information up to each match),
* ``rating(team)``            – current rating,
* ``predict_proba(home, away, neutral, host)`` -> (P(H), P(D), P(A)),
* ``expected_goals(...)``     – a convenience supremacy -> goals mapping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from . import config


def expected_score(rating_a: float, rating_b: float) -> float:
    """Classic Elo expectation that A beats B (0..1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _k_factor(goal_diff: int) -> float:
    """Scale K by margin of victory (draw/1-goal = base K)."""
    g = abs(goal_diff)
    if not config.ELO_GOAL_MULTIPLIER or g <= 1:
        return config.ELO_K
    if g == 2:
        return config.ELO_K * 1.5
    # 3+ goals
    return config.ELO_K * (1.75 + (g - 3) / 8.0)


@dataclass
class EloModel:
    ratings: dict[str, float] = field(default_factory=dict)
    # Width of the "draw zone": larger => more draws predicted for close games.
    draw_width: float = 0.30

    # ------------------------------------------------------------------ fit
    def _get(self, team: str) -> float:
        return self.ratings.get(team, config.ELO_INITIAL)

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        """Replay matches in chronological order to learn ratings."""
        self.ratings = {}
        matches = matches.sort_values("date")
        for row in matches.itertuples(index=False):
            home, away = row.home_team, row.away_team
            rh, ra = self._get(home), self._get(away)

            # Home advantage only when not on neutral ground.
            ha = 0.0 if getattr(row, "neutral", 0) else config.ELO_HOME_ADVANTAGE
            exp_home = expected_score(rh + ha, ra)

            if row.home_score > row.away_score:
                score_home = 1.0
            elif row.home_score < row.away_score:
                score_home = 0.0
            else:
                score_home = 0.5

            k = _k_factor(row.home_score - row.away_score)
            delta = k * (score_home - exp_home)
            self.ratings[home] = rh + delta
            self.ratings[away] = ra - delta
        return self

    # -------------------------------------------------------------- predict
    def rating(self, team: str) -> float:
        return self._get(team)

    def _effective_ratings(self, home, away, neutral, host_side):
        """Apply home + host advantage to produce effective ratings."""
        rh, ra = self._get(home), self._get(away)
        if not neutral:
            rh += config.ELO_HOME_ADVANTAGE
        # Host bonus: at the 2026 tournament every match is "neutral", but the
        # three hosts still get a documented boost when they play.
        if host_side == "home" or home in config.HOST_TEAMS:
            rh += config.HOST_ELO_BONUS
        if host_side == "away" or away in config.HOST_TEAMS:
            ra += config.HOST_ELO_BONUS
        return rh, ra

    def predict_proba(self, home: str, away: str, neutral: bool = True,
                      host_side: str | None = None) -> tuple[float, float, float]:
        """Return (P(home win), P(draw), P(away win)).

        The draw probability follows a bell shape in the rating difference: it
        peaks for evenly matched teams and decays for mismatches.
        """
        rh, ra = self._effective_ratings(home, away, neutral, host_side)
        p_home_no_draw = expected_score(rh, ra)  # 0..1 ignoring draws

        diff = (rh - ra) / 400.0
        # Draw probability model (empirically ~0.25 peak for even games).
        p_draw = self.draw_width * math.exp(-(diff ** 2) / (2 * 0.5 ** 2))
        p_draw = min(p_draw, 0.5)

        # Split the remaining mass according to the no-draw expectation.
        remaining = 1.0 - p_draw
        p_home = remaining * p_home_no_draw
        p_away = remaining * (1.0 - p_home_no_draw)
        return p_home, p_draw, p_away

    def expected_goals(self, home: str, away: str, neutral: bool = True,
                       host_side: str | None = None) -> tuple[float, float]:
        """Map the Elo supremacy onto expected goals for each side."""
        rh, ra = self._effective_ratings(home, away, neutral, host_side)
        diff = (rh - ra) / 400.0
        lam_home = config.POISSON_BASE_HOME_GOALS * (10 ** (diff / 2))
        lam_away = config.POISSON_BASE_AWAY_GOALS * (10 ** (-diff / 2))
        return float(min(lam_home, 6.0)), float(min(lam_away, 6.0))

    # ----------------------------------------------------------- persistence
    def to_frame(self) -> pd.DataFrame:
        return (pd.DataFrame({"team": list(self.ratings), "elo": list(self.ratings.values())})
                .sort_values("elo", ascending=False).reset_index(drop=True))

    def save(self, path=config.ELO_RATINGS_CSV) -> None:
        self.to_frame().to_csv(path, index=False)


def train_elo(matches: pd.DataFrame) -> EloModel:
    model = EloModel().fit(matches)
    model.save()
    print(f"[elo_model] trained on {len(matches)} matches; "
          f"{len(model.ratings)} teams rated")
    return model
