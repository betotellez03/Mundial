"""
Poisson expected-goals model.

Estimates per-team **attack** and **defence** strengths from the goals scored
and conceded across the match history, then models each match as two
(approximately independent) Poisson processes:

    home goals ~ Poisson(mu_home)
    away goals ~ Poisson(mu_away)

with

    mu_home = base_home * attack[home] * defence[away] * host_factor
    mu_away = base_away * attack[away] * defence[home] * host_factor

From the joint score grid we derive:
* 1X2 probabilities,
* most likely scoreline,
* expected goals for each side.

Strengths are computed as ratios to the league average, which is the standard
Dixon–Coles / Maher independent-Poisson formulation (without the low-score
correction, to keep the MVP dependency-light – only scipy is required).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import config


@dataclass
class PoissonModel:
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    base_home: float = config.POISSON_BASE_HOME_GOALS
    base_away: float = config.POISSON_BASE_AWAY_GOALS
    max_goals: int = config.POISSON_MAX_GOALS

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        """Estimate attack/defence ratios relative to the league mean."""
        scored = []
        conceded = []
        for r in matches.itertuples(index=False):
            scored.append((r.home_team, r.home_score))
            scored.append((r.away_team, r.away_score))
            conceded.append((r.home_team, r.away_score))
            conceded.append((r.away_team, r.home_score))

        gf = pd.DataFrame(scored, columns=["team", "g"]).groupby("team")["g"].mean()
        ga = pd.DataFrame(conceded, columns=["team", "g"]).groupby("team")["g"].mean()
        league_avg = float(np.mean([r.home_score + r.away_score for r in
                                    matches.itertuples(index=False)]) / 2.0)
        league_avg = max(league_avg, 0.5)

        # attack > 1 => scores more than average; defence < 1 => concedes less.
        self.attack = (gf / league_avg).to_dict()
        self.defence = (ga / league_avg).to_dict()
        self._league_avg = league_avg
        return self

    # ----------------------------------------------------------- parameters
    def _strength(self, d: dict, team: str) -> float:
        return float(d.get(team, 1.0))

    def expected_goals(self, home: str, away: str, neutral: bool = True,
                       host_side: str | None = None) -> tuple[float, float]:
        atk_h, def_h = self._strength(self.attack, home), self._strength(self.defence, home)
        atk_a, def_a = self._strength(self.attack, away), self._strength(self.defence, away)

        base_h = self.base_home if not neutral else (self.base_home + self.base_away) / 2
        base_a = self.base_away if not neutral else (self.base_home + self.base_away) / 2

        mu_home = base_h * atk_h * def_a
        mu_away = base_a * atk_a * def_h

        # Host advantage scales expected goals modestly.
        host_factor = 1.0 + config.HOST_ELO_BONUS / 400.0
        if host_side == "home" or home in config.HOST_TEAMS:
            mu_home *= host_factor
        if host_side == "away" or away in config.HOST_TEAMS:
            mu_away *= host_factor

        return float(np.clip(mu_home, 0.15, 6.0)), float(np.clip(mu_away, 0.15, 6.0))

    # ------------------------------------------------------------- score grid
    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     host_side: str | None = None) -> np.ndarray:
        """Joint probability matrix P(home=i, away=j)."""
        mu_home, mu_away = self.expected_goals(home, away, neutral, host_side)
        ks = np.arange(0, self.max_goals + 1)
        ph = poisson.pmf(ks, mu_home)
        pa = poisson.pmf(ks, mu_away)
        grid = np.outer(ph, pa)
        grid /= grid.sum()  # renormalise after truncation
        return grid

    def predict_proba(self, home: str, away: str, neutral: bool = True,
                      host_side: str | None = None) -> tuple[float, float, float]:
        grid = self.score_matrix(home, away, neutral, host_side)
        p_home = float(np.tril(grid, -1).sum())  # home goals > away goals
        p_away = float(np.triu(grid, 1).sum())   # away goals > home goals
        p_draw = float(np.trace(grid))
        return p_home, p_draw, p_away

    def most_likely_score(self, home: str, away: str, neutral: bool = True,
                          host_side: str | None = None) -> tuple[int, int]:
        grid = self.score_matrix(home, away, neutral, host_side)
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        return int(i), int(j)

    def sample_score(self, home: str, away: str, rng: np.random.Generator,
                     neutral: bool = True, host_side: str | None = None) -> tuple[int, int]:
        """Draw a random scoreline – used by the Monte-Carlo simulator."""
        mu_home, mu_away = self.expected_goals(home, away, neutral, host_side)
        return int(rng.poisson(mu_home)), int(rng.poisson(mu_away))


def train_poisson(matches: pd.DataFrame) -> PoissonModel:
    model = PoissonModel().fit(matches)
    print(f"[poisson_model] estimated strengths for {len(model.attack)} teams")
    return model
