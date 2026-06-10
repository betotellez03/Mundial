"""
Tournament simulator (official FIFA World Cup 2026 format).

48 teams -> 12 groups of 4 -> top 2 of each group + 8 best third-placed teams
-> Round of 32 -> R16 -> QF -> SF -> Final.  Host nations (Mexico, USA, Canada)
carry a documented advantage that is baked into the base-model predictions.

Performance
-----------
Running >=10,000 Monte-Carlo tournaments means evaluating millions of matches.
To stay fast we **pre-compute**, once, for every ordered team pair:
* expected goals (Poisson) -> used to sample group-stage scorelines,
* ensemble 1X2 probabilities -> used to resolve knockout ties.
The Monte-Carlo loop then only does cheap dict look-ups and RNG draws.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from . import config
from . import data_collection
from .match_predictor import MatchPredictor

# Knockout stage labels in order.
KO_STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


# --------------------------------------------------------------------------- #
# Bracket seeding helpers
# --------------------------------------------------------------------------- #
def seeding_order(n: int) -> list[int]:
    """Standard single-elimination seeding so #1 and #2 can only meet in the final."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        new = []
        for x in order:
            new.append(x)
            new.append(m + 1 - x)
        order = new
    return order


class TournamentSimulator:
    def __init__(self, predictor: MatchPredictor, groups: pd.DataFrame | None = None):
        self.predictor = predictor
        self.groups = groups if groups is not None else data_collection.load_groups()
        self.group_map = {g: sub["team"].tolist()
                          for g, sub in self.groups.groupby("group")}
        self.teams = self.groups["team"].tolist()
        self._eg_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._ko_cache: dict[tuple[str, str], float] = {}
        self._precompute()

    # ----------------------------------------------------------- pre-compute
    def _precompute(self) -> None:
        """Cache expected goals and knockout-advance probabilities per pairing."""
        for a, b in combinations(self.teams, 2):
            for home, away in ((a, b), (b, a)):
                self._eg_cache[(home, away)] = \
                    self.predictor.poisson.expected_goals(home, away, neutral=True)
                self._ko_cache[(home, away)] = \
                    self.predictor.knockout_win_prob(home, away)

    # ------------------------------------------------------------ group stage
    def _sample_score(self, home: str, away: str, rng) -> tuple[int, int]:
        lam_h, lam_a = self._eg_cache[(home, away)]
        return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))

    def _simulate_group(self, teams: list[str], rng) -> list[dict]:
        """Round-robin; return standings sorted by the FIFA tie-break order."""
        stats = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                     "GF": 0, "GA": 0, "Pts": 0} for t in teams}
        for home, away in combinations(teams, 2):
            hg, ag = self._sample_score(home, away, rng)
            stats[home]["GF"] += hg; stats[home]["GA"] += ag
            stats[away]["GF"] += ag; stats[away]["GA"] += hg
            stats[home]["P"] += 1; stats[away]["P"] += 1
            if hg > ag:
                stats[home]["Pts"] += config.WIN_POINTS; stats[home]["W"] += 1; stats[away]["L"] += 1
            elif hg < ag:
                stats[away]["Pts"] += config.WIN_POINTS; stats[away]["W"] += 1; stats[home]["L"] += 1
            else:
                stats[home]["Pts"] += config.DRAW_POINTS; stats[away]["Pts"] += config.DRAW_POINTS
                stats[home]["D"] += 1; stats[away]["D"] += 1

        rows = list(stats.values())
        for r in rows:
            r["GD"] = r["GF"] - r["GA"]
            r["elo"] = self.predictor.elo.rating(r["team"])
        # Tie-breakers: points, goal difference, goals for, then Elo (proxy for
        # the lower-priority FIFA criteria such as fair play / drawing of lots).
        rows.sort(key=lambda r: (r["Pts"], r["GD"], r["GF"], r["elo"]), reverse=True)
        return rows

    # --------------------------------------------------------- knockout stage
    def _knockout_winner(self, a: str, b: str, rng) -> str:
        return a if rng.random() < self._ko_cache[(a, b)] else b

    # --------------------------------------------------- single full tournament
    def simulate_once(self, rng, record: bool = False) -> dict:
        """Play one entire tournament; return per-team stage reached.

        If ``record`` is True, also return the group tables and the bracket of
        this single run (used to export a representative example).
        """
        winners, runners, thirds = [], [], []
        group_tables = []

        for g in config.GROUP_LABELS:
            standings = self._simulate_group(self.group_map[g], rng)
            if record:
                for pos, r in enumerate(standings, start=1):
                    group_tables.append({"group": g, "position": pos, **r})
            winners.append((g, standings[0]))
            runners.append((g, standings[1]))
            thirds.append((g, standings[2]))

        # Best 8 third-placed teams.
        thirds_sorted = sorted(
            thirds,
            key=lambda gr: (gr[1]["Pts"], gr[1]["GD"], gr[1]["GF"], gr[1]["elo"]),
            reverse=True,
        )
        best_thirds = thirds_sorted[:config.N_BEST_THIRDS]

        # Stage tracking.
        reached = {}

        # Build a qualification ranking: winners (best), then runners, then thirds.
        def qrank(entry, base):
            r = entry[1]
            return (base, r["Pts"], r["GD"], r["GF"], r["elo"])

        ranked_entries = (
            sorted(winners, key=lambda e: qrank(e, 3), reverse=True)
            + sorted(runners, key=lambda e: qrank(e, 2), reverse=True)
            + sorted(best_thirds, key=lambda e: qrank(e, 1), reverse=True)
        )
        ranked_teams = [e[1]["team"] for e in ranked_entries]

        for t in ranked_teams:
            reached[t] = "R32"

        # Seed into the 32-team bracket.
        order = seeding_order(config.N_KNOCKOUT_TEAMS)
        bracket_teams = [ranked_teams[o - 1] for o in order]

        bracket_log = []
        current = bracket_teams
        for stage_idx, stage in enumerate(KO_STAGES[:-1]):  # R32..Final
            next_round = []
            for i in range(0, len(current), 2):
                a, b = current[i], current[i + 1]
                w = self._knockout_winner(a, b, rng)
                loser = b if w == a else a
                # Loser's deepest stage is the current one; winner advances.
                next_stage = KO_STAGES[stage_idx + 1]
                reached[w] = next_stage
                if record:
                    bracket_log.append({"stage": stage, "team_a": a, "team_b": b,
                                        "winner": w, "loser": loser})
                next_round.append(w)
            current = next_round
        champion = current[0]

        out = {"reached": reached, "champion": champion}
        if record:
            out["group_tables"] = group_tables
            out["bracket"] = bracket_log
        return out

    # --------------------------------------------------------- Monte-Carlo run
    def run_monte_carlo(self, n_sims: int = config.N_SIMULATIONS,
                        seed: int = config.RANDOM_SEED) -> dict:
        """Run ``n_sims`` tournaments and aggregate probabilities per team."""
        rng = np.random.default_rng(seed)
        stage_rank = {s: i for i, s in enumerate(KO_STAGES)}

        # counts[team][stage] = number of sims the team reached AT LEAST stage.
        counts = {t: defaultdict(int) for t in self.teams}
        champion_counts = defaultdict(int)
        # Expected group points accumulator.
        group_pts = defaultdict(float)

        # Capture one representative run for the example exports.
        example = self.simulate_once(rng, record=True)

        for s in range(n_sims):
            res = self.simulate_once(rng)
            for team, deepest in res["reached"].items():
                # Increment every stage up to and including the deepest reached.
                deep_rank = stage_rank[deepest]
                for st, rk in stage_rank.items():
                    if rk <= deep_rank:
                        counts[team][st] += 1
            champion_counts[res["champion"]] += 1

        # Add example group points to expected (cheap, single run – purely
        # illustrative; the probabilities below are the rigorous output).
        for row in example["group_tables"]:
            group_pts[row["team"]] += row["Pts"]

        summary = []
        for t in self.teams:
            row = {"team": t}
            for st in KO_STAGES:
                row[f"P_{st}"] = counts[t][st] / n_sims
            row["P_Champion"] = champion_counts[t] / n_sims
            summary.append(row)
        summary_df = (pd.DataFrame(summary)
                      .sort_values("P_Champion", ascending=False)
                      .reset_index(drop=True))

        return {
            "n_sims": n_sims,
            "summary": summary_df,
            "example_group_tables": pd.DataFrame(example["group_tables"]),
            "example_bracket": pd.DataFrame(example["bracket"]),
        }


# --------------------------------------------------------------------------- #
# Convenience exports
# --------------------------------------------------------------------------- #
def export_results(mc: dict) -> None:
    """Write champion probabilities, group tables and bracket to ``outputs/``."""
    mc["summary"].to_csv(config.CHAMPION_PROBS_CSV, index=False)
    mc["example_group_tables"].to_csv(config.GROUP_TABLES_CSV, index=False)
    mc["example_bracket"].to_csv(config.BRACKET_CSV, index=False)
    print(f"[tournament_simulator] wrote {config.CHAMPION_PROBS_CSV}")
    print(f"[tournament_simulator] wrote {config.GROUP_TABLES_CSV}")
    print(f"[tournament_simulator] wrote {config.BRACKET_CSV}")
