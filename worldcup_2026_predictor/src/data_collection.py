"""
Data collection / bootstrapping.

This module makes the project runnable *immediately* even when no real data is
available.  It writes three CSV templates into ``data/raw`` if they do not
already exist:

* ``teams.csv``        – the 48 qualified nations, confederation, seed Elo, host flag.
* ``groups_2026.csv``  – the 12 groups of 4 (a plausible draw).
* ``matches.csv``      – a synthetic but *realistic* history of international
                         results used to train the ML and Poisson models.

IMPORTANT
---------
The historical results are **simulated from each team's latent strength**, not
hand-picked outcomes.  Nothing about the 2026 predictions is hard-coded: the
models learn from this data exactly as they would from real results.  To use
real data, simply replace the files in ``data/raw`` with your own (same
columns) – the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# The 48-team field.  Elo values are rough public-domain estimates used only as
# a starting point; the Elo model re-rates every team from the match history.
# (team, confederation, seed_elo)
# ---------------------------------------------------------------------------
_TEAMS = [
    # UEFA
    ("France", "UEFA", 2010), ("Spain", "UEFA", 2040), ("England", "UEFA", 1980),
    ("Portugal", "UEFA", 1970), ("Netherlands", "UEFA", 1950), ("Belgium", "UEFA", 1930),
    ("Italy", "UEFA", 1940), ("Croatia", "UEFA", 1900), ("Germany", "UEFA", 1960),
    ("Switzerland", "UEFA", 1830), ("Denmark", "UEFA", 1860), ("Austria", "UEFA", 1840),
    ("Ukraine", "UEFA", 1790), ("Turkey", "UEFA", 1810), ("Poland", "UEFA", 1800),
    ("Serbia", "UEFA", 1820),
    # CONMEBOL
    ("Argentina", "CONMEBOL", 2080), ("Brazil", "CONMEBOL", 2020),
    ("Uruguay", "CONMEBOL", 1920), ("Colombia", "CONMEBOL", 1910),
    ("Ecuador", "CONMEBOL", 1820), ("Paraguay", "CONMEBOL", 1740),
    ("Bolivia", "CONMEBOL", 1620),
    # CONCACAF (includes hosts)
    ("United States", "CONCACAF", 1800), ("Mexico", "CONCACAF", 1820),
    ("Canada", "CONCACAF", 1790), ("Costa Rica", "CONCACAF", 1690),
    ("Panama", "CONCACAF", 1670), ("Jamaica", "CONCACAF", 1650),
    # CAF
    ("Morocco", "CAF", 1890), ("Senegal", "CAF", 1870), ("Egypt", "CAF", 1790),
    ("Nigeria", "CAF", 1810), ("Algeria", "CAF", 1800), ("Cameroon", "CAF", 1750),
    ("Ivory Coast", "CAF", 1760), ("Tunisia", "CAF", 1730), ("Ghana", "CAF", 1720),
    ("DR Congo", "CAF", 1700),
    # AFC
    ("Japan", "AFC", 1840), ("Iran", "AFC", 1810), ("South Korea", "AFC", 1820),
    ("Australia", "AFC", 1760), ("Saudi Arabia", "AFC", 1680),
    ("Qatar", "AFC", 1670), ("Iraq", "AFC", 1640), ("Uzbekistan", "AFC", 1660),
    # OFC
    ("New Zealand", "OFC", 1610),
]


def _build_teams_frame() -> pd.DataFrame:
    df = pd.DataFrame(_TEAMS, columns=["team", "confederation", "seed_elo"])
    df["is_host"] = df["team"].isin(config.HOST_TEAMS).astype(int)
    assert len(df) == config.N_TEAMS, f"expected {config.N_TEAMS} teams, got {len(df)}"
    return df


def _build_groups_frame(teams: pd.DataFrame) -> pd.DataFrame:
    """Draw 12 groups of 4 using pot-based seeding.

    Pot 1 are the strongest teams (hosts auto-seeded), and so on.  Each group
    gets one team from each pot – a simplified version of the real draw, good
    enough for simulation and easily replaced by the official draw.
    """
    ordered = teams.sort_values("seed_elo", ascending=False).reset_index(drop=True)
    # Ensure hosts sit in pot 1 regardless of Elo.
    hosts = ordered[ordered["is_host"] == 1]
    non_hosts = ordered[ordered["is_host"] == 0]
    pot1 = pd.concat([hosts, non_hosts]).head(config.N_GROUPS)
    rest = ordered[~ordered["team"].isin(pot1["team"])]
    pot2 = rest.iloc[0:config.N_GROUPS]
    pot3 = rest.iloc[config.N_GROUPS:2 * config.N_GROUPS]
    pot4 = rest.iloc[2 * config.N_GROUPS:3 * config.N_GROUPS]

    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for pot in (pot1, pot2, pot3, pot4):
        teams_in_pot = pot["team"].tolist()
        rng.shuffle(teams_in_pot)
        for g, team in zip(config.GROUP_LABELS, teams_in_pot):
            rows.append({"group": g, "team": team})
    return pd.DataFrame(rows).sort_values(["group", "team"]).reset_index(drop=True)


def _simulate_match_goals(rng, elo_home, elo_away) -> tuple[int, int]:
    """Draw a realistic scoreline from two teams' latent Elo strengths."""
    # Expected goal supremacy grows with the Elo difference.
    diff = (elo_home - elo_away) / 400.0
    lam_home = config.POISSON_BASE_HOME_GOALS * (10 ** (diff / 2))
    lam_away = config.POISSON_BASE_AWAY_GOALS * (10 ** (-diff / 2))
    lam_home = float(np.clip(lam_home, 0.15, 6.0))
    lam_away = float(np.clip(lam_away, 0.15, 6.0))
    return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))


def _build_matches_frame(teams: pd.DataFrame, n_matches: int = 4000) -> pd.DataFrame:
    """Generate a synthetic international match history spanning several years."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    elo = dict(zip(teams["team"], teams["seed_elo"].astype(float)))
    names = teams["team"].tolist()

    start = np.datetime64("2018-01-01")
    rows = []
    for i in range(n_matches):
        home, away = rng.choice(names, size=2, replace=False)
        hg, ag = _simulate_match_goals(rng, elo[home] + 50, elo[away])  # +50 home edge
        # Spread matches roughly evenly across ~8 years.
        day = int(i / n_matches * 365 * 8)
        date = start + np.timedelta64(day, "D")
        rows.append({
            "date": np.datetime_as_string(date, unit="D"),
            "home_team": home,
            "away_team": away,
            "home_score": hg,
            "away_score": ag,
            "tournament": "Friendly" if rng.random() < 0.4 else "Qualifier",
            "neutral": int(rng.random() < 0.2),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def generate_sample_data(force: bool = False) -> None:
    """Write the three raw CSV templates if they are missing (or ``force``)."""
    teams = _build_teams_frame()

    if force or not config.TEAMS_CSV.exists():
        teams.to_csv(config.TEAMS_CSV, index=False)
        print(f"[data_collection] wrote {config.TEAMS_CSV}")

    if force or not config.GROUPS_CSV.exists():
        _build_groups_frame(teams).to_csv(config.GROUPS_CSV, index=False)
        print(f"[data_collection] wrote {config.GROUPS_CSV}")

    if force or not config.MATCHES_CSV.exists():
        _build_matches_frame(teams).to_csv(config.MATCHES_CSV, index=False)
        print(f"[data_collection] wrote {config.MATCHES_CSV}")


def load_teams() -> pd.DataFrame:
    if not config.TEAMS_CSV.exists():
        generate_sample_data()
    return pd.read_csv(config.TEAMS_CSV)


def load_groups() -> pd.DataFrame:
    if not config.GROUPS_CSV.exists():
        generate_sample_data()
    return pd.read_csv(config.GROUPS_CSV)


def load_matches() -> pd.DataFrame:
    if not config.MATCHES_CSV.exists():
        generate_sample_data()
    df = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"])
    return df


if __name__ == "__main__":
    generate_sample_data(force=True)
    print(load_teams().head())
    print(load_groups().head())
    print(load_matches().head())
