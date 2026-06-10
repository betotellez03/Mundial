"""
Data cleaning.

Takes the raw ``matches.csv`` and produces a tidy ``matches_clean.csv`` with:

* canonical team names (trimmed, alias-mapped),
* parsed dates sorted chronologically (critical for temporal validation),
* a derived 1X2 ``result`` label (H / D / A),
* basic sanity filtering (no negative scores, no team playing itself).
"""
from __future__ import annotations

import pandas as pd

from . import config
from . import data_collection

# Common alternative spellings -> canonical name used across the project.
TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
}


def canonical_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def result_label(home_score: int, away_score: int) -> str:
    """Return the 1X2 outcome from the home team's perspective."""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def clean_matches(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    if matches is None:
        matches = data_collection.load_matches()

    df = matches.copy()

    # Normalise team names.
    df["home_team"] = df["home_team"].map(canonical_team)
    df["away_team"] = df["away_team"].map(canonical_team)

    # Parse + sort by date (temporal order is required downstream).
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])

    # Numeric scores.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Sanity filters.
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    df = df[df["home_team"] != df["away_team"]]

    if "neutral" not in df.columns:
        df["neutral"] = 0
    df["neutral"] = df["neutral"].fillna(0).astype(int)

    df["result"] = [result_label(h, a) for h, a in zip(df["home_score"], df["away_score"])]

    df = df.sort_values("date").reset_index(drop=True)
    return df


def run() -> pd.DataFrame:
    """Clean the raw matches and persist them to ``processed/``."""
    data_collection.generate_sample_data()  # ensure raw data exists
    clean = clean_matches()
    clean.to_csv(config.CLEAN_MATCHES_CSV, index=False)
    print(f"[data_cleaning] wrote {config.CLEAN_MATCHES_CSV} ({len(clean)} rows)")
    return clean


def load_clean_matches() -> pd.DataFrame:
    if not config.CLEAN_MATCHES_CSV.exists():
        return run()
    return pd.read_csv(config.CLEAN_MATCHES_CSV, parse_dates=["date"])


if __name__ == "__main__":
    print(run().head())
