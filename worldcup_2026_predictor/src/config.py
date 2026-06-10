"""
Central configuration for the World Cup 2026 Predictor.

All paths, model hyper-parameters and tournament-format constants live here so
that every other module imports a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root = the directory that contains the ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Raw input templates
TEAMS_CSV = RAW_DIR / "teams.csv"
MATCHES_CSV = RAW_DIR / "matches.csv"
GROUPS_CSV = RAW_DIR / "groups_2026.csv"

# Processed artefacts
CLEAN_MATCHES_CSV = PROCESSED_DIR / "matches_clean.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
ELO_RATINGS_CSV = PROCESSED_DIR / "elo_ratings.csv"

# Output artefacts
MATCH_PROBS_CSV = OUTPUT_DIR / "match_probabilities.csv"
GROUP_TABLES_CSV = OUTPUT_DIR / "group_tables.csv"
BRACKET_CSV = OUTPUT_DIR / "bracket_results.csv"
CHAMPION_PROBS_CSV = OUTPUT_DIR / "champion_probabilities.csv"
EVALUATION_CSV = OUTPUT_DIR / "model_evaluation.csv"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Tournament format (official FIFA World Cup 2026)
# ---------------------------------------------------------------------------
N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
GROUP_LABELS = [chr(ord("A") + i) for i in range(N_GROUPS)]  # A .. L

# Two best per group advance directly, plus the 8 best third-placed teams.
N_DIRECT_QUALIFIERS_PER_GROUP = 2
N_BEST_THIRDS = 8
# 12*2 + 8 = 32 teams reach the knockout stage (Round of 32).
N_KNOCKOUT_TEAMS = N_GROUPS * N_DIRECT_QUALIFIERS_PER_GROUP + N_BEST_THIRDS

# Hosts of the 2026 tournament.
HOST_TEAMS = ["Mexico", "United States", "Canada"]

# Group-stage points
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0

# ---------------------------------------------------------------------------
# Elo model
# ---------------------------------------------------------------------------
ELO_K = 40.0                 # base K-factor
ELO_HOME_ADVANTAGE = 65.0    # rating points added to the home/host side
ELO_GOAL_MULTIPLIER = True   # scale K by margin of victory (World-Football-Elo style)
ELO_INITIAL = 1500.0         # rating for teams never seen before

# Host advantage expressed as a one-off Elo bonus for the three host nations
# when they play AT the tournament.  Kept modest and clearly documented so it
# is not mistaken for a hard-coded result.
HOST_ELO_BONUS = 40.0

# ---------------------------------------------------------------------------
# Poisson expected-goals model
# ---------------------------------------------------------------------------
# League/tournament-wide average goals scored by one team in a match.  Used as
# the baseline before applying attack / defence strengths.
POISSON_BASE_HOME_GOALS = 1.45
POISSON_BASE_AWAY_GOALS = 1.15
POISSON_MAX_GOALS = 10        # truncate the score grid here

# ---------------------------------------------------------------------------
# Machine-learning model
# ---------------------------------------------------------------------------
# Fraction (by time) of matches used for training; the most recent matches are
# held out for temporal validation to avoid look-ahead leakage.
ML_TRAIN_FRACTION = 0.8
ML_RANDOM_STATE = RANDOM_SEED

# ---------------------------------------------------------------------------
# Ensemble weights (must sum to 1.0).  Tune these from evaluation results.
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {
    "elo": 0.34,
    "poisson": 0.33,
    "ml": 0.33,
}

# ---------------------------------------------------------------------------
# Monte-Carlo simulation
# ---------------------------------------------------------------------------
N_SIMULATIONS = 10_000

# Outcome labels used everywhere (home win / draw / away win).
OUTCOME_LABELS = ["H", "D", "A"]
