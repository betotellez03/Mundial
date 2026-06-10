"""
End-to-end pipeline for the World Cup 2026 Predictor.

Run with:

    python main.py                 # full pipeline, default 10,000 simulations
    python main.py --sims 20000    # more Monte-Carlo simulations
    python main.py --force-data    # regenerate the sample CSV templates

Steps
-----
1. Ensure raw data exists (sample templates created on first run).
2. Clean the match history.
3. Engineer leakage-free, temporally ordered features.
4. Train the Elo, Poisson and ML base models (ML uses a temporal split).
5. Build the ensemble match predictor.
6. Evaluate every model on the held-out recent matches.
7. Export predicted probabilities for all group-stage fixtures.
8. Run the Monte-Carlo tournament simulation and export results.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from src import config
from src import data_collection
from src import data_cleaning
from src import feature_engineering as fe
from src import elo_model
from src import poisson_model
from src import ml_model
from src import evaluation
from src.match_predictor import build_predictor
from src.tournament_simulator import TournamentSimulator, export_results


def export_match_probabilities(predictor, groups: pd.DataFrame) -> pd.DataFrame:
    """Predict every group-stage fixture and write match_probabilities.csv."""
    rows = []
    for g, sub in groups.groupby("group"):
        teams = sub["team"].tolist()
        for home, away in combinations(teams, 2):
            pred = predictor.predict_match(home, away, neutral=True)
            rows.append({
                "group": g, "home": home, "away": away,
                "p_home_win": round(pred["p_home"], 4),
                "p_draw": round(pred["p_draw"], 4),
                "p_away_win": round(pred["p_away"], 4),
                "exp_goals_home": round(pred["exp_goals_home"], 2),
                "exp_goals_away": round(pred["exp_goals_away"], 2),
                "likely_score": f"{pred['likely_score'][0]}-{pred['likely_score'][1]}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(config.MATCH_PROBS_CSV, index=False)
    print(f"[main] wrote {config.MATCH_PROBS_CSV} ({len(df)} fixtures)")
    return df


def run_pipeline(n_sims: int = config.N_SIMULATIONS, force_data: bool = False):
    print("=" * 70)
    print("FIFA World Cup 2026 Predictor — full pipeline")
    print("=" * 70)

    # 1. data
    data_collection.generate_sample_data(force=force_data)
    teams = data_collection.load_teams()
    groups = data_collection.load_groups()
    print(f"[main] {len(teams)} teams, {groups['group'].nunique()} groups loaded")

    # 2. clean
    clean = data_cleaning.run()

    # 3. features (leakage-free)
    features = fe.run(clean)

    # 4. base models
    elo = elo_model.train_elo(clean)
    poisson = poisson_model.train_poisson(clean)
    ml, test = ml_model.train_ml(features)

    # 5. ensemble predictor
    predictor = build_predictor(elo, poisson, ml, clean)

    # 6. evaluation (temporal validation)
    print("\n--- Model evaluation (temporal hold-out) ---")
    evaluation.run(predictor, test)

    # 7. group-stage match probabilities
    print("\n--- Group-stage match probabilities ---")
    export_match_probabilities(predictor, groups)

    # 8. Monte-Carlo tournament
    print(f"\n--- Monte-Carlo simulation ({n_sims:,} tournaments) ---")
    sim = TournamentSimulator(predictor, groups)
    mc = sim.run_monte_carlo(n_sims=n_sims)
    export_results(mc)

    print("\nTop 10 title contenders:")
    print(mc["summary"][["team", "P_Champion", "P_Final", "P_SF"]]
          .head(10).to_string(index=False))
    print(f"\nAll outputs written to: {config.OUTPUT_DIR}")
    return mc


def parse_args():
    p = argparse.ArgumentParser(description="World Cup 2026 Predictor pipeline")
    p.add_argument("--sims", type=int, default=config.N_SIMULATIONS,
                   help="number of Monte-Carlo simulations (default: 10000)")
    p.add_argument("--force-data", action="store_true",
                   help="regenerate the sample CSV templates")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(n_sims=args.sims, force_data=args.force_data)
