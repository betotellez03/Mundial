"""
Streamlit front-end for the World Cup 2026 Predictor.

Run with:

    streamlit run app/streamlit_app.py

Features
--------
* Single-match predictor: pick two teams, see ensemble + per-model 1X2 odds,
  expected goals and the most likely scoreline.
* Tournament simulator: run N Monte-Carlo tournaments and view champion /
  stage-advancement probabilities with an interactive chart.
* Browse the group draw and the cached output CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the ``src`` package importable when launched from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                       # noqa: E402
from src import data_collection              # noqa: E402
from src import data_cleaning                # noqa: E402
from src import feature_engineering as fe    # noqa: E402
from src import elo_model                    # noqa: E402
from src import poisson_model                # noqa: E402
from src import ml_model                     # noqa: E402
from src.match_predictor import build_predictor          # noqa: E402
from src.tournament_simulator import TournamentSimulator  # noqa: E402

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Training models on match history…")
def load_everything():
    """Train all models once and cache them for the session."""
    data_collection.generate_sample_data()
    clean = data_cleaning.run()
    features = fe.run(clean)
    elo = elo_model.train_elo(clean)
    poisson = poisson_model.train_poisson(clean)
    ml, _test = ml_model.train_ml(features)
    predictor = build_predictor(elo, poisson, ml, clean)
    groups = data_collection.load_groups()
    return predictor, groups


@st.cache_data(show_spinner="Running Monte-Carlo simulations…")
def run_simulation(n_sims: int):
    predictor, groups = load_everything()
    sim = TournamentSimulator(predictor, groups)
    mc = sim.run_monte_carlo(n_sims=n_sims)
    # Return only serialisable frames for st.cache_data.
    return mc["summary"], mc["example_group_tables"], mc["example_bracket"]


predictor, groups = load_everything()
teams = sorted(groups["team"].tolist())

st.title("⚽ FIFA World Cup 2026 Predictor")
st.caption("Elo · Poisson xG · Machine Learning · Ensemble — Monte-Carlo tournament simulation")

tab_match, tab_sim, tab_data = st.tabs(
    ["🔮 Single match", "🏆 Tournament simulation", "📋 Data & groups"]
)

# --------------------------------------------------------------------------- #
# Single-match prediction
# --------------------------------------------------------------------------- #
with tab_match:
    st.subheader("Predict a single match")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home / Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = c2.selectbox("Away / Team B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        pred = predictor.predict_match(home, away, neutral=True)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{pred['p_home']*100:.1f}%")
        m2.metric("Draw", f"{pred['p_draw']*100:.1f}%")
        m3.metric(f"{away} win", f"{pred['p_away']*100:.1f}%")

        st.write(
            f"**Expected goals:** {home} {pred['exp_goals_home']:.2f} – "
            f"{pred['exp_goals_away']:.2f} {away}  |  "
            f"**Most likely score:** {pred['likely_score'][0]}–{pred['likely_score'][1]}"
        )

        breakdown = pd.DataFrame(
            {
                "Model": ["Elo", "Poisson", "ML", "Ensemble"],
                f"{home} win": [pred["elo"][0], pred["poisson"][0], pred["ml"][0], pred["ensemble"][0]],
                "Draw": [pred["elo"][1], pred["poisson"][1], pred["ml"][1], pred["ensemble"][1]],
                f"{away} win": [pred["elo"][2], pred["poisson"][2], pred["ml"][2], pred["ensemble"][2]],
            }
        )
        st.dataframe(
            breakdown.style.format({c: "{:.1%}" for c in breakdown.columns[1:]}),
            use_container_width=True,
        )

# --------------------------------------------------------------------------- #
# Tournament simulation
# --------------------------------------------------------------------------- #
with tab_sim:
    st.subheader("Monte-Carlo tournament simulation")
    n_sims = st.slider("Number of simulations", 1_000, 50_000, config.N_SIMULATIONS, step=1_000)
    if st.button("Run simulation", type="primary"):
        summary, group_tables, bracket = run_simulation(n_sims)

        st.markdown("#### Title probabilities")
        top = summary.head(15)
        fig = px.bar(top, x="P_Champion", y="team", orientation="h",
                     labels={"P_Champion": "Champion probability", "team": ""},
                     title=f"Top 15 contenders ({n_sims:,} simulations)")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            summary.style.format({c: "{:.1%}" for c in summary.columns if c.startswith("P_")}),
            use_container_width=True,
        )

        cga, cbr = st.columns(2)
        with cga:
            st.markdown("#### Example group tables (one simulation)")
            st.dataframe(group_tables, use_container_width=True, height=400)
        with cbr:
            st.markdown("#### Example bracket (one simulation)")
            st.dataframe(bracket, use_container_width=True, height=400)
    else:
        st.info("Set the number of simulations and click **Run simulation**.")

# --------------------------------------------------------------------------- #
# Data & groups
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Group draw (12 groups of 4)")
    pivot = (groups.assign(n=groups.groupby("group").cumcount() + 1)
             .pivot(index="n", columns="group", values="team"))
    st.dataframe(pivot, use_container_width=True)

    st.subheader("Cached output files")
    for label, path in [
        ("Match probabilities", config.MATCH_PROBS_CSV),
        ("Champion probabilities", config.CHAMPION_PROBS_CSV),
        ("Group tables (example)", config.GROUP_TABLES_CSV),
        ("Bracket (example)", config.BRACKET_CSV),
        ("Model evaluation", config.EVALUATION_CSV),
    ]:
        if path.exists():
            with st.expander(f"{label} — {path.name}"):
                st.dataframe(pd.read_csv(path), use_container_width=True)
        else:
            st.caption(f"{label}: not generated yet (run `python main.py`).")
