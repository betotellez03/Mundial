# ⚽ World Cup 2026 Predictor

An end-to-end, **runnable-out-of-the-box** Python project that predicts FIFA
World Cup 2026 match results and simulates the entire tournament with
Monte-Carlo methods.

It combines three complementary models into an ensemble:

| Model | What it captures |
|-------|------------------|
| **Elo** | Long-run team strength, updated chronologically with margin-of-victory scaling and home/host advantage. |
| **Poisson (xG)** | Goal-scoring rates via per-team attack/defence strengths → full scoreline distribution. |
| **Machine learning** | Non-linear 1X2 patterns (gradient boosting) over engineered, leakage-free features. |
| **Ensemble** | Weighted average of the three, normalised to a valid probability vector. |

The tournament simulator uses the **official 2026 format**:

- **48 teams**, **12 groups of 4**
- **Top 2 of each group** advance, plus the **8 best third-placed teams** → 32 teams
- Knockout stage starts at the **Round of 32** → R16 → QF → SF → Final
- **Host advantage** for **Mexico, USA and Canada** (documented Elo/xG boost)
- **≥ 10,000 Monte-Carlo simulations** (default 10,000, configurable)

---

## 📁 Project structure

```
worldcup_2026_predictor/
├── data/
│   ├── raw/            # input CSV templates (auto-generated sample data)
│   ├── processed/      # cleaned matches, features, Elo ratings
│   └── outputs/        # probabilities, group tables, bracket, evaluation
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── config.py               # paths, format constants, hyper-parameters
│   ├── data_collection.py      # sample-data bootstrapping + loaders
│   ├── data_cleaning.py        # canonical names, 1X2 labels, sorting
│   ├── feature_engineering.py  # leakage-free rolling features
│   ├── elo_model.py            # Elo rating system + 1X2 + xG
│   ├── poisson_model.py        # attack/defence Poisson xG model
│   ├── ml_model.py             # gradient-boosting 1X2 (temporal split)
│   ├── match_predictor.py      # ensemble + single-match prediction
│   ├── tournament_simulator.py # group + knockout Monte-Carlo engine
│   └── evaluation.py           # accuracy, log loss, Brier, calibration
├── app/
│   └── streamlit_app.py        # interactive web UI
├── requirements.txt
├── README.md
└── main.py                     # full pipeline entry point
```

---

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (creates sample data, trains, evaluates, simulates)
python main.py

# 3. Launch the interactive app
streamlit run app/streamlit_app.py
```

The first run automatically writes sample CSV templates to `data/raw/` and all
result files to `data/outputs/`. No external data or network access required.

### Command-line options

```bash
python main.py --sims 20000      # more Monte-Carlo simulations
python main.py --force-data      # regenerate the sample CSV templates
```

---

## 📊 Outputs

After `python main.py` you will find in `data/outputs/`:

| File | Contents |
|------|----------|
| `match_probabilities.csv` | 1X2 probabilities, expected goals and most-likely score for every group-stage fixture. |
| `group_tables.csv` | Final standings of a representative simulated group stage. |
| `bracket_results.csv` | Round-by-round winners/losers of a representative knockout run. |
| `champion_probabilities.csv` | Per-team probability of reaching each stage and winning the title. |
| `model_evaluation.csv` | Accuracy, log loss and Brier score per model (temporal hold-out). |
| `calibration_ensemble.csv` | Reliability table (predicted vs observed home-win frequency). |

---

## 🧪 Using your own (real) data

Replace the files in `data/raw/` with real data using the **same columns**, then
re-run `python main.py`. The rest of the pipeline is unchanged.

**`teams.csv`**
```
team,confederation,seed_elo,is_host
Brazil,CONMEBOL,2020,0
Mexico,CONCACAF,1820,1
...
```

**`groups_2026.csv`**
```
group,team
A,Mexico
A,...
```

**`matches.csv`** (international results history; one row per match)
```
date,home_team,away_team,home_score,away_score,tournament,neutral
2024-06-20,Brazil,Argentina,1,1,Qualifier,0
...
```

> The bundled `matches.csv` is **synthetic** — results are simulated from each
> team's latent strength, *not* hand-picked. The models learn from it exactly
> as they would from real data, so no predictions are hard-coded.

---

## 🔬 Methodology notes

- **No data leakage.** Every ML feature (Elo, rolling form, goals for/against)
  is computed using **only matches before** the one being predicted. The
  train/test split is **temporal** (oldest 80 % train, most recent 20 % test).
- **Host advantage** is applied as a documented Elo/xG bonus for the three
  hosts (`config.HOST_ELO_BONUS`), not as a fixed result.
- **Knockout draws** are resolved with a penalty-shootout model that dampens
  toward 50/50 while slightly favouring the stronger side.
- **Tie-breakers** follow points → goal difference → goals for → Elo (a proxy
  for the lower-priority FIFA criteria).
- **Performance:** per-pair expected goals and 1X2 probabilities are cached
  once, so 10,000+ full tournaments run in seconds.

---

## ⚙️ Configuration

All knobs live in `src/config.py`: paths, format constants, Elo K-factor and
home/host advantage, Poisson baselines, ML train fraction, **ensemble weights**
and the number of simulations. Tune the ensemble weights using
`model_evaluation.csv` as evidence.

---

## 📓 Notebook

`notebooks/exploration.ipynb` shows how to load the trained models, predict a
single match, and inspect simulation output interactively.

---

## ⚠️ Disclaimer

This is an MVP for educational and entertainment purposes. The bundled data is
synthetic; predictions are only as good as the data you feed in. For serious
forecasting, supply a high-quality historical match dataset.
