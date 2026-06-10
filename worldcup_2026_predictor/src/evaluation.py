"""
Model evaluation with temporal validation.

Computes, on the held-out (most recent) matches:
* accuracy of the argmax 1X2 prediction,
* multi-class log loss,
* multi-class Brier score (mean squared error of the probability vectors),
* a reliability / calibration table (predicted vs observed frequency).

These metrics are reported for each base model and the ensemble so the weights
in ``config.ENSEMBLE_WEIGHTS`` can be tuned with evidence rather than guesswork.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from . import config
from . import feature_engineering as fe

CLASSES = ["H", "D", "A"]
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def _onehot(y: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        oh[i, _CLASS_IDX[label]] = 1.0
    return oh


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    return float(np.mean(np.sum((proba - _onehot(y_true)) ** 2, axis=1)))


def calibration_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability of the predicted home-win probability."""
    p_home = proba[:, 0]
    y_home = (y_true == "H").astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_home, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(p_home[mask].mean()),
            "observed_freq": float(y_home[mask].mean()),
        })
    return pd.DataFrame(rows)


def evaluate_proba(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    proba = np.clip(proba, 1e-6, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = np.array(CLASSES)[np.argmax(proba, axis=1)]
    # ``labels=CLASSES`` makes log_loss align to our fixed H/D/A column order;
    # the (harmless) "not lexicographically ordered" warning is suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ll = float(log_loss(y_true, proba, labels=CLASSES))
    return {
        "model": name,
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": ll,
        "brier": brier_multiclass(y_true, proba),
    }


def _proba_for_rows(predictor, test: pd.DataFrame, which: str) -> np.ndarray:
    """Produce a (n,3) probability array for a base model or the ensemble."""
    out = np.zeros((len(test), 3))
    for i, r in enumerate(test.itertuples(index=False)):
        home, away = r.home_team, r.away_team
        neutral = bool(getattr(r, "neutral", 0))
        if which == "elo":
            p = predictor.elo.predict_proba(home, away, neutral=neutral)
        elif which == "poisson":
            p = predictor.poisson.predict_proba(home, away, neutral=neutral)
        elif which == "ml":
            row = {c: getattr(r, c) for c in fe.FEATURE_COLUMNS}
            p = predictor.ml.predict_proba_match(row)
        else:  # ensemble
            p = predictor.predict_proba(home, away, neutral=neutral)
        out[i] = p
    return out


def evaluate_all(predictor, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on the temporal test set.

    Returns (metrics_table, ensemble_calibration_table).
    """
    y_true = test["result"].values
    results = []
    ens_proba = None
    for which in ("elo", "poisson", "ml", "ensemble"):
        proba = _proba_for_rows(predictor, test, which)
        results.append(evaluate_proba(which, y_true, proba))
        if which == "ensemble":
            ens_proba = proba

    metrics = pd.DataFrame(results)
    calib = calibration_table(y_true, ens_proba)
    return metrics, calib


def run(predictor, test: pd.DataFrame) -> pd.DataFrame:
    metrics, calib = evaluate_all(predictor, test)
    metrics.to_csv(config.EVALUATION_CSV, index=False)
    calib.to_csv(config.OUTPUT_DIR / "calibration_ensemble.csv", index=False)
    print(f"[evaluation] wrote {config.EVALUATION_CSV}")
    print(metrics.to_string(index=False))
    return metrics
