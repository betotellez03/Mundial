"""
Machine-learning 1X2 model.

A scikit-learn classifier predicts P(Home win), P(Draw), P(Away win) from the
engineered features.  Key points:

* **Temporal validation** – the data is split by *time* (earliest 80 % train,
  most recent 20 % test) rather than randomly, so we never train on the future.
* A calibrated, regularised pipeline (StandardScaler -> GradientBoosting or
  LogisticRegression) keeps probabilities sensible.
* ``predict_proba_match`` predicts a single upcoming match given pre-computed
  features (Elo + neutral default for World-Cup conditions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from . import feature_engineering as fe

CLASSES = ["H", "D", "A"]  # fixed class order for consistent probability columns


def temporal_split(features: pd.DataFrame, train_fraction: float = config.ML_TRAIN_FRACTION):
    """Split chronologically: oldest ``train_fraction`` -> train, rest -> test."""
    features = features.sort_values("date").reset_index(drop=True)
    cut = int(len(features) * train_fraction)
    return features.iloc[:cut].copy(), features.iloc[cut:].copy()


@dataclass
class MLModel:
    pipeline: Pipeline | None = None

    def _new_pipeline(self) -> Pipeline:
        # GradientBoosting handles non-linear feature interactions well and
        # returns reasonably calibrated multi-class probabilities.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                random_state=config.ML_RANDOM_STATE,
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
            )),
        ])

    def fit(self, train: pd.DataFrame) -> "MLModel":
        X = train[fe.FEATURE_COLUMNS].values
        y = train["result"].values
        self.pipeline = self._new_pipeline()
        self.pipeline.fit(X, y)
        return self

    def _align_proba(self, raw: np.ndarray) -> np.ndarray:
        """Reorder classifier output columns to the fixed H, D, A order."""
        classes = list(self.pipeline.named_steps["clf"].classes_)
        idx = [classes.index(c) for c in CLASSES]
        return raw[:, idx]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[fe.FEATURE_COLUMNS].values
        return self._align_proba(self.pipeline.predict_proba(X))

    def predict_proba_match(self, feature_row: dict) -> tuple[float, float, float]:
        """Predict a single match from a dict of features -> (P(H), P(D), P(A))."""
        df = pd.DataFrame([feature_row])[fe.FEATURE_COLUMNS]
        p = self._align_proba(self.pipeline.predict_proba(df.values))[0]
        return float(p[0]), float(p[1]), float(p[2])


def train_ml(features: pd.DataFrame):
    """Train on the temporal-train split; return (model, test_frame)."""
    train, test = temporal_split(features)
    model = MLModel().fit(train)
    print(f"[ml_model] trained on {len(train)} matches, holding out {len(test)} for validation")
    return model, test
