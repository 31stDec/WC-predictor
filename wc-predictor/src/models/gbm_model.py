"""
Gradient boosting classifier (XGBoost) for 1X2 prediction, trained on
engineered features from src/features/build_features.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FEATURE_COLS = [
    "elo_diff",
    "home_form_5",
    "away_form_5",
    "home_goal_avg_5",
    "away_goal_avg_5",
    "h2h_home_win_rate",
    "is_world_cup",
    "neutral",
]

MODEL_PATH = Path("data/processed/gbm_model.pkl")


class GBMResultModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        self.label_encoder = LabelEncoder()

    def fit(self, features: pd.DataFrame) -> "GBMResultModel":
        X = features[FEATURE_COLS]
        y = self.label_encoder.fit_transform(features["result"])  # H/D/A -> 0/1/2

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = self.model.predict_proba(X_val)
        print("Validation accuracy:", accuracy_score(y_val, val_pred.argmax(axis=1)))
        print("Validation log loss:", log_loss(y_val, val_pred))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns columns ordered as self.label_encoder.classes_."""
        return self.model.predict_proba(X[FEATURE_COLS])

    def match_probabilities(self, feature_row: dict) -> dict[str, float]:
        X = pd.DataFrame([feature_row])[FEATURE_COLS]
        probs = self.model.predict_proba(X)[0]
        label_map = dict(zip(self.label_encoder.classes_, probs))
        return {
            "home_win": label_map.get("H", 0.0),
            "draw": label_map.get("D", 0.0),
            "away_win": label_map.get("A", 0.0),
        }

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "GBMResultModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def main() -> None:
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print(f"{features_path} not found. Run `python -m src.features.build_features` first.")
        return

    features = pd.read_parquet(features_path)
    model = GBMResultModel().fit(features)
    model.save()
    print(f"Saved trained GBM model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
