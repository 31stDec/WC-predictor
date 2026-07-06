"""
Ensemble combining Elo, Dixon-Coles, and GBM predictions.

Default strategy: weighted average of the three models' 1X2 probabilities.
Weights can be tuned via backtesting (see notebooks/ for an example).
Score prediction (for "exact score") comes from Dixon-Coles only, since
Elo/GBM don't model score distributions directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatingSystem
from src.models.gbm_model import GBMResultModel


@dataclass
class EnsembleWeights:
    elo: float = 0.3
    dixon_coles: float = 0.3
    gbm: float = 0.4

    def normalized(self) -> "EnsembleWeights":
        total = self.elo + self.dixon_coles + self.gbm
        return EnsembleWeights(self.elo / total, self.dixon_coles / total, self.gbm / total)


class EnsembleModel:
    def __init__(
        self,
        elo_model: EloRatingSystem,
        dc_model: DixonColesModel,
        gbm_model: GBMResultModel,
        weights: EnsembleWeights = field(default_factory=EnsembleWeights),
    ):
        self.elo_model = elo_model
        self.dc_model = dc_model
        self.gbm_model = gbm_model
        self.weights = weights.normalized()

    def match_probabilities(
        self, home_team: str, away_team: str, gbm_feature_row: dict, neutral: bool = True
    ) -> dict[str, float]:
        p_elo = self.elo_model.match_probabilities(home_team, away_team, neutral=neutral)
        p_dc = self.dc_model.match_probabilities(home_team, away_team, neutral=neutral)
        p_gbm = self.gbm_model.match_probabilities(gbm_feature_row)

        combined = {}
        for outcome in ("home_win", "draw", "away_win"):
            combined[outcome] = (
                self.weights.elo * p_elo[outcome]
                + self.weights.dixon_coles * p_dc[outcome]
                + self.weights.gbm * p_gbm[outcome]
            )

        total = sum(combined.values())
        return {k: v / total for k, v in combined.items()}

    def predicted_score(self, home_team: str, away_team: str, neutral: bool = True):
        """Exact score comes from Dixon-Coles (only model with a score distribution)."""
        return self.dc_model.most_likely_score(home_team, away_team, neutral=neutral)
