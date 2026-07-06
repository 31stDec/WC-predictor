"""
Elo rating model for international football teams.

Standard World Football Elo approach (similar to eloratings.net), with:
- K-factor scaled by tournament importance
- Goal difference multiplier
- Home advantage bonus

Reference for the weighting idea (not copied, reimplemented from scratch):
https://en.wikipedia.org/wiki/World_Football_Elo_Ratings
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 60.0  # Elo points added to home team's effective rating

# Tournament importance weights (K-factor base)
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Friendly": 20,
}
DEFAULT_TOURNAMENT_WEIGHT = 30


class EloRatingSystem:
    def __init__(self, default_rating: float = DEFAULT_RATING):
        self.ratings: dict[str, float] = defaultdict(lambda: default_rating)
        self.history: list[dict] = []

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that team A beats/draws-favorably against team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _goal_diff_multiplier(self, goal_diff: int) -> float:
        """Larger wins move the rating more, with diminishing returns."""
        goal_diff = abs(goal_diff)
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return (11 + goal_diff) / 8.0

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> None:
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        home_effective = home_rating + (0 if neutral else HOME_ADVANTAGE)

        expected_home = self.expected_score(home_effective, away_rating)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = TOURNAMENT_WEIGHT.get(tournament, DEFAULT_TOURNAMENT_WEIGHT)
        gd_mult = self._goal_diff_multiplier(home_score - away_score)

        delta = k * gd_mult * (actual_home - expected_home)

        self.ratings[home_team] = home_rating + delta
        self.ratings[away_team] = away_rating - delta

        self.history.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_rating_before": home_rating,
                "away_rating_before": away_rating,
                "home_rating_after": self.ratings[home_team],
                "away_rating_after": self.ratings[away_team],
            }
        )

    def fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Replay full match history in chronological order to build ratings."""
        for _, row in matches.iterrows():
            self.update_match(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=row["home_score"],
                away_score=row["away_score"],
                tournament=row.get("tournament", "Friendly"),
                neutral=bool(row.get("neutral", False)),
            )
        return self

    def current_ratings_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                [{"team": t, "elo": r} for t, r in self.ratings.items()]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def match_probabilities(
        self, home_team: str, away_team: str, neutral: bool = True
    ) -> dict[str, float]:
        """
        Return P(home win), P(draw), P(away win) using a simple logistic
        mapping from Elo diff, with a draw band around the pick'em point.
        """
        home_rating = self.get_rating(home_team) + (0 if neutral else HOME_ADVANTAGE)
        away_rating = self.get_rating(away_team)

        p_home_vs_away = self.expected_score(home_rating, away_rating)

        # Heuristic draw probability: peaks when teams are evenly matched
        draw_prob = 0.28 - 0.20 * abs(p_home_vs_away - 0.5) * 2
        draw_prob = max(0.15, min(0.32, draw_prob))

        p_home = p_home_vs_away * (1 - draw_prob)
        p_away = (1 - p_home_vs_away) * (1 - draw_prob)

        total = p_home + draw_prob + p_away
        return {
            "home_win": p_home / total,
            "draw": draw_prob / total,
            "away_win": p_away / total,
        }


def main() -> None:
    processed_path = Path("data/processed/matches.parquet")
    if not processed_path.exists():
        print(
            f"{processed_path} not found. Run `python -m src.data.load_data` first."
        )
        return

    matches = pd.read_parquet(processed_path)
    elo = EloRatingSystem().fit(matches)

    out_path = Path("data/processed/elo_ratings.csv")
    elo.current_ratings_df().to_csv(out_path, index=False)
    print(f"Saved current Elo ratings to {out_path}")
    print(elo.current_ratings_df().head(10))


if __name__ == "__main__":
    main()
