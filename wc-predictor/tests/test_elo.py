import pandas as pd

from src.models.elo import EloRatingSystem


def test_elo_updates_after_win():
    elo = EloRatingSystem()
    before_home = elo.get_rating("A")
    before_away = elo.get_rating("B")

    elo.update_match("A", "B", home_score=2, away_score=0, tournament="Friendly", neutral=True)

    assert elo.get_rating("A") > before_home
    assert elo.get_rating("B") < before_away


def test_elo_probabilities_sum_to_one():
    elo = EloRatingSystem()
    elo.update_match("A", "B", 3, 1, neutral=True)
    probs = elo.match_probabilities("A", "B")
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_fit_from_dataframe():
    df = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "tournament": "Friendly", "neutral": False},
            {"home_team": "B", "away_team": "A", "home_score": 2, "away_score": 2, "tournament": "Friendly", "neutral": False},
        ]
    )
    elo = EloRatingSystem().fit(df)
    assert "A" in elo.ratings
    assert "B" in elo.ratings
