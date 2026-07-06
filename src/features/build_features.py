"""
Build ML-ready features from cleaned match history + Elo ratings.

Features (per match, from home team's perspective):
    elo_diff            : home_elo - away_elo (using pre-match ratings)
    home_form_5         : points per game in last 5 matches (home team)
    away_form_5         : points per game in last 5 matches (away team)
    home_goal_avg_5     : avg goals scored, last 5 matches
    away_goal_avg_5     : avg goals scored, last 5 matches
    h2h_home_win_rate   : historical win rate of home team vs this opponent
    is_world_cup        : 1 if tournament is FIFA World Cup
    neutral             : 1 if played at neutral venue
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from src.models.elo import EloRatingSystem

N_FORM_MATCHES = 5


def _points(result: str, is_home: bool) -> int:
    if result == "D":
        return 1
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3
    return 0


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)

    elo = EloRatingSystem()
    recent_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=N_FORM_MATCHES))
    recent_goals: dict[str, deque] = defaultdict(lambda: deque(maxlen=N_FORM_MATCHES))
    h2h_record: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [wins, games]

    rows = []
    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]

        # --- Snapshot features BEFORE this match updates state ---
        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)

        home_form = sum(recent_results[home]) / len(recent_results[home]) if recent_results[home] else 1.0
        away_form = sum(recent_results[away]) / len(recent_results[away]) if recent_results[away] else 1.0

        home_goal_avg = sum(recent_goals[home]) / len(recent_goals[home]) if recent_goals[home] else 1.2
        away_goal_avg = sum(recent_goals[away]) / len(recent_goals[away]) if recent_goals[away] else 1.2

        h2h_key = (home, away)
        h2h_wins, h2h_games = h2h_record[h2h_key]
        h2h_win_rate = h2h_wins / h2h_games if h2h_games > 0 else 0.5

        rows.append(
            {
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "elo_diff": home_elo - away_elo,
                "home_form_5": home_form,
                "away_form_5": away_form,
                "home_goal_avg_5": home_goal_avg,
                "away_goal_avg_5": away_goal_avg,
                "h2h_home_win_rate": h2h_win_rate,
                "is_world_cup": int(row.get("tournament", "") == "FIFA World Cup"),
                "neutral": int(bool(row.get("neutral", False))),
                "result": row["result"],
            }
        )

        # --- Update state AFTER recording features ---
        elo.update_match(
            home, away, row["home_score"], row["away_score"],
            tournament=row.get("tournament", "Friendly"),
            neutral=bool(row.get("neutral", False)),
        )

        recent_results[home].append(_points(row["result"], is_home=True))
        recent_results[away].append(_points(row["result"], is_home=False))
        recent_goals[home].append(row["home_score"])
        recent_goals[away].append(row["away_score"])

        h2h_record[h2h_key][1] += 1
        if row["result"] == "H":
            h2h_record[h2h_key][0] += 1

    return pd.DataFrame(rows)


def main() -> None:
    processed_path = Path("data/processed/matches.parquet")
    if not processed_path.exists():
        print(f"{processed_path} not found. Run `python -m src.data.load_data` first.")
        return

    matches = pd.read_parquet(processed_path)
    features = build_features(matches)

    out_path = Path("data/processed/features.parquet")
    features.to_parquet(out_path, index=False)
    print(f"Saved {len(features)} rows of features to {out_path}")


if __name__ == "__main__":
    main()
