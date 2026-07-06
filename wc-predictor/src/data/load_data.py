"""
Load and clean historical international football match results.

Expected input: data/raw/results.csv from Kaggle dataset
"International football results from 1872 to 2026" (martj42).

Expected columns in the raw CSV:
    date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/results.csv")
PROCESSED_PATH = Path("data/processed/matches.parquet")


def load_raw_results(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle results CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'results.csv' from the Kaggle dataset "
            "'International football results from 1872 to 2026' (martj42) "
            f"and place it at {path}."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, normalize team names, sort by date."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    # Normalize team name whitespace/casing artifacts
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["result"] = df.apply(_match_result, axis=1)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _match_result(row: pd.Series) -> str:
    """Return 'H', 'D', or 'A' from the home team's perspective."""
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} cleaned matches to {path}")


def main() -> None:
    df = load_raw_results()
    df = clean_results(df)
    save_processed(df)


if __name__ == "__main__":
    main()
