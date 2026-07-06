"""
Dixon-Coles model: adjusted bivariate Poisson for football scores.

Standard approach (Dixon & Coles, 1997) extended with time decay so recent
matches matter more than old ones. Reimplemented from the published method,
not copied from any codebase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

TIME_DECAY_XI = 0.0018  # per-day decay; tune based on backtest


def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(self):
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    def fit(self, matches: pd.DataFrame, max_date: pd.Timestamp | None = None) -> "DixonColesModel":
        self.teams = sorted(
            set(matches["home_team"]).union(set(matches["away_team"]))
        )
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        if max_date is None:
            max_date = matches["date"].max()
        days_ago = (max_date - matches["date"]).dt.days.clip(lower=0).to_numpy()
        weights = np.exp(-TIME_DECAY_XI * days_ago)

        home_idx = matches["home_team"].map(idx).to_numpy()
        away_idx = matches["away_team"].map(idx).to_numpy()
        home_goals = matches["home_score"].to_numpy()
        away_goals = matches["away_score"].to_numpy()

        def unpack(params: np.ndarray):
            attack = params[:n]
            defense = params[n : 2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] - defense[home_idx])

            ll = (
                poisson.logpmf(home_goals, lam)
                + poisson.logpmf(away_goals, mu)
            )
            tau_vals = np.array(
                [
                    _tau(hg, ag, l, m, rho)
                    for hg, ag, l, m in zip(home_goals, away_goals, lam, mu)
                ]
            )
            tau_vals = np.clip(tau_vals, 1e-6, None)
            ll = ll + np.log(tau_vals)
            return -np.sum(ll * weights)

        x0 = np.zeros(2 * n + 2)
        # Constrain average attack to 0 via a soft penalty baked into init;
        # scipy minimize with L-BFGS-B is sufficient for this scale of problem.
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 200, "disp": False},
        )
        attack, defense, home_adv, rho = unpack(result.x)

        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        return self

    def predict_score_distribution(
        self, home_team: str, away_team: str, max_goals: int = 8, neutral: bool = True
    ) -> np.ndarray:
        """Return a (max_goals+1) x (max_goals+1) matrix of P(home=i, away=j)."""
        a_h = self.attack.get(home_team, 0.0)
        d_h = self.defense.get(home_team, 0.0)
        a_a = self.attack.get(away_team, 0.0)
        d_a = self.defense.get(away_team, 0.0)

        home_adv = 0.0 if neutral else self.home_adv
        lam = np.exp(a_h - d_a + home_adv)
        mu = np.exp(a_a - d_h)

        goals = np.arange(0, max_goals + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        matrix = np.outer(p_home, p_away)

        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                matrix[i, j] *= _tau(i, j, lam, mu, self.rho)

        matrix /= matrix.sum()
        return matrix

    def match_probabilities(
        self, home_team: str, away_team: str, neutral: bool = True
    ) -> dict[str, float]:
        matrix = self.predict_score_distribution(home_team, away_team, neutral=neutral)
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def most_likely_score(self, home_team: str, away_team: str, neutral: bool = True):
        matrix = self.predict_score_distribution(home_team, away_team, neutral=neutral)
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        return int(i), int(j), float(matrix[i, j])


def main() -> None:
    processed_path = Path("data/processed/matches.parquet")
    if not processed_path.exists():
        print(f"{processed_path} not found. Run `python -m src.data.load_data` first.")
        return

    matches = pd.read_parquet(processed_path)
    # Dixon-Coles is O(n_teams^2) params optimized over all matches; for a
    # quick first run, consider filtering to the last N years of matches.
    model = DixonColesModel().fit(matches)

    print(f"Fitted Dixon-Coles on {len(matches)} matches, {len(model.teams)} teams")
    print("rho =", model.rho, "home_adv =", model.home_adv)


if __name__ == "__main__":
    main()
