"""
Monte Carlo tournament simulator: run the full WC 2026 format (groups ->
knockouts) many times using match-outcome probabilities from the ensemble
model, and aggregate how often each team wins the tournament.

Usage:
    groups = {"A": ["Team1", "Team2", "Team3", "Team4"], ...}   # 12 groups
    prob_fn(home, away) -> {"home_win": .., "draw": .., "away_win": ..}
    score_fn(home, away) -> (home_goals, away_goals)            # for group GD

    sim = TournamentSimulator(groups, prob_fn, score_fn)
    results = sim.run(n_simulations=10000)
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable

from src.simulation.bracket import (
    GroupStageResult,
    GroupStanding,
    best_third_placed,
    build_round_of_32_pairs,
    next_round_pairs,
    rank_group,
    round_robin_fixtures,
    update_standing,
)

ProbFn = Callable[[str, str], dict[str, float]]
ScoreFn = Callable[[str, str], tuple[int, int]]


class TournamentSimulator:
    def __init__(self, groups: dict[str, list[str]], prob_fn: ProbFn, score_fn: ScoreFn):
        """
        groups: e.g. {"A": [team1, team2, team3, team4], "B": [...], ...} (12 groups)
        prob_fn: returns 1X2 probabilities for a given (home, away) pairing
        score_fn: returns a plausible (home_goals, away_goals) sample for a match
                  (e.g. sampled from Dixon-Coles score distribution)
        """
        self.groups = groups
        self.prob_fn = prob_fn
        self.score_fn = score_fn

    def _simulate_group(self, teams: list[str]) -> dict[str, GroupStanding]:
        standings = {t: GroupStanding(team=t) for t in teams}
        for home, away in round_robin_fixtures(teams):
            hg, ag = self.score_fn(home, away)
            update_standing(standings, home, away, hg, ag)
        return standings

    def _simulate_group_stage(self) -> GroupStageResult:
        winners, runners_up = [], []
        third_place_standings: dict[str, GroupStanding] = {}

        for _, teams in self.groups.items():
            standings = self._simulate_group(teams)
            ranked = rank_group(standings)
            winners.append(ranked[0])
            runners_up.append(ranked[1])
            third_place_standings[ranked[2]] = standings[ranked[2]]

        best_thirds = best_third_placed(third_place_standings, n=8)
        return GroupStageResult(winners=winners, runners_up=runners_up, best_thirds=best_thirds)

    def _simulate_single_match_winner(self, home: str, away: str) -> str:
        """Knockout match: no draws allowed, sample from home/away win prob
        (renormalized) as a stand-in for extra time / penalties."""
        probs = self.prob_fn(home, away)
        p_home = probs["home_win"] + probs["draw"] / 2
        p_away = probs["away_win"] + probs["draw"] / 2
        total = p_home + p_away
        return home if random.random() < (p_home / total) else away

    def run(self, n_simulations: int = 10000) -> Counter:
        """Run n full-tournament simulations, return a Counter of championship wins per team."""
        champion_counts: Counter = Counter()

        for _ in range(n_simulations):
            group_result = self._simulate_group_stage()
            pairs = build_round_of_32_pairs(group_result)

            round_winners = [self._simulate_single_match_winner(h, a) for h, a in pairs]

            # R16 -> QF -> SF -> Final
            while len(round_winners) > 1:
                next_pairs = next_round_pairs(round_winners)
                round_winners = [
                    self._simulate_single_match_winner(h, a) for h, a in next_pairs
                ]

            champion = round_winners[0]
            champion_counts[champion] += 1

        return champion_counts

    def championship_probabilities(self, n_simulations: int = 10000) -> dict[str, float]:
        counts = self.run(n_simulations)
        return {team: n / n_simulations for team, n in counts.most_common()}


def main() -> None:
    # Minimal runnable example with placeholder teams/probabilities.
    # Replace `groups` with the real WC 2026 draw and `prob_fn`/`score_fn`
    # with your trained ensemble + Dixon-Coles model once available.
    import string

    teams = [f"Team{i}" for i in range(1, 49)]
    letters = list(string.ascii_uppercase[:12])
    groups = {letters[i]: teams[i * 4 : i * 4 + 4] for i in range(12)}

    def dummy_prob_fn(home: str, away: str) -> dict[str, float]:
        return {"home_win": 0.4, "draw": 0.25, "away_win": 0.35}

    def dummy_score_fn(home: str, away: str) -> tuple[int, int]:
        return random.choice([0, 1, 1, 2]), random.choice([0, 1, 1, 2])

    sim = TournamentSimulator(groups, dummy_prob_fn, dummy_score_fn)
    probs = sim.championship_probabilities(n_simulations=2000)

    print("Top 10 championship probabilities (dummy data):")
    for team, p in list(probs.items())[:10]:
        print(f"  {team}: {p:.2%}")


if __name__ == "__main__":
    main()
