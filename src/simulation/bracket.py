"""
World Cup 2026 format: 48 teams, 12 groups of 4 (A-L).
Group stage -> Round of 32 -> R16 -> QF -> SF -> Final.

Round of 32 qualifiers: group winners + runners-up (24 teams) + the
8 best third-placed teams across all 12 groups.

NOTE: FIFA publishes an official fixed slot mapping for how the 8 best
third-place teams plug into the Round of 32 bracket, which depends on
*which* groups those third-place teams come from. That mapping is not
hardcoded here (fill in `THIRD_PLACE_SLOT_MAP` once your groups are set,
or once FIFA confirms it for the actual 2026 draw) -- this module gives
a reasonable placeholder ordering (seeded by group standing) so the
simulator is runnable end-to-end today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class GroupStanding:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    played: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def sort_key(self):
        # Simplified tiebreak: points, then goal diff, then goals for.
        # (Real FIFA tiebreak rules also cover head-to-head, fair play etc.)
        return (-self.points, -self.goal_diff, -self.goals_for)


def round_robin_fixtures(teams: list[str]) -> list[tuple[str, str]]:
    """All pairings within a group (single round-robin, 4 teams -> 6 matches)."""
    return list(combinations(teams, 2))


def update_standing(
    standings: dict[str, GroupStanding], home: str, away: str, home_goals: int, away_goals: int
) -> None:
    h, a = standings[home], standings[away]
    h.played += 1
    a.played += 1
    h.goals_for += home_goals
    h.goals_against += away_goals
    a.goals_for += away_goals
    a.goals_against += home_goals

    if home_goals > away_goals:
        h.points += 3
    elif home_goals < away_goals:
        a.points += 3
    else:
        h.points += 1
        a.points += 1


def rank_group(standings: dict[str, GroupStanding]) -> list[str]:
    """Return team names sorted 1st -> 4th within a group."""
    ordered = sorted(standings.values(), key=lambda s: s.sort_key())
    return [s.team for s in ordered]


def best_third_placed(third_place_standings: dict[str, GroupStanding], n: int = 8) -> list[str]:
    """Pick the best n third-placed teams across all groups."""
    ordered = sorted(third_place_standings.values(), key=lambda s: s.sort_key())
    return [s.team for s in ordered[:n]]


@dataclass
class GroupStageResult:
    winners: list[str] = field(default_factory=list)       # 1st place, 12 teams
    runners_up: list[str] = field(default_factory=list)     # 2nd place, 12 teams
    best_thirds: list[str] = field(default_factory=list)    # best 8 of the 3rd place teams

    def round_of_32_field(self) -> list[str]:
        """All 32 teams advancing (order not yet bracketed)."""
        return self.winners + self.runners_up + self.best_thirds


def build_round_of_32_pairs(result: GroupStageResult) -> list[tuple[str, str]]:
    """
    Simplified seeding: pair winners against runners-up/thirds in a fixed
    rotation so no team meets a team from its own group in Round of 32.
    This is a reasonable stand-in for the official FIFA slot table --
    replace with the exact mapping once available for full accuracy.
    """
    winners = result.winners
    runners_up = result.runners_up
    thirds = result.best_thirds

    # 8 winners face the 8 best thirds; the remaining 4 winners face
    # runners-up from "opposite" groups; runners-up face each other otherwise.
    pairs: list[tuple[str, str]] = []
    n_thirds = len(thirds)

    for i in range(n_thirds):
        pairs.append((winners[i], thirds[i]))

    remaining_winners = winners[n_thirds:]
    for i, w in enumerate(remaining_winners):
        pairs.append((w, runners_up[i]))

    remaining_runners_up = runners_up[len(remaining_winners):]
    for i in range(0, len(remaining_runners_up) - 1, 2):
        pairs.append((remaining_runners_up[i], remaining_runners_up[i + 1]))

    return pairs


def next_round_pairs(winners: list[str]) -> list[tuple[str, str]]:
    """Pair up winners of the previous round in order (R32->R16->QF->SF->Final)."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
