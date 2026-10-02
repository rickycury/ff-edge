"""
Roster-aware value: what does this player add to MY starting lineup?

VORP is static — it says a running back is worth 92 points over replacement
whether he is your RB1 or your RB5. That is correct as a market valuation and
wrong as a draft instruction: once two good RBs are rostered, the next one
only competes for a flex or bench slot, so what he actually adds collapses,
while an empty TE slot still pays full freight.

Without this the board happily walks you into five running backs and no
quarterback.

Method: fill every starting slot the roster cannot cover with a synthetic
replacement-level player (using the baselines vorp.solve_replacement_levels
already computes), then measure how much the optimal starting lineup improves
when the candidate is added. On an empty roster this reduces exactly to VORP,
and it decays naturally as slots fill — no separate "positional need" fudge
factor required.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from vorp import FLEX_ELIGIBILITY, LeagueConfig, Player


@dataclass
class Lineup:
    points: float
    assignment: dict[str, list[Player]]   # slot label -> players in it
    bench: list[Player]


def optimal_lineup(players: Sequence[Player], league: LeagueConfig) -> Lineup:
    """
    Best legal starting lineup from this set of players.

    Greedy is optimal for this slot structure: dedicated slots accept exactly
    one position, so they must take the best player at that position; flex
    slots are then filled most-restrictive-first, which prevents a wide flex
    (SUPER_FLEX) from eating a player that only a narrow one (REC_FLEX) could
    have used.
    """
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        by_pos.setdefault(p.pos, []).append(p)
    for pool in by_pos.values():
        pool.sort(key=lambda p: p.proj, reverse=True)

    used: set[int] = set()
    assignment: dict[str, list[Player]] = {}
    total = 0.0

    for pos, count in league.dedicated.items():
        taken: list[Player] = []
        for p in by_pos.get(pos, []):
            if len(taken) >= count:
                break
            if id(p) in used:
                continue
            used.add(id(p))
            taken.append(p)
            total += p.proj
        if taken:
            assignment[pos] = taken

    flex_order = sorted(
        league.flex_slots.items(), key=lambda kv: len(FLEX_ELIGIBILITY[kv[0]])
    )
    for flex_type, count in flex_order:
        eligible_positions = FLEX_ELIGIBILITY[flex_type]
        pool = [
            p for pos, group in by_pos.items() if pos in eligible_positions
            for p in group if id(p) not in used
        ]
        pool.sort(key=lambda p: p.proj, reverse=True)
        taken = []
        for p in pool[:count]:
            used.add(id(p))
            taken.append(p)
            total += p.proj
        if taken:
            assignment[flex_type] = taken

    bench = [p for p in players if id(p) not in used]
    return Lineup(points=total, assignment=assignment, bench=bench)


def replacement_players(
    baselines: dict[str, float], league: LeagueConfig
) -> list[Player]:
    """
    Synthetic players at replacement level, enough to fill every starting slot.

    These stand in for slots the real roster has not covered yet. Using the
    solved baselines (not a guess) is what keeps marginal value consistent
    with the VORP the rest of the board is built on.
    """
    fillers: list[Player] = []
    for pos, count in league.dedicated.items():
        pts = baselines.get(pos, 0.0)
        for i in range(count):
            fillers.append(Player(name=f"repl {pos}{i+1}", pos=pos, proj=pts))
    for flex_type, count in league.flex_slots.items():
        # Fill a flex with the best replacement among its eligible positions —
        # that is genuinely what is available off waivers for that slot.
        eligible = [
            (baselines.get(p, 0.0), p) for p in FLEX_ELIGIBILITY[flex_type]
            if p in baselines
        ]
        if not eligible:
            continue
        pts, pos = max(eligible)
        for i in range(count):
            fillers.append(Player(name=f"repl {flex_type}{i+1}", pos=pos, proj=pts))
    return fillers


class RosterValuer:
    """
    Marginal-value calculator for one roster.

    Build once per roster, then ask about many candidates — the baseline
    lineup is computed a single time rather than per player.
    """

    def __init__(
        self,
        roster: Iterable[Player],
        league: LeagueConfig,
        baselines: dict[str, float],
    ):
        self.league = league
        self.baselines = baselines
        self.roster = list(roster)
        self._fillers = replacement_players(baselines, league)
        self._base_pool = self.roster + self._fillers
        self.baseline_points = optimal_lineup(self._base_pool, league).points

    def marginal_value(self, candidate: Player) -> float:
        """
        Points this player adds to our optimal starting lineup, over and above
        what a replacement-level body at his slot would have given us.

        Equals his VORP on an empty roster; shrinks toward zero as the slots
        he could fill get covered by players already better than him.
        """
        with_him = optimal_lineup(self._base_pool + [candidate], self.league).points
        return with_him - self.baseline_points

    def positional_need(self, pool: Sequence[Player]) -> dict[str, float]:
        """
        Marginal value of the best available player at each position.

        This is "need" expressed in points rather than vibes: a position where
        the best available adds 60 points to the lineup is a hole; one where he
        adds 4 is already covered, however thin the depth chart looks.
        """
        best: dict[str, Player] = {}
        for p in pool:
            cur = best.get(p.pos)
            if cur is None or p.proj > cur.proj:
                best[p.pos] = p
        return {pos: self.marginal_value(p) for pos, p in best.items()}

    def starters(self) -> Lineup:
        """Current best lineup, replacement fillers included, for display."""
        return optimal_lineup(self._base_pool, self.league)
