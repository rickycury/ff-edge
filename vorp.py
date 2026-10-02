"""
Flex-aware VORP and tier engine for fantasy football draft boards.

The core idea: replacement level is not "position rank N_teams * starters".
Flex slots pull extra RB/WR/TE into starting lineups, which pushes the real
baseline deeper. We solve for it iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics


# Which positions can fill each Sleeper flex type.
FLEX_ELIGIBILITY: dict[str, set[str]] = {
    "FLEX": {"RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}

# Slots that don't participate in VORP at all.
IGNORED_SLOTS = {"BN", "IR", "TAXI"}


@dataclass
class LeagueConfig:
    """Derived from Sleeper's league endpoint, or built by hand."""

    n_teams: int
    roster_positions: list[str]

    # populated in __post_init__
    dedicated: dict[str, int] = field(default_factory=dict)
    flex_slots: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dedicated = {}
        self.flex_slots = {}
        for slot in self.roster_positions:
            if slot in IGNORED_SLOTS:
                continue
            if slot in FLEX_ELIGIBILITY:
                self.flex_slots[slot] = self.flex_slots.get(slot, 0) + 1
            else:
                self.dedicated[slot] = self.dedicated.get(slot, 0) + 1

    @property
    def scored_positions(self) -> set[str]:
        pos = set(self.dedicated)
        for flex in self.flex_slots:
            pos |= FLEX_ELIGIBILITY[flex]
        return pos


@dataclass
class Player:
    name: str
    pos: str
    player_id: str | None = None      # Sleeper id, the cross-source join key
    team: str | None = None
    injury_status: str | None = None
    proj: float = 0.0                 # populated by projections layer
    adp: float | None = None

    # filled in by the engine
    vorp: float = 0.0
    tier: int = 0


def _baseline_points(pool: list[Player], n_starters: int) -> float:
    """
    Replacement level = the best player at this position who is NOT
    expected to hold a starting job. pool must be sorted desc by proj.

    If the position is so thin that every rostered player starts, fall back
    to the last player in the pool rather than blowing up.
    """
    if not pool:
        return 0.0
    idx = min(n_starters, len(pool) - 1)
    return pool[idx].proj


def solve_replacement_levels(
    players: list[Player],
    league: LeagueConfig,
    max_iters: int = 25,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Iteratively solve for flex-aware replacement level.

    Returns (baseline_points_by_pos, effective_starter_count_by_pos).
    """
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        if p.pos in league.scored_positions:
            by_pos.setdefault(p.pos, []).append(p)
    for pool in by_pos.values():
        pool.sort(key=lambda p: p.proj, reverse=True)

    # Pass 0: ignore flex entirely.
    starters = {
        pos: league.n_teams * league.dedicated.get(pos, 0) for pos in by_pos
    }

    for _ in range(max_iters):
        baselines = {
            pos: _baseline_points(pool, starters[pos]) for pos, pool in by_pos.items()
        }

        # Everyone below the current dedicated-starter cutoff competes for flex.
        flex_taken: dict[str, int] = {pos: 0 for pos in by_pos}
        for flex_type, count in league.flex_slots.items():
            eligible = FLEX_ELIGIBILITY[flex_type]
            candidates: list[Player] = []
            for pos, pool in by_pos.items():
                if pos not in eligible:
                    continue
                dedicated_cut = league.n_teams * league.dedicated.get(pos, 0)
                already = flex_taken[pos]
                for p in pool[dedicated_cut + already:]:
                    candidates.append(p)
            candidates.sort(key=lambda p: p.proj - baselines[p.pos], reverse=True)
            for p in candidates[: league.n_teams * count]:
                flex_taken[p.pos] += 1

        new_starters = {
            pos: league.n_teams * league.dedicated.get(pos, 0) + flex_taken[pos]
            for pos in by_pos
        }
        if new_starters == starters:
            break
        starters = new_starters

    baselines = {
        pos: _baseline_points(pool, starters[pos]) for pos, pool in by_pos.items()
    }
    return baselines, starters


def assign_vorp(players: list[Player], league: LeagueConfig) -> dict[str, float]:
    baselines, _ = solve_replacement_levels(players, league)
    for p in players:
        p.vorp = p.proj - baselines.get(p.pos, 0.0)
    return baselines


def assign_tiers(
    players: list[Player],
    pos: str,
    width_frac: float = 0.08,
    cliff_z: float = 2.0,
    max_rank: int | None = None,
) -> None:
    """
    Two rules, either of which opens a new tier:

    1. WIDTH  - cumulative VORP drop from the current tier's leader exceeds
                width_frac * (VORP range of the position). This is what makes
                tiers behave sensibly on a smooth curve: it says "these guys
                are within X points of each other, treat them as fungible."
    2. CLIFF  - a single gap exceeds mean + cliff_z * stdev of gaps. This
                catches genuine structural breaks that the width rule would
                otherwise walk straight through.

    Bigger width_frac -> fewer, chunkier tiers. Lower cliff_z -> more breaks.
    """
    pool = sorted(
        [p for p in players if p.pos == pos], key=lambda p: p.vorp, reverse=True
    )
    if max_rank:
        pool = pool[:max_rank]
    if len(pool) < 3:
        for p in pool:
            p.tier = 1
        return

    gaps = [pool[i].vorp - pool[i + 1].vorp for i in range(len(pool) - 1)]
    spread = pool[0].vorp - pool[-1].vorp
    width_limit = width_frac * spread if spread > 0 else float("inf")
    cliff_limit = statistics.mean(gaps) + cliff_z * statistics.pstdev(gaps)

    tier = 1
    pool[0].tier = tier
    tier_leader_vorp = pool[0].vorp

    for i, gap in enumerate(gaps):
        nxt = pool[i + 1]
        drop_from_leader = tier_leader_vorp - nxt.vorp
        if drop_from_leader > width_limit or gap > cliff_limit:
            tier += 1
            tier_leader_vorp = nxt.vorp
        nxt.tier = tier
