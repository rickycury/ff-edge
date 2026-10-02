"""
Survival math: will he still be there when I pick again?

This is the draft-day edge. Everything upstream answers "what is he worth";
this answers "what do I lose by waiting", which is the only question you
actually face on the clock.

Model: a player's real draft position D is treated as Normal(ADP, SD), so
P(available at pick N) = P(D >= N). Both parameters are measured, not
assumed — ADP and the cross-platform standard deviation both come from the
Flock export (see market.py). That matters: two players with identical ADP 76
behave completely differently if one has SD 4 and the other SD 25.

Which ADP: Sleeper's, because our league drafts on Sleeper. The outside
market is the wrong predictor of our own room (see market.HOME_PLATFORM).

Stdlib only. The normal CDF comes from math.erf rather than pulling in scipy.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from vorp import Player


# A player nobody disagrees about still isn't perfectly predictable — drafts
# have randomness beyond cross-platform consensus. Never let SD go to zero or
# survival becomes a step function and the whole model turns brittle.
SD_FLOOR = 2.5

# Used when a player has too few platforms to measure a spread. Disagreement
# grows with depth, so a flat default would badly understate uncertainty late.
FALLBACK_SD_FRACTION = 0.18


def normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fallback_sd(adp: float) -> float:
    """
    Estimate spread for a player the market barely priced.

    Measured on the 2026 Flock export, cross-platform SD scales roughly with
    ADP (median ~11 picks across the top 200, near zero in round 1, 30+ by the
    double-digit rounds). A proportional estimate is crude but far better than
    a constant, and it fails in the safe direction: it assumes MORE
    uncertainty, so you wait less on players we know least about.
    """
    return max(SD_FLOOR, adp * FALLBACK_SD_FRACTION)


def survival_probability(
    adp: float | None,
    sd: float | None,
    pick: int,
    *,
    already_gone: bool = False,
) -> float:
    """
    P(this player is still on the board when pick `pick` comes around).

    An unpriced player (no ADP anywhere) returns 1.0: the market has no
    opinion, which in practice means nobody is racing to take him.
    """
    if already_gone:
        return 0.0
    if adp is None:
        return 1.0
    spread = sd if sd is not None and sd > 0 else fallback_sd(adp)
    spread = max(spread, SD_FLOOR)
    # P(D >= pick) — he goes at or after our pick.
    return 1.0 - normal_cdf((pick - adp) / spread)


@dataclass
class Candidate:
    """A player plus the market parameters survival math needs."""

    player: Player
    adp: float | None
    sd: float | None
    gone: bool = False

    @property
    def vorp(self) -> float:
        return self.player.vorp

    def survives(self, pick: int) -> float:
        return survival_probability(self.adp, self.sd, pick, already_gone=self.gone)


def expected_survivors(candidates: Iterable[Candidate], pick: int) -> float:
    """
    How many of this group we expect to still be available at `pick`.

    Applied to a tier this is the number that matters: "2.4 players left in TE
    tier 2 at your next pick" tells you whether the tier survives the round.
    """
    return sum(c.survives(pick) for c in candidates)


def expected_best_available(
    candidates: Sequence[Candidate], pick: int
) -> tuple[float, list[tuple[Candidate, float]]]:
    """
    Expected VORP of the best player still on the board at `pick`.

    Walk the pool best-first: player i is the best available exactly when he
    survives AND everyone better than him is gone. Those events are treated as
    independent, which is a simplification — draft runs correlate — but it is
    the standard approximation and it degrades gracefully.

    Returns (expected VORP, [(candidate, P(he is the best available))]).
    """
    ordered = sorted(candidates, key=lambda c: c.vorp, reverse=True)
    total = 0.0
    breakdown: list[tuple[Candidate, float]] = []
    all_better_gone = 1.0
    for c in ordered:
        p_survives = c.survives(pick)
        p_is_best = all_better_gone * p_survives
        total += c.vorp * p_is_best
        breakdown.append((c, p_is_best))
        all_better_gone *= (1.0 - p_survives)
    return total, breakdown


def cost_of_waiting(
    candidate: Candidate,
    pool: Sequence[Candidate],
    next_pick: int,
) -> float:
    """
    VORP surrendered by passing on this player now.

    = his value today  -  the value we expect to salvage at this position when
    we pick again. The candidate stays IN the pool, because waiting does not
    guarantee losing him; if he is near-certain to survive, the cost collapses
    to roughly zero and the model correctly tells you to take someone else.

    This is the number to draft by. A high VORP player who will obviously last
    is not a good pick right now; a moderate one about to disappear is.
    """
    expected, _ = expected_best_available(pool, next_pick)
    return candidate.vorp - expected


def build_candidates(
    players: Iterable[Player],
    views: dict,
    drafted: set[str] | None = None,
) -> list[Candidate]:
    """
    Join the player pool to market data.

    `views` is {player_id: market.MarketView}. Players the market never priced
    still come through — they just carry no ADP and survive by default.
    """
    drafted = drafted or set()
    out: list[Candidate] = []
    for p in players:
        v = views.get(p.player_id) if p.player_id else None
        out.append(
            Candidate(
                player=p,
                adp=v.home_adp if v else None,
                sd=v.adp_sd if v else None,
                gone=p.player_id in drafted,
            )
        )
    return out


def recommend(
    candidates: Sequence[Candidate],
    next_pick: int,
    *,
    positions: Iterable[str] | None = None,
    top_n: int = 15,
    pool_depth: int = 40,
) -> list[dict]:
    """
    Rank available players by what passing on them actually costs.

    Cost is computed per position: the alternative to taking a TE now is the
    best TE left at our next pick, not the best player overall. pool_depth
    caps how deep we look for that alternative — beyond the top few dozen at a
    position the VORP curve is flat and the extra terms contribute nothing.
    """
    live = [c for c in candidates if not c.gone]
    by_pos: dict[str, list[Candidate]] = {}
    for c in live:
        by_pos.setdefault(c.player.pos, []).append(c)
    for pool in by_pos.values():
        pool.sort(key=lambda c: c.vorp, reverse=True)

    wanted = set(positions) if positions else set(by_pos)
    rows: list[dict] = []
    for pos, pool in by_pos.items():
        if pos not in wanted:
            continue
        shortlist = pool[:pool_depth]
        for c in shortlist:
            rows.append({
                "player": c.player,
                "pos": pos,
                "vorp": c.vorp,
                "adp": c.adp,
                "sd": c.sd,
                "survives": c.survives(next_pick),
                "cost_of_waiting": cost_of_waiting(c, shortlist, next_pick),
            })
    rows.sort(key=lambda r: r["cost_of_waiting"], reverse=True)
    return rows[:top_n]
