"""
Market layer: what the room thinks, and how much it disagrees with itself.

Projections answer "what is he worth". This module answers two different
questions that the projection layer cannot:

  1. WHEN will he actually be gone — and in *our* room specifically, which
     drafts on Sleeper. Sleeper's ADP is the right predictor for that, and
     blending it into a cross-platform average destroys the signal.
  2. HOW SURE is anyone. Six independent platform ADPs give a measured
     per-player standard deviation instead of an assumed one. That spread is
     the parameter survival math needs: a player at ADP 40 +/- 3 and a player
     at ADP 40 +/- 20 are completely different decisions at pick 30.

Measured on the 2026 Flock export: Sleeper's mean signed difference from the
other five platforms is +0.4 picks (essentially unbiased in aggregate) but
mean ABSOLUTE difference is 8.8 picks, with extremes past 45. So the room is
not systematically early or late — it is idiosyncratically early or late on
specific players, which is exactly what is exploitable.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from projections import build_name_index, normalize_name
from vorp import Player


# The platform columns in Flock's ADP export, in file order.
PLATFORMS = ["Sleeper", "ESPN", "Yahoo", "Underdog", "CBS", "FFPC"]

# Our league drafts here, so this platform predicts our room; the rest form
# the outside-market baseline we measure it against.
HOME_PLATFORM = "Sleeper"

# Nickname variants that pure string folding cannot bridge. Sleeper and the
# ranking sites disagree on legal-vs-common first names. Keyed by the
# normalized name as it appears in the CSV.
NAME_ALIASES = {
    "kenneth gainwell": "kenny gainwell",
    "chigoziem okonkwo": "chig okonkwo",
    "joshua palmer": "josh palmer",
    "cameron ward": "cam ward",
    "michael penix jr": "michael penix",
    "marquise brown": "hollywood brown",
    "gabriel davis": "gabe davis",
}

_POS_RANK = re.compile(r"\d+$")


@dataclass
class MarketView:
    """Everything the market says about one player."""

    player_id: str
    name: str
    pos: str
    team: str | None = None

    platform_adp: dict[str, float] = field(default_factory=dict)
    expert_rank: float | None = None       # from the expert rankings file
    expert_adp: float | None = None        # Flock's own expert ADP column

    # ---- derived ----

    @property
    def home_adp(self) -> float | None:
        """ADP on the platform we actually draft on."""
        return self.platform_adp.get(HOME_PLATFORM)

    @property
    def outside_adp(self) -> list[float]:
        return [v for k, v in self.platform_adp.items() if k != HOME_PLATFORM]

    @property
    def market_adp(self) -> float | None:
        """Mean across every platform that priced him."""
        vals = list(self.platform_adp.values())
        return statistics.mean(vals) if vals else None

    @property
    def outside_market_adp(self) -> float | None:
        """Mean excluding our own platform — the honest outside baseline."""
        vals = self.outside_adp
        return statistics.mean(vals) if vals else None

    @property
    def adp_sd(self) -> float | None:
        """
        Cross-platform standard deviation: how much the market disagrees.

        This is the survival-curve width. None when fewer than two platforms
        priced him, because a standard deviation of one number is a lie.
        """
        vals = list(self.platform_adp.values())
        return statistics.stdev(vals) if len(vals) > 1 else None

    @property
    def n_platforms(self) -> int:
        return len(self.platform_adp)

    @property
    def home_vs_market(self) -> float | None:
        """
        Our platform minus the outside market. Positive = he lasts LONGER in
        our room than the market implies (don't reach). Negative = our room
        takes him EARLIER (go get him or forget him).
        """
        home, outside = self.home_adp, self.outside_market_adp
        return None if home is None or outside is None else home - outside

    @property
    def expert_vs_market(self) -> float | None:
        """
        Expert rank minus market ADP. Negative = experts rate him higher than
        the room drafts him, i.e. the crowd is late on a good player.
        """
        mkt = self.market_adp
        exp = self.expert_rank if self.expert_rank is not None else self.expert_adp
        return None if mkt is None or exp is None else exp - mkt

    @property
    def expert_vs_home(self) -> float | None:
        """Expert rank minus our room's ADP — the version that matters to us."""
        exp = self.expert_rank if self.expert_rank is not None else self.expert_adp
        return None if self.home_adp is None or exp is None else exp - self.home_adp


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _strip_pos_rank(pos: str) -> str:
    """Flock writes 'RB1', 'WR12' — the positional rank rides along with it."""
    return _POS_RANK.sub("", (pos or "").strip().upper())


def _as_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


class Matcher:
    """
    Resolve CSV rows to Sleeper player_ids.

    Three passes, most specific first. Exact name+position gets 199 of the top
    200 on the 2026 Flock export; the rest is nickname drift, which is why the
    alias table and the surname fallback exist. Anything still unmatched is
    returned to the caller rather than dropped — a missed startable player is
    a hole in the board you want to see before the draft, not after.
    """

    def __init__(self, players: Iterable[Player]):
        players = list(players)
        self.by_name_pos = build_name_index(players)
        self.by_surname: dict[tuple[str, str], list[Player]] = {}
        for p in players:
            parts = normalize_name(p.name).split()
            if parts:
                self.by_surname.setdefault((parts[-1], p.pos), []).append(p)

    def match(self, name: str, pos: str, team: str | None = None) -> Player | None:
        norm = normalize_name(name)
        pos = (pos or "").strip().upper()

        hit = self.by_name_pos.get((norm, pos))
        if hit is not None:
            return hit

        aliased = NAME_ALIASES.get(norm)
        if aliased:
            hit = self.by_name_pos.get((aliased, pos))
            if hit is not None:
                return hit

        # Surname + position, disambiguated by team when the CSV supplies one.
        parts = norm.split()
        if parts:
            candidates = self.by_surname.get((parts[-1], pos), [])
            if len(candidates) == 1:
                return candidates[0]
            if team:
                team = team.strip().upper()
                on_team = [c for c in candidates if (c.team or "").upper() == team]
                if len(on_team) == 1:
                    return on_team[0]
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_adp(
    path: str | Path, players: Iterable[Player], *, platforms: Iterable[str] = PLATFORMS
) -> tuple[dict[str, MarketView], list[str]]:
    """
    Read Flock's ADP export into MarketViews keyed by Sleeper player_id.

    Platform coverage thins out with depth (all six for roughly the top 100,
    around four by pick 300), so every consumer has to tolerate a missing
    platform. We record what is present and let n_platforms speak for itself
    rather than imputing.
    """
    platforms = list(platforms)
    matcher = Matcher(players)
    views: dict[str, MarketView] = {}
    unmatched: list[str] = []

    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Player") or "").strip()
            pos = _strip_pos_rank(row.get("POS", ""))
            team = (row.get("Team") or "").strip() or None
            if not name or not pos:
                continue

            hit = matcher.match(name, pos, team)
            if hit is None or hit.player_id is None:
                unmatched.append(f"{name} ({pos}, {team or 'FA'})")
                continue
            # A duplicate name deeper in the file must not clobber the real
            # player already matched higher up.
            if hit.player_id in views:
                continue

            adp = {}
            for plat in platforms:
                val = _as_float(row.get(plat))
                if val is not None:
                    adp[plat] = val

            views[hit.player_id] = MarketView(
                player_id=hit.player_id, name=hit.name, pos=hit.pos,
                team=hit.team, platform_adp=adp,
                expert_adp=_as_float(row.get("Expert")),
            )

    return views, unmatched


def load_expert_ranks(
    path: str | Path, players: Iterable[Player]
) -> tuple[dict[str, float], list[str]]:
    """Read the expert consensus rankings into {player_id: expert_rank}."""
    matcher = Matcher(players)
    ranks: dict[str, float] = {}
    unmatched: list[str] = []

    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").strip()
            pos = _strip_pos_rank(row.get("Position", ""))
            team = (row.get("Team") or "").strip() or None
            rank = _as_float(row.get("Expert Rank"))
            if not name or rank is None:
                continue
            hit = matcher.match(name, pos, team)
            if hit is None or hit.player_id is None:
                unmatched.append(f"{name} ({pos}, {team or 'FA'})")
                continue
            ranks.setdefault(hit.player_id, rank)

    return ranks, unmatched


def merge_expert_ranks(
    views: dict[str, MarketView], ranks: dict[str, float]
) -> int:
    applied = 0
    for pid, rank in ranks.items():
        if pid in views:
            views[pid].expert_rank = rank
            applied += 1
    return applied


def load_all(
    adp_path: str | Path,
    expert_path: str | Path | None,
    players: Iterable[Player],
) -> tuple[dict[str, MarketView], dict[str, list[str]]]:
    """Both files in, MarketViews plus a report of what failed to match."""
    players = list(players)
    views, adp_missed = load_adp(adp_path, players)
    report = {"adp_unmatched": adp_missed, "expert_unmatched": []}
    if expert_path:
        ranks, exp_missed = load_expert_ranks(expert_path, players)
        merge_expert_ranks(views, ranks)
        report["expert_unmatched"] = exp_missed
    return views, report


# ---------------------------------------------------------------------------
# Divergence scoring
# ---------------------------------------------------------------------------


def zscores(values: dict[str, float]) -> dict[str, float]:
    """
    Standardize a divergence measure across players.

    Normalizing globally would be wrong on its own — a 10-pick disagreement at
    ADP 5 is enormous and at ADP 150 is noise — so callers should z-score
    within a depth band. This helper does the arithmetic; banding is the
    caller's job (see divergence_table).
    """
    if len(values) < 2:
        return {k: 0.0 for k in values}
    vals = list(values.values())
    mean = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    if sd == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / sd for k, v in values.items()}


def divergence_table(
    views: dict[str, MarketView],
    metric: str = "home_vs_market",
    *,
    band_size: int = 50,
    min_platforms: int = 3,
) -> dict[str, float]:
    """
    Z-score `metric` within depth bands of `band_size` picks.

    Banding is what keeps the ranking honest: without it, deep players with
    huge absolute divergences swamp genuinely mispriced early-round players,
    the same way raw ADP-minus-rank let kickers dominate the value tab.
    """
    scored: dict[str, float] = {}
    ranked = [
        v for v in views.values()
        if v.market_adp is not None and v.n_platforms >= min_platforms
        and getattr(v, metric) is not None
    ]
    ranked.sort(key=lambda v: v.market_adp)

    for start in range(0, len(ranked), band_size):
        band = ranked[start:start + band_size]
        raw = {v.player_id: float(getattr(v, metric)) for v in band}
        scored.update(zscores(raw))
    return scored
