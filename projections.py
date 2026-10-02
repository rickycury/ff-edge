"""
Projection ingest: get one number per player into Player.proj.

Deliberately source-agnostic. A source's only job is to produce
{player_id: ProjectedLine}; everything downstream (VORP, tiers, survival)
is identical regardless of where the numbers came from. Swapping Sleeper's
feed for FantasyPros consensus should be a new function in this file, not a
change anywhere else.

Two sources ship today:

  * from_sleeper()  — Sleeper's projections host. Free, no signup, and keyed
    by player_id so there is no name matching to get wrong. Currently sourced
    entirely from rotowire, i.e. ONE provider's numbers, not a consensus.
  * from_csv()      — any CSV with name/position/points columns. This is the
    upgrade path; it has to match on names, which is the part that goes wrong,
    so it reports its misses loudly instead of silently dropping players.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import sleeper
from sleeper import SleeperClient
from vorp import Player


# Sleeper's projections live on a different host from the v1 API.
PROJECTIONS_BASE = "https://api.sleeper.com"
TTL_PROJECTIONS = 6 * 60 * 60  # numbers move slowly outside game weeks

# Which stat key holds "points" depends on the league's receptions scoring.
SCORING_KEYS = {"ppr": "pts_ppr", "half_ppr": "pts_half_ppr", "std": "pts_std"}
ADP_KEYS = {"ppr": "adp_ppr", "half_ppr": "adp_half_ppr", "std": "adp_std"}

# order_by is part of the query string and therefore part of the cache key,
# but every row carries all three pts_* variants regardless of how the server
# sorted them — and we index by player_id, so the order is meaningless to us.
# Pinning it means one fetch and one cache entry serves PPR, half and standard
# instead of three identical 3300-row downloads.
ORDER_BY = "pts_ppr"


@dataclass
class ProjectedLine:
    proj: float
    adp: float | None = None


@dataclass
class AttachReport:
    """What happened when projections met the player pool."""

    matched: int = 0
    unmatched_players: list[Player] = None  # players left at 0.0
    unused_source_ids: int = 0              # projections with nobody to attach to

    def __post_init__(self) -> None:
        if self.unmatched_players is None:
            self.unmatched_players = []

    def summary(self) -> str:
        return (
            f"{self.matched} players projected, "
            f"{len(self.unmatched_players)} left unprojected, "
            f"{self.unused_source_ids} source rows unused"
        )


def scoring_kind(league: dict[str, Any]) -> str:
    """
    Derive PPR / half / standard from the league's own scoring_settings.

    Reading it from the league is the same principle as roster_positions:
    the league defines itself, we don't hardcode a guess per league.
    """
    rec = (league.get("scoring_settings") or {}).get("rec")
    try:
        rec = float(rec)
    except (TypeError, ValueError):
        return "ppr"
    if rec >= 0.75:
        return "ppr"
    if rec >= 0.25:
        return "half_ppr"
    return "std"


# ---------------------------------------------------------------------------
# Source: Sleeper
# ---------------------------------------------------------------------------


def projections_client(
    cache_dir: str | Path = ".cache", *, offline: bool = False
) -> SleeperClient:
    """A client pointed at the projections host, sharing the global rate budget."""
    return SleeperClient(cache_dir, offline=offline, base_url=PROJECTIONS_BASE)


def sleeper_projections_endpoint(
    season: str | int, positions: Iterable[str], order_by: str = ORDER_BY
) -> str:
    qs = "&".join(f"position[]={p}" for p in sorted(positions))
    return (
        f"projections/nfl/{season}?season_type=regular&{qs}&order_by={order_by}"
    )


def from_sleeper(
    client: SleeperClient,
    season: str | int,
    *,
    positions: Iterable[str] = ("QB", "RB", "WR", "TE", "K", "DEF"),
    kind: str = "ppr",
    ttl: float = TTL_PROJECTIONS,
) -> dict[str, ProjectedLine]:
    """
    Season-long projections keyed by Sleeper player_id.

    Rows without a points value are skipped rather than coerced to 0.0 — most
    of the payload is deep-bench players with no projection at all, and
    treating "no projection" as "projected zero" would plant them at the
    replacement baseline instead of below it.
    """
    pts_key = SCORING_KEYS.get(kind, "pts_ppr")
    adp_key = ADP_KEYS.get(kind, "adp_ppr")
    endpoint = sleeper_projections_endpoint(season, positions)
    rows = client.get(endpoint, ttl) or []

    out: dict[str, ProjectedLine] = {}
    for row in rows:
        stats = row.get("stats") or {}
        pts = stats.get(pts_key)
        if pts is None:
            continue
        pid = row.get("player_id")
        if pid is None:
            continue
        adp = stats.get(adp_key)
        out[str(pid)] = ProjectedLine(
            proj=float(pts),
            adp=float(adp) if adp is not None else None,
        )
    return out


# ---------------------------------------------------------------------------
# Source: CSV
# ---------------------------------------------------------------------------

_SEPARATORS = re.compile(r"[-–—/]+")   # word breaks
_PUNCT = re.compile(r"[^a-z0-9 ]+")     # everything else, deleted
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """
    Fold a name to a matching key: strip accents and generational suffixes,
    delete intra-word punctuation, and treat hyphens as word breaks.

    The hyphen/apostrophe split is deliberate and comes from how sources
    actually differ. Apostrophes are dropped so "Ja'Marr" == "JaMarr".
    Hyphens become spaces so "Amon-Ra" == "Amon Ra" — deleting them instead
    yields "amonra", which fails against any source that writes the space.
    Name mismatches are the main way a CSV import silently loses a startable
    player, so this is worth being fussy about.
    """
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    folded = _SEPARATORS.sub(" ", folded)
    folded = _PUNCT.sub("", folded)
    parts = [p for p in folded.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def build_name_index(players: Iterable[Player]) -> dict[tuple[str, str], Player]:
    """(normalized name, pos) -> Player. Position is part of the key because
    name collisions across positions are real (e.g. two Michael Thomases)."""
    index: dict[tuple[str, str], Player] = {}
    for p in players:
        index[(normalize_name(p.name), p.pos)] = p
    return index


def from_csv(
    path: str | Path,
    players: Iterable[Player],
    *,
    name_col: str = "name",
    pos_col: str = "pos",
    points_col: str = "points",
    adp_col: str | None = None,
) -> tuple[dict[str, ProjectedLine], list[str]]:
    """
    Read a projections CSV and map it onto player_ids by name+position.

    Returns (projections, unmatched_rows). Unmatched rows are returned rather
    than logged and forgotten: a name-matching failure on a startable player
    is a silent hole in the board, and you want to see it before the draft.
    """
    index = build_name_index(players)
    out: dict[str, ProjectedLine] = {}
    unmatched: list[str] = []

    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        for wanted in (name_col, pos_col, points_col):
            if wanted.lower() not in cols:
                raise ValueError(
                    f"{path}: missing column {wanted!r}. Found: {reader.fieldnames}"
                )
        for row in reader:
            name = (row.get(cols[name_col.lower()]) or "").strip()
            pos = (row.get(cols[pos_col.lower()]) or "").strip().upper()
            raw_pts = (row.get(cols[points_col.lower()]) or "").strip()
            if not name or not raw_pts:
                continue
            try:
                pts = float(raw_pts.replace(",", ""))
            except ValueError:
                continue
            match = index.get((normalize_name(name), pos))
            if match is None or match.player_id is None:
                unmatched.append(f"{name} ({pos})")
                continue
            adp = None
            if adp_col and adp_col.lower() in cols:
                try:
                    adp = float(row[cols[adp_col.lower()]])
                except (TypeError, ValueError, KeyError):
                    adp = None
            out[match.player_id] = ProjectedLine(proj=pts, adp=adp)

    return out, unmatched


# ---------------------------------------------------------------------------
# Attaching
# ---------------------------------------------------------------------------


def attach(
    players: Iterable[Player], projections: dict[str, ProjectedLine]
) -> AttachReport:
    """
    Write projections onto Player.proj / Player.adp in place.

    Players with no projection keep proj=0.0. That is the honest encoding:
    they are worse than replacement and will sort to the bottom, which is
    where an unprojected deep-bench player belongs.
    """
    players = list(players)
    report = AttachReport()
    seen: set[str] = set()

    for p in players:
        line = projections.get(p.player_id) if p.player_id else None
        if line is None:
            report.unmatched_players.append(p)
            continue
        p.proj = line.proj
        if line.adp is not None:
            p.adp = line.adp
        report.matched += 1
        seen.add(p.player_id)

    report.unused_source_ids = len(set(projections) - seen)
    return report


def load_for_league(
    client: SleeperClient,
    league: dict[str, Any],
    players: list[Player],
    *,
    season: str | int | None = None,
    cache_dir: str | Path = ".cache",
    offline: bool = False,
) -> tuple[AttachReport, str]:
    """
    Convenience path: league payload in, projected players out.

    Returns (report, scoring_kind). The scoring kind comes from the league's
    own scoring_settings, so a half-PPR league gets half-PPR numbers without
    anyone remembering to pass a flag.
    """
    kind = scoring_kind(league)
    season = season or league.get("season")
    pc = projections_client(cache_dir, offline=offline)
    proj = from_sleeper(pc, season, kind=kind)
    return attach(players, proj), kind
