"""
Sleeper data layer: cached HTTP client, league config, live draft state.

The math layer (vorp.py) never makes network calls. Everything that touches
the wire lives here, behind a disk cache and a hard rate limiter.

Field shapes in this module were verified against live responses on
2026-08-05, not taken from documentation or memory. Where the live data
contradicted the obvious assumption, the surprise is noted at the point of
use — see normalize_players in particular.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from vorp import FLEX_ELIGIBILITY, IGNORED_SLOTS, LeagueConfig, Player


BASE_URL = "https://api.sleeper.app/v1"
USER_AGENT = "ff-edge/0.1 (+personal fantasy tool)"

# Cache lifetimes, seconds.
TTL_PLAYERS = 24 * 60 * 60  # ~5MB on the wire; Sleeper asks for daily at most
TTL_LEAGUE = 10 * 60        # league/rosters/users/drafts: slow-moving
TTL_DRAFT_PICKS = 3.0       # live draft polling floor; callers may override
TTL_STATE = 10 * 60

# Sleeper's documented ceiling is 1000 calls/min per IP, above which the IP is
# blocked. We default below it: our minute and their minute are not the same
# sliding window, and being blocked mid-draft is unrecoverable in the moment.
RATE_LIMIT_CALLS = 900
RATE_LIMIT_PERIOD = 60.0


class SleeperError(RuntimeError):
    """Base for everything this module raises."""


class OfflineError(SleeperError):
    """A cache miss occurred on a client that is forbidden from using the network."""


class LeagueConfigError(SleeperError, ValueError):
    """roster_positions contained something we refuse to interpret."""


class DraftShapeError(SleeperError, ValueError):
    """A draft payload was missing or self-contradictory in a way that would
    make pick ordering wrong."""


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Hard sliding-window limiter. Blocks rather than exceeding the budget.

    This counts *network calls only* — cache hits never reach it. That's the
    point: the cache is an optimization and its hit rate depends on what the
    caller asks for, so it cannot be the thing keeping us under the limit. A
    tight poll loop on a 3s-TTL endpoint will still generate real traffic, and
    this is what stops it.

    clock/sleep are injectable so tests can drive it without real time.
    """

    def __init__(
        self,
        max_calls: int = RATE_LIMIT_CALLS,
        period: float = RATE_LIMIT_PERIOD,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()
        self.total_waited = 0.0
        self.total_calls = 0

    def _evict(self, now: float) -> None:
        cutoff = now - self.period
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def acquire(self) -> float:
        """Reserve one call slot, blocking if needed. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._evict(now)
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    self.total_calls += 1
                    self.total_waited += waited
                    return waited
                # Sleep until the oldest call ages out of the window, then
                # re-check under the lock — another thread may have taken the
                # slot we were waiting for.
                wait = self._calls[0] + self.period - now
            wait = max(wait, 0.0)
            self._sleep(wait)
            waited += wait

    @property
    def in_window(self) -> int:
        with self._lock:
            self._evict(self._clock())
            return len(self._calls)


_SHARED_LIMITER: RateLimiter | None = None
_SHARED_LOCK = threading.Lock()


def shared_limiter() -> RateLimiter:
    """
    One limiter for the whole process. The rate limit is per-IP, so every
    client — including projections.py, which talks to a different Sleeper
    host — has to draw from the same budget.
    """
    global _SHARED_LIMITER
    with _SHARED_LOCK:
        if _SHARED_LIMITER is None:
            _SHARED_LIMITER = RateLimiter()
        return _SHARED_LIMITER


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

_MISS = object()
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class DiskCache:
    """
    JSON-on-disk cache keyed by endpoint path.

    Filenames are a readable slug plus a hash of the full endpoint, so you can
    eyeball .cache/ during a draft and still never collide after slugging.
    """

    def __init__(self, root: str | Path = ".cache"):
        self.root = Path(root)

    def key_for(self, endpoint: str) -> str:
        trimmed = endpoint.strip("/")
        slug = _UNSAFE.sub("_", trimmed)[:80] or "root"
        digest = hashlib.sha1(trimmed.encode("utf-8")).hexdigest()[:10]
        return f"{slug}.{digest}.json"

    def path_for(self, endpoint: str) -> Path:
        return self.root / self.key_for(endpoint)

    def get(self, endpoint: str, ttl: float, *, now: float | None = None) -> Any:
        """Return the cached payload, or the _MISS sentinel."""
        if ttl <= 0:
            return _MISS
        path = self.path_for(endpoint)
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            # Missing, unreadable, or half-written by a killed process. Either
            # way there is nothing to trust; refetch and overwrite.
            return _MISS
        try:
            fetched_at = float(entry["fetched_at"])
            payload = entry["payload"]
        except (KeyError, TypeError, ValueError):
            return _MISS
        age = (time.time() if now is None else now) - fetched_at
        if age > ttl:
            return _MISS
        return payload

    def put(self, endpoint: str, payload: Any, *, now: float | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(endpoint)
        entry = {
            "endpoint": endpoint,
            "fetched_at": time.time() if now is None else now,
            "payload": payload,
        }
        # Write-then-rename so a crash mid-write can't leave a truncated file
        # that a later run would have to parse.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        tmp.replace(path)

    def age_of(self, endpoint: str) -> float | None:
        path = self.path_for(endpoint)
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
            return time.time() - float(entry["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def http_transport(
    endpoint: str, timeout: float = 15.0, base_url: str = BASE_URL
) -> Any:
    """
    Fetch one endpoint and return decoded JSON.

    gzip is requested explicitly: /players/nfl is ~14.6MB of JSON that
    compresses to ~2.5MB on the wire, which is the difference between a
    tolerable daily refresh and a painful one.
    """
    url = f"{base_url}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        raise SleeperError(f"GET {url} -> HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SleeperError(f"GET {url} failed: {exc.reason}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise SleeperError(f"GET {url} returned non-JSON body") from exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SleeperClient:
    """
    Read-only Sleeper client: disk cache in front, rate limiter behind.

    Set offline=True and every cache miss raises OfflineError instead of
    reaching the network. Tests use that plus a fixture transport, so a test
    that accidentally asks for an unrecorded endpoint fails loudly rather than
    quietly making a real request.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        *,
        transport: Callable[[str], Any] | None = None,
        rate_limiter: RateLimiter | None = None,
        offline: bool = False,
        timeout: float = 15.0,
        base_url: str = BASE_URL,
    ):
        self.cache = DiskCache(cache_dir)
        # Default to the process-wide limiter, not a fresh one. Sleeper's limit
        # is per-IP, so two clients each holding their own budget would let us
        # spend 2x the ceiling — exactly the failure the limiter exists to
        # prevent. projections.py talks to a different host on the same IP.
        self.limiter = rate_limiter if rate_limiter is not None else shared_limiter()
        self.offline = offline
        self.base_url = base_url
        self._timeout = timeout
        self._transport = transport or (
            lambda ep: http_transport(ep, timeout, base_url)
        )
        self.network_calls = 0
        self.cache_hits = 0

    def get(self, endpoint: str, ttl: float) -> Any:
        endpoint = endpoint.strip("/")
        payload = self.cache.get(endpoint, ttl)
        if payload is not _MISS:
            self.cache_hits += 1
            return payload
        if self.offline:
            raise OfflineError(
                f"cache miss for {endpoint!r} on an offline client "
                f"(ttl={ttl}). Record a fixture for it or widen the TTL."
            )
        self.limiter.acquire()
        payload = self._transport(endpoint)
        self.network_calls += 1
        self.cache.put(endpoint, payload)
        return payload

    # ---- endpoints ------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return self.get("state/nfl", TTL_STATE)

    def get_league(self, league_id: str) -> dict[str, Any]:
        payload = self.get(f"league/{league_id}", TTL_LEAGUE)
        if not isinstance(payload, dict):
            raise SleeperError(
                f"league/{league_id} returned {type(payload).__name__}, not an "
                f"object — is the league_id right?"
            )
        return payload

    def get_rosters(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"league/{league_id}/rosters", TTL_LEAGUE) or []

    def get_users(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"league/{league_id}/users", TTL_LEAGUE) or []

    def get_matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self.get(f"league/{league_id}/matchups/{week}", TTL_LEAGUE) or []

    def get_transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self.get(f"league/{league_id}/transactions/{week}", TTL_LEAGUE) or []

    def get_drafts(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"league/{league_id}/drafts", TTL_LEAGUE) or []

    def get_traded_picks(
        self, league_id: str, ttl: float = TTL_LEAGUE
    ) -> list[dict[str, Any]]:
        """
        Draft picks that have changed hands. Keyed by (season, round,
        roster_id) where roster_id is the pick's *original* owner and
        owner_id is who holds it now.

        This is not optional bookkeeping: in the recorded 2025 RE Fantasy
        draft, 62 of 192 picks were traded. Without this, pick ownership is
        wrong for a third of the draft.

        ttl is caller-supplied because leagues can trade picks mid-draft.
        """
        return self.get(f"league/{league_id}/traded_picks", ttl) or []

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        payload = self.get(f"draft/{draft_id}", TTL_LEAGUE)
        if not isinstance(payload, dict):
            raise SleeperError(f"draft/{draft_id} returned no object")
        return payload

    def get_draft_picks(
        self, draft_id: str, ttl: float = TTL_DRAFT_PICKS
    ) -> list[dict[str, Any]]:
        """
        Picks so far. ttl is caller-supplied because this is the one endpoint
        polled live; the default is a floor, not a recommendation. Polling
        faster than a few seconds is what gets an IP blocked.
        """
        return self.get(f"draft/{draft_id}/picks", ttl) or []

    def get_players(self, ttl: float = TTL_PLAYERS) -> dict[str, Any]:
        payload = self.get("players/nfl", ttl)
        if not isinstance(payload, dict):
            raise SleeperError("players/nfl returned no object")
        return payload

    def get_user(self, username_or_id: str) -> dict[str, Any]:
        payload = self.get(f"user/{username_or_id}", TTL_LEAGUE)
        if not isinstance(payload, dict):
            raise SleeperError(f"no Sleeper user {username_or_id!r}")
        return payload

    def get_user_leagues(self, user_id: str, season: str | int) -> list[dict[str, Any]]:
        return self.get(f"user/{user_id}/leagues/nfl/{season}", TTL_LEAGUE) or []

    # ---- convenience ----------------------------------------------------

    def previous_league_id(self, league_id: str) -> str | None:
        """
        The same league's prior season, via previous_league_id. Useful because
        a league that hasn't drafted yet has an empty picks array — last
        season's completed draft is the only one that can validate ordering.
        """
        return self.get_league(league_id).get("previous_league_id") or None


# ---------------------------------------------------------------------------
# League config
# ---------------------------------------------------------------------------

# Slots we accept as real, single-position starting slots. Anything else in
# roster_positions is either a flex type (must be in vorp.FLEX_ELIGIBILITY),
# an ignored slot (vorp.IGNORED_SLOTS), or something new that we refuse to
# guess about. Sleeper adds slot types; we want a traceback, not a silent
# reinterpretation of the league.
KNOWN_DEDICATED_SLOTS = {
    "QB", "RB", "WR", "TE", "K", "DEF",
    "DL", "LB", "DB", "IDP",  # IDP leagues; harmless if unused
}


def _validate_roster_positions(roster_positions: list[str]) -> None:
    """
    Reject any slot we can't account for, before LeagueConfig sees it.

    Two distinct failure modes, both loud:

    1. An unknown *_FLEX type. LeagueConfig routes anything not in
       FLEX_ELIGIBILITY into `dedicated`, so a future IDP_FLEX would quietly
       become "one dedicated IDP_FLEX starter" and skew replacement level for
       every position it should have been pulling from.
    2. An unknown single-position slot. Same routing, same problem, but it
       also means scored_positions gains a position with no players in it.
    """
    for slot in roster_positions:
        if slot in IGNORED_SLOTS or slot in FLEX_ELIGIBILITY:
            continue
        if "FLEX" in slot:
            raise LeagueConfigError(
                f"unknown flex slot {slot!r} in roster_positions. "
                f"vorp.FLEX_ELIGIBILITY knows {sorted(FLEX_ELIGIBILITY)}. "
                f"Add {slot!r} there with its eligible positions — leaving it "
                f"would silently count it as a dedicated position."
            )
        if slot not in KNOWN_DEDICATED_SLOTS:
            raise LeagueConfigError(
                f"unknown roster slot {slot!r} in roster_positions. "
                f"Add it to sleeper.KNOWN_DEDICATED_SLOTS if it is a real "
                f"starting position, or to vorp.IGNORED_SLOTS if it is not."
            )


def _assert_slots_round_trip(league: LeagueConfig, raw: list[str]) -> None:
    """
    Rebuild the raw slot multiset from what LeagueConfig parsed out and
    require an exact match. Catches slots dropped on the floor by a future
    change to __post_init__, independent of the name-level checks above.
    """
    rebuilt: dict[str, int] = {}
    for bucket in (league.dedicated, league.flex_slots):
        for slot, n in bucket.items():
            rebuilt[slot] = rebuilt.get(slot, 0) + n
    for slot in raw:
        if slot in IGNORED_SLOTS:
            rebuilt[slot] = rebuilt.get(slot, 0) + 1

    counted = dict(collections.Counter(raw))
    if rebuilt != counted:
        mismatched = {
            k: {"raw": counted.get(k, 0), "parsed": rebuilt.get(k, 0)}
            for k in set(counted) | set(rebuilt)
            if counted.get(k, 0) != rebuilt.get(k, 0)
        }
        raise LeagueConfigError(
            f"roster_positions did not round-trip through LeagueConfig. "
            f"raw={counted} parsed={rebuilt} mismatched={mismatched}"
        )


def build_league_config(client: SleeperClient, league_id: str) -> LeagueConfig:
    """
    Pull roster_positions and total_rosters straight from the league endpoint.

    Nothing about either league is hardcoded here — this is the only place
    league shape enters the system, and it enters verbatim.
    """
    league = client.get_league(league_id)

    roster_positions = league.get("roster_positions")
    if not isinstance(roster_positions, list) or not roster_positions:
        raise LeagueConfigError(
            f"league {league_id}: roster_positions missing or not a list "
            f"(got {roster_positions!r})"
        )
    if not all(isinstance(s, str) for s in roster_positions):
        raise LeagueConfigError(
            f"league {league_id}: roster_positions contains non-strings: "
            f"{roster_positions!r}"
        )

    total_rosters = league.get("total_rosters")
    if not isinstance(total_rosters, int) or total_rosters < 1:
        raise LeagueConfigError(
            f"league {league_id}: total_rosters missing or not a positive int "
            f"(got {total_rosters!r})"
        )

    _validate_roster_positions(roster_positions)
    cfg = LeagueConfig(n_teams=total_rosters, roster_positions=list(roster_positions))
    _assert_slots_round_trip(cfg, roster_positions)

    if not cfg.dedicated and not cfg.flex_slots:
        raise LeagueConfigError(
            f"league {league_id}: no startable slots parsed from "
            f"{roster_positions!r} — every slot was ignored"
        )
    return cfg


# ---------------------------------------------------------------------------
# Draft pick ordering
# ---------------------------------------------------------------------------

SNAKE_TYPES = {"snake"}
LINEAR_TYPES = {"linear"}
AUCTION_TYPES = {"auction"}


def pick_number(
    round_: int,
    slot: int,
    n_teams: int,
    *,
    snake: bool,
    reversal_round: int = 0,
) -> int:
    """
    Overall (1-indexed) pick number for a given draft slot in a given round.

    Linear: every round runs 1..N, so slot s always picks (r-1)*N + s.

    Snake: odd rounds run 1..N, even rounds run N..1.

    Snake with reversal (Sleeper's `settings.reversal_round`, 0 = off): the
    reversal round repeats the previous round's direction instead of flipping,
    which inverts the parity for that round and every round after it. With
    reversal_round=3 and N=12: R1 = 1..12, R2 = 12..1, R3 = 12..1 (the
    reversal), R4 = 1..12. So the rule is just "flip the normal snake
    direction from the reversal round onward".
    """
    if n_teams < 1:
        raise ValueError(f"n_teams must be >= 1, got {n_teams}")
    if not 1 <= slot <= n_teams:
        raise ValueError(f"slot {slot} out of range for {n_teams}-team draft")
    if round_ < 1:
        raise ValueError(f"round must be >= 1, got {round_}")

    descending = snake and round_ % 2 == 0
    if snake and reversal_round and round_ >= reversal_round:
        descending = not descending

    pos_in_round = (n_teams + 1 - slot) if descending else slot
    return (round_ - 1) * n_teams + pos_in_round


@dataclass
class DraftMeta:
    """The subset of GET /draft/{id} that pick ordering depends on."""

    draft_id: str
    type: str
    status: str
    n_teams: int
    rounds: int
    reversal_round: int
    draft_order: dict[str, int]        # user_id -> draft slot (1-indexed)
    slot_to_roster_id: dict[int, int]  # draft slot -> roster_id
    league_id: str | None = None
    season: str | None = None

    @classmethod
    def from_payload(cls, draft: dict[str, Any]) -> "DraftMeta":
        draft_id = str(draft.get("draft_id", "?"))
        settings = draft.get("settings") or {}

        n_teams = settings.get("teams")
        rounds = settings.get("rounds")
        if not isinstance(n_teams, int) or n_teams < 1:
            raise DraftShapeError(
                f"draft {draft_id}: settings.teams missing or invalid ({n_teams!r})"
            )
        if not isinstance(rounds, int) or rounds < 1:
            raise DraftShapeError(
                f"draft {draft_id}: settings.rounds missing or invalid ({rounds!r})"
            )

        # JSON object keys are strings; slot_to_roster_id keys are slot numbers.
        raw_slots = draft.get("slot_to_roster_id") or {}
        slot_to_roster_id: dict[int, int] = {}
        for slot, roster_id in raw_slots.items():
            try:
                slot_to_roster_id[int(slot)] = roster_id
            except (TypeError, ValueError):
                raise DraftShapeError(
                    f"draft {draft_id}: non-integer key {slot!r} in slot_to_roster_id"
                ) from None

        raw_order = draft.get("draft_order") or {}
        draft_order: dict[str, int] = {}
        for user_id, slot in raw_order.items():
            try:
                draft_order[str(user_id)] = int(slot)
            except (TypeError, ValueError):
                raise DraftShapeError(
                    f"draft {draft_id}: non-integer slot {slot!r} for user {user_id!r}"
                ) from None

        if slot_to_roster_id and len(slot_to_roster_id) != n_teams:
            raise DraftShapeError(
                f"draft {draft_id}: settings.teams={n_teams} but "
                f"slot_to_roster_id has {len(slot_to_roster_id)} slots "
                f"{sorted(slot_to_roster_id)}. Pick ordering depends on the "
                f"team count, so this disagreement is not safe to guess past."
            )

        return cls(
            draft_id=draft_id,
            type=str(draft.get("type") or ""),
            status=str(draft.get("status") or ""),
            n_teams=n_teams,
            rounds=rounds,
            # reversal_round is absent on most drafts and 0 when disabled.
            reversal_round=int(settings.get("reversal_round") or 0),
            draft_order=draft_order,
            slot_to_roster_id=slot_to_roster_id,
            league_id=str(draft["league_id"]) if draft.get("league_id") else None,
            season=str(draft["season"]) if draft.get("season") else None,
        )


def build_traded_map(
    traded_picks: Iterable[dict[str, Any]], season: str | None
) -> dict[tuple[int, int], int]:
    """
    (round, original_roster_id) -> current owner roster_id, for one season.

    The season filter matters: /league/{id}/traded_picks returns future-season
    picks too (dynasty leagues trade them years out), and applying a 2027 trade
    to the 2025 draft would silently hand picks to the wrong team.
    """
    out: dict[tuple[int, int], int] = {}
    for t in traded_picks:
        if season is not None and str(t.get("season")) != str(season):
            continue
        rnd, orig, owner = t.get("round"), t.get("roster_id"), t.get("owner_id")
        if rnd is None or orig is None or owner is None:
            continue
        out[(int(rnd), int(orig))] = int(owner)
    return out


class DraftState:
    """Live view of one draft: who's gone, and when I'm up next."""

    def __init__(
        self,
        meta: DraftMeta,
        picks: list[dict[str, Any]],
        traded_picks: Iterable[dict[str, Any]] | None = None,
    ):
        self.meta = meta
        self.picks = list(picks)
        self.traded = build_traded_map(traded_picks or [], meta.season)

    @classmethod
    def from_client(
        cls,
        client: SleeperClient,
        draft_id: str,
        *,
        picks_ttl: float = TTL_DRAFT_PICKS,
        traded_ttl: float = TTL_LEAGUE,
        include_traded: bool = True,
    ) -> "DraftState":
        meta = DraftMeta.from_payload(client.get_draft(draft_id))
        traded: list[dict[str, Any]] = []
        if include_traded and meta.league_id:
            traded = client.get_traded_picks(meta.league_id, ttl=traded_ttl)
        return cls(meta, client.get_draft_picks(draft_id, ttl=picks_ttl), traded)

    def refresh(
        self, client: SleeperClient, *, picks_ttl: float = TTL_DRAFT_PICKS
    ) -> "DraftState":
        """Re-poll picks only. The draft object itself does not change mid-draft."""
        self.picks = list(client.get_draft_picks(self.meta.draft_id, ttl=picks_ttl))
        return self

    # ---- what's gone ----------------------------------------------------

    @property
    def drafted_player_ids(self) -> set[str]:
        return {
            str(p["player_id"]) for p in self.picks if p.get("player_id") is not None
        }

    @property
    def picks_made(self) -> int:
        """
        Highest overall pick number recorded, not len(picks) — Sleeper can
        return picks out of order, and an unfilled gap would make a length
        count understate where the draft actually is.
        """
        nos = [p["pick_no"] for p in self.picks if p.get("pick_no") is not None]
        return max(nos) if nos else 0

    def picks_by_slot(self, slot: int) -> list[dict[str, Any]]:
        return [p for p in self.picks if p.get("draft_slot") == slot]

    def picks_by_user(self, user_id: str) -> list[dict[str, Any]]:
        slots = set(self.slots_for_user(user_id))
        return [p for p in self.picks if p.get("draft_slot") in slots]

    def available(self, players: Iterable[Player]) -> list[Player]:
        gone = self.drafted_player_ids
        return [p for p in players if p.player_id not in gone]

    # ---- ordering -------------------------------------------------------

    def slots_for_user(self, user_id: str) -> list[int]:
        """
        Draft slots this user controls.

        draft_order is a JSON object keyed by user_id, so in practice this is
        always exactly one slot — a single user_id cannot appear twice. The
        list return is defensive, not speculative: it keeps the caller honest
        if a co-owner or multi-team arrangement ever has to be resolved
        through slot_to_roster_id instead.

        slot_to_roster_id is the authority on which slots exist at all, so a
        draft_order entry pointing at a slot with no roster is a data problem
        worth failing on rather than drafting against.
        """
        user_id = str(user_id)
        slots = sorted(
            slot for uid, slot in self.meta.draft_order.items() if uid == user_id
        )
        if not slots:
            raise KeyError(
                f"user_id {user_id!r} has no slot in draft {self.meta.draft_id}. "
                f"Known user_ids: {sorted(self.meta.draft_order)}"
            )
        unknown = [s for s in slots if s not in self.meta.slot_to_roster_id]
        if unknown and self.meta.slot_to_roster_id:
            raise DraftShapeError(
                f"draft_order puts user {user_id!r} in slot(s) {unknown}, which "
                f"are absent from slot_to_roster_id "
                f"{sorted(self.meta.slot_to_roster_id)}"
            )
        return slots

    def _is_snake(self) -> bool:
        dtype = self.meta.type
        if dtype in AUCTION_TYPES:
            raise NotImplementedError(
                f"draft {self.meta.draft_id} is an auction draft; there is no "
                f"pick order to compute a next pick from. Nomination order is "
                f"not a pick sequence, and returning a number here would be "
                f"wrong rather than approximate. Auction budgets need their "
                f"own model."
            )
        if dtype in SNAKE_TYPES:
            return True
        if dtype in LINEAR_TYPES:
            return False
        raise DraftShapeError(
            f"draft {self.meta.draft_id}: unrecognized draft type {dtype!r} "
            f"(known: {sorted(SNAKE_TYPES | LINEAR_TYPES | AUCTION_TYPES)})"
        )

    def roster_for_user(self, user_id: str) -> int:
        """
        This user's roster_id, resolved through their draft slot.

        Ownership is tracked per *roster*, not per slot: a traded pick keeps
        its slot (and therefore its position in the snake) but changes hands.
        """
        slot = self.slots_for_user(user_id)[0]
        try:
            return self.meta.slot_to_roster_id[slot]
        except KeyError:
            raise DraftShapeError(
                f"slot {slot} for user {user_id!r} is absent from "
                f"slot_to_roster_id {sorted(self.meta.slot_to_roster_id)}"
            ) from None

    def owner_of(self, round_: int, slot: int) -> int:
        """
        Which roster owns the pick at (round, slot), after trades.

        The slot fixes *when* the pick happens; traded_picks decides *whose*
        it is. Falls back to the slot's original roster when the pick was
        never traded.
        """
        original = self.meta.slot_to_roster_id.get(slot, slot)
        return self.traded.get((round_, original), original)

    def pick_numbers_for_roster(self, roster_id: int) -> list[int]:
        """Every overall pick number this roster owns, ascending, after trades."""
        snake = self._is_snake()
        nums = [
            pick_number(
                rnd, slot, self.meta.n_teams,
                snake=snake, reversal_round=self.meta.reversal_round,
            )
            for rnd in range(1, self.meta.rounds + 1)
            for slot in range(1, self.meta.n_teams + 1)
            if self.owner_of(rnd, slot) == roster_id
        ]
        return sorted(nums)

    def pick_numbers_for_user(self, user_id: str) -> list[int]:
        """Every overall pick number this user owns, ascending, after trades."""
        return self.pick_numbers_for_roster(self.roster_for_user(user_id))

    def my_next_pick(self, user_id: str) -> int | None:
        """
        Overall pick number of this user's next turn, or None if they have no
        picks left. A pick counts as gone once its number is <= picks_made.

        Traded picks are accounted for. This is not an edge case: in the
        recorded 2025 draft, 62 of 192 picks had changed hands, and computing
        this from draft_order alone put four of the user's own first eight
        picks in the wrong place (round 4 came out as 41 instead of 37). If
        traded_picks was not loaded, this silently degrades to original slot
        ownership — see from_client(include_traded=...).

        Auction drafts raise NotImplementedError rather than handing back
        something that looks like an answer.
        """
        made = self.picks_made
        for n in self.pick_numbers_for_user(user_id):
            if n > made:
                return n
        return None

    def picks_until(self, user_id: str) -> int | None:
        """How many picks happen before this user is up. 0 means on the clock."""
        nxt = self.my_next_pick(user_id)
        return None if nxt is None else nxt - self.picks_made - 1


# ---------------------------------------------------------------------------
# Player normalization
# ---------------------------------------------------------------------------

# Offensive fantasy positions. IDP positions are deliberately excluded by
# default: neither target league starts them, and including ~4000 defenders
# would drag them into the replacement-level solver for positions the league
# never scores.
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

# Verified against the live blob on 2026-08-05, and this is the one filter in
# the module that is easy to get quietly wrong:
#
#   * There is no "Retired" status. Retirement shows up as active=False
#     (Greg Olsen, Dwayne Haskins). Matching on a "Retired" string finds zero
#     players and silently filters nothing.
#   * status is None for all 32 DEF entries, so `status == "Active"` as a
#     filter deletes every defense in the league.
#   * A second stale cohort sits at active=True + status="Inactive" — Larry
#     Fitzgerald, Jarvis Landry, Mark Ingram. Long retired, but the active
#     flag alone does not catch them.
#
# Hence: require active, and drop the Inactive status specifically. Injured
# Reserve / PUP / NFI are kept — those players are rosterable and draftable.
DROP_STATUSES = {"Inactive"}


def player_name(raw: dict[str, Any]) -> str:
    """
    full_name is None for DEF entries, where the name lives split across
    first_name/last_name ("Houston" + "Texans"). Fall back through both before
    giving up on the id.
    """
    full = (raw.get("full_name") or "").strip()
    if full:
        return full
    parts = [(raw.get("first_name") or "").strip(), (raw.get("last_name") or "").strip()]
    joined = " ".join(p for p in parts if p)
    return joined or str(raw.get("player_id") or "UNKNOWN")


def normalize_players(
    blob: dict[str, Any],
    *,
    positions: Iterable[str] = FANTASY_POSITIONS,
    require_active: bool = True,
    drop_statuses: Iterable[str] = DROP_STATUSES,
) -> list[Player]:
    """
    Turn the raw /players/nfl blob into vorp.Player objects.

    proj is deliberately left at its 0.0 default — this layer has no
    projections and will not invent any. projections.py sets it later; keying
    on player_id is what makes that join possible.
    """
    wanted = set(positions)
    drop = set(drop_statuses)
    out: list[Player] = []

    for player_id, raw in blob.items():
        if not isinstance(raw, dict):
            continue
        pos = raw.get("position")
        if pos not in wanted:
            continue
        if require_active and not raw.get("active"):
            continue
        if raw.get("status") in drop:
            continue
        out.append(
            Player(
                name=player_name(raw),
                pos=pos,
                player_id=str(raw.get("player_id") or player_id),
                team=raw.get("team") or None,
                injury_status=raw.get("injury_status") or None,
            )
        )

    out.sort(key=lambda p: (p.pos, p.name))
    return out
