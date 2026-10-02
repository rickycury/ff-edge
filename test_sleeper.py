"""
Offline tests for the Sleeper data layer.

Nothing here touches the network. Fixtures in tests/fixtures/ were recorded
once by record_fixtures.py and scrubbed of league-mate identity; the client
under test is wired to a transport that reads them and raises on any endpoint
that was not recorded, so an accidental live call fails loudly.

    python3 -m unittest test_sleeper -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import sleeper
import vorp
from sleeper import (
    DiskCache,
    DraftMeta,
    DraftState,
    LeagueConfigError,
    OfflineError,
    RateLimiter,
    SleeperClient,
    build_league_config,
    normalize_players,
    pick_number,
)

FIXTURES = Path(__file__).parent / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def fixture_index() -> dict[str, str]:
    path = FIXTURES / "index.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_fixture(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        return json.load(fh)


INDEX = fixture_index()


def have(endpoint: str) -> bool:
    return endpoint in INDEX


def fixture_for(endpoint: str) -> Any:
    return load_fixture(INDEX[endpoint])


class FixtureTransport:
    """Stands in for HTTP. Unrecorded endpoint -> loud failure, never a request."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, endpoint: str) -> Any:
        self.calls.append(endpoint)
        name = INDEX.get(endpoint)
        if name is None:
            raise AssertionError(
                f"test requested unrecorded endpoint {endpoint!r}. "
                f"Recorded: {sorted(INDEX)}"
            )
        return load_fixture(name)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class StubClient:
    """Minimal stand-in for SleeperClient where only get_league is exercised."""

    def __init__(self, league: dict[str, Any]):
        self._league = league

    def get_league(self, league_id: str) -> dict[str, Any]:
        return self._league


class CacheTempDirMixin:
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ff-edge-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def client(self, **kw: Any) -> SleeperClient:
        kw.setdefault("transport", FixtureTransport())
        return SleeperClient(cache_dir=self.tmp, **kw)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter(unittest.TestCase):
    def test_stays_under_budget_and_blocks(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(max_calls=10, period=60.0, clock=clock, sleep=clock.sleep)
        for _ in range(10):
            self.assertEqual(rl.acquire(), 0.0)
        self.assertEqual(rl.in_window, 10)
        # 11th must block until the first call ages out.
        waited = rl.acquire()
        self.assertGreater(waited, 0.0)
        self.assertEqual(clock.t, 60.0)
        self.assertLessEqual(rl.in_window, 10)

    def test_never_exceeds_in_any_window(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(max_calls=5, period=60.0, clock=clock, sleep=clock.sleep)
        stamps = []
        for _ in range(40):
            rl.acquire()
            stamps.append(clock.t)
        # Brute force: no 60s window may contain more than 5 calls.
        for start in stamps:
            in_win = [s for s in stamps if start - 60.0 < s <= start]
            self.assertLessEqual(len(in_win), 5, f"window ending {start} had {len(in_win)}")

    def test_default_leaves_headroom_under_sleeper_ceiling(self) -> None:
        self.assertLess(sleeper.RATE_LIMIT_CALLS, 1000)

    def test_cache_hits_do_not_consume_budget(self) -> None:
        # The limiter must count network calls only; the cache is an
        # optimization and cannot be what keeps us under the ceiling.
        tmp = tempfile.mkdtemp(prefix="ff-edge-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        clock = FakeClock()
        rl = RateLimiter(max_calls=100, period=60.0, clock=clock, sleep=clock.sleep)
        c = SleeperClient(cache_dir=tmp, transport=FixtureTransport(), rate_limiter=rl)
        for _ in range(20):
            c.get_players()
        self.assertEqual(rl.total_calls, 1)
        self.assertEqual(c.network_calls, 1)
        self.assertEqual(c.cache_hits, 19)

    def test_rejects_nonsense_config(self) -> None:
        with self.assertRaises(ValueError):
            RateLimiter(max_calls=0)
        with self.assertRaises(ValueError):
            RateLimiter(period=0)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class TestDiskCache(CacheTempDirMixin, unittest.TestCase):
    def test_roundtrip_and_expiry(self) -> None:
        cache = DiskCache(self.tmp)
        cache.put("league/123", {"a": 1}, now=1000.0)
        self.assertEqual(cache.get("league/123", ttl=600, now=1300.0), {"a": 1})
        self.assertIs(cache.get("league/123", ttl=600, now=2000.0), sleeper._MISS)

    def test_zero_ttl_always_misses(self) -> None:
        cache = DiskCache(self.tmp)
        cache.put("draft/1/picks", [1, 2], now=1000.0)
        self.assertIs(cache.get("draft/1/picks", ttl=0, now=1000.0), sleeper._MISS)

    def test_distinct_endpoints_never_collide(self) -> None:
        cache = DiskCache(self.tmp)
        eps = [
            "league/123", "league/123/users", "league/123/rosters",
            "draft/9/picks", "draft/9", "players/nfl",
            "user/1/leagues/nfl/2025", "user/1/leagues/nfl/2026",
        ]
        keys = {cache.key_for(e) for e in eps}
        self.assertEqual(len(keys), len(eps))

    def test_corrupt_entry_is_a_miss_not_a_crash(self) -> None:
        cache = DiskCache(self.tmp)
        path = cache.path_for("league/123")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json", encoding="utf-8")
        self.assertIs(cache.get("league/123", ttl=600), sleeper._MISS)

    def test_ttls_match_the_documented_policy(self) -> None:
        self.assertEqual(sleeper.TTL_PLAYERS, 24 * 3600)
        self.assertEqual(sleeper.TTL_LEAGUE, 600)
        self.assertEqual(sleeper.TTL_DRAFT_PICKS, 3.0)


class TestClientCaching(CacheTempDirMixin, unittest.TestCase):
    def test_picks_ttl_is_caller_supplied(self) -> None:
        c = self.client()
        if not have("players/nfl"):
            self.skipTest("no players fixture")
        c.get_players()
        c.get_players()
        self.assertEqual(c.network_calls, 1)

    def test_offline_client_raises_on_miss(self) -> None:
        c = SleeperClient(cache_dir=self.tmp, offline=True)
        with self.assertRaises(OfflineError):
            c.get_league("123")

    def test_offline_client_serves_from_cache(self) -> None:
        DiskCache(self.tmp).put("league/123", {"total_rosters": 12})
        c = SleeperClient(cache_dir=self.tmp, offline=True)
        self.assertEqual(c.get_league("123"), {"total_rosters": 12})


# ---------------------------------------------------------------------------
# League config
# ---------------------------------------------------------------------------


class TestBuildLeagueConfig(unittest.TestCase):
    BASE = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * 6

    def build(self, roster_positions: list[str], total_rosters: int = 12) -> vorp.LeagueConfig:
        return build_league_config(
            StubClient({"roster_positions": roster_positions,
                        "total_rosters": total_rosters}),
            "L1",
        )

    def test_parses_straight_from_the_endpoint(self) -> None:
        cfg = self.build(self.BASE)
        self.assertEqual(cfg.n_teams, 12)
        self.assertEqual(cfg.dedicated, {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1})
        self.assertEqual(cfg.flex_slots, {"FLEX": 1})

    def test_superflex_and_multiflex(self) -> None:
        cfg = self.build(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX",
                          "SUPER_FLEX", "BN", "BN"])
        self.assertEqual(cfg.flex_slots, {"FLEX": 2, "SUPER_FLEX": 1})
        self.assertIn("QB", cfg.scored_positions)

    def test_unknown_flex_type_fails_loudly(self) -> None:
        # The exact silent-corruption case: LeagueConfig would file IDP_FLEX
        # under `dedicated` and quietly treat it as a one-off position.
        with self.assertRaises(LeagueConfigError) as ctx:
            self.build(self.BASE + ["IDP_FLEX"])
        self.assertIn("IDP_FLEX", str(ctx.exception))
        self.assertIn("FLEX_ELIGIBILITY", str(ctx.exception))

    def test_unknown_dedicated_slot_fails_loudly(self) -> None:
        with self.assertRaises(LeagueConfigError) as ctx:
            self.build(self.BASE + ["WOMBAT"])
        self.assertIn("WOMBAT", str(ctx.exception))

    def test_slots_round_trip_exactly(self) -> None:
        for positions in (self.BASE,
                          ["QB", "SUPER_FLEX", "BN", "IR", "TAXI"],
                          ["QB"] * 3 + ["FLEX"] * 2 + ["BN"] * 10):
            cfg = self.build(positions)
            rebuilt: dict[str, int] = {}
            for bucket in (cfg.dedicated, cfg.flex_slots):
                for slot, n in bucket.items():
                    rebuilt[slot] = rebuilt.get(slot, 0) + n
            ignored = sum(1 for s in positions if s in vorp.IGNORED_SLOTS)
            self.assertEqual(sum(rebuilt.values()) + ignored, len(positions))

    def test_bad_total_rosters_rejected(self) -> None:
        for bad in (None, 0, -3, "12", 12.0):
            with self.assertRaises(LeagueConfigError):
                self.build(self.BASE, total_rosters=bad)  # type: ignore[arg-type]

    def test_missing_or_malformed_roster_positions_rejected(self) -> None:
        for bad in (None, [], "QB,RB", [1, 2, 3]):
            with self.assertRaises(LeagueConfigError):
                build_league_config(
                    StubClient({"roster_positions": bad, "total_rosters": 12}), "L1"
                )

    def test_all_bench_league_rejected(self) -> None:
        with self.assertRaises(LeagueConfigError):
            self.build(["BN"] * 10)

    @unittest.skipUnless(INDEX, "no fixtures recorded")
    def test_against_recorded_league(self) -> None:
        eps = [e for e in INDEX if e.startswith("league/") and e.count("/") == 1]
        if not eps:
            self.skipTest("no league fixture recorded (needs --username run)")
        for ep in eps:
            payload = fixture_for(ep)
            cfg = build_league_config(StubClient(payload), ep.split("/")[1])
            self.assertEqual(cfg.n_teams, payload["total_rosters"])
            self.assertGreater(len(cfg.dedicated) + len(cfg.flex_slots), 0)


# ---------------------------------------------------------------------------
# Pick ordering
# ---------------------------------------------------------------------------


class TestPickNumber(unittest.TestCase):
    N = 12

    def order(self, rnd: int, **kw: Any) -> list[int]:
        return [pick_number(rnd, s, self.N, **kw) for s in range(1, self.N + 1)]

    def test_snake_alternates(self) -> None:
        self.assertEqual(self.order(1, snake=True), list(range(1, 13)))
        self.assertEqual(self.order(2, snake=True), list(range(24, 12, -1)))
        self.assertEqual(self.order(3, snake=True), list(range(25, 37)))

    def test_linear_repeats(self) -> None:
        self.assertEqual(self.order(1, snake=False), list(range(1, 13)))
        self.assertEqual(self.order(2, snake=False), list(range(13, 25)))

    def test_third_round_reversal(self) -> None:
        kw = {"snake": True, "reversal_round": 3}
        self.assertEqual(self.order(1, **kw), list(range(1, 13)))
        self.assertEqual(self.order(2, **kw), list(range(24, 12, -1)))
        # The reversal: round 3 repeats round 2's direction instead of flipping.
        self.assertEqual(self.order(3, **kw), list(range(36, 24, -1)))
        self.assertEqual(self.order(4, **kw), list(range(37, 49)))

    def test_reversal_zero_is_plain_snake(self) -> None:
        for rnd in range(1, 8):
            self.assertEqual(self.order(rnd, snake=True, reversal_round=0),
                             self.order(rnd, snake=True))

    def test_every_pick_hit_exactly_once(self) -> None:
        rounds = 15
        for kw in ({"snake": True}, {"snake": False},
                   {"snake": True, "reversal_round": 3},
                   {"snake": True, "reversal_round": 2}):
            got = sorted(pick_number(r, s, self.N, **kw)
                         for r in range(1, rounds + 1)
                         for s in range(1, self.N + 1))
            self.assertEqual(got, list(range(1, self.N * rounds + 1)), kw)

    def test_rejects_out_of_range(self) -> None:
        for args in ((1, 0, 12), (1, 13, 12), (0, 1, 12)):
            with self.assertRaises(ValueError):
                pick_number(*args, snake=True)


# ---------------------------------------------------------------------------
# Draft state
# ---------------------------------------------------------------------------


def synthetic_meta(**kw: Any) -> DraftMeta:
    base = dict(
        draft_id="d1", type="snake", status="complete", n_teams=12, rounds=15,
        reversal_round=0,
        draft_order={"me": 4, "other": 9},
        slot_to_roster_id={s: s for s in range(1, 13)},
        league_id="L1",
        season=None,
    )
    base.update(kw)
    return DraftMeta(**base)  # type: ignore[arg-type]


class TestDraftStateSynthetic(unittest.TestCase):
    def state(self, made: int, **kw: Any) -> DraftState:
        return DraftState(synthetic_meta(**kw),
                          [{"pick_no": i} for i in range(1, made + 1)])

    def test_next_pick_progression(self) -> None:
        for made, expected in [(0, 4), (3, 4), (4, 21), (20, 21), (21, 28), (27, 28)]:
            self.assertEqual(self.state(made).my_next_pick("me"), expected, f"made={made}")

    def test_exhausted_returns_none(self) -> None:
        self.assertIsNone(self.state(180).my_next_pick("me"))

    def test_picks_until(self) -> None:
        self.assertEqual(self.state(3).picks_until("me"), 0)   # on the clock
        self.assertEqual(self.state(4).picks_until("me"), 16)

    def test_picks_made_uses_max_not_len(self) -> None:
        # Sleeper can return picks out of order or with gaps; a length count
        # would understate how far along the draft is.
        st = DraftState(synthetic_meta(), [{"pick_no": 9}, {"pick_no": 3}])
        self.assertEqual(st.picks_made, 9)
        self.assertEqual(st.my_next_pick("me"), 21)

    def test_auction_raises_not_implemented(self) -> None:
        st = self.state(0, type="auction")
        with self.assertRaises(NotImplementedError) as ctx:
            st.my_next_pick("me")
        self.assertIn("auction", str(ctx.exception).lower())

    def test_unknown_type_raises(self) -> None:
        with self.assertRaises(sleeper.DraftShapeError):
            self.state(0, type="bogus").my_next_pick("me")

    def test_linear_draft(self) -> None:
        self.assertEqual(self.state(0, type="linear").my_next_pick("me"), 4)
        self.assertEqual(self.state(4, type="linear").my_next_pick("me"), 16)

    def test_reversal_round_changes_the_answer(self) -> None:
        # Guard against reversal_round being quietly dropped: slot 4's third
        # pick differs between plain snake and 3RR.
        plain = self.state(24).my_next_pick("me")
        rr = self.state(24, reversal_round=3).my_next_pick("me")
        self.assertEqual(plain, 28)
        self.assertEqual(rr, 33)
        self.assertNotEqual(plain, rr)

    def test_each_user_resolves_to_their_own_slot(self) -> None:
        st = self.state(0, draft_order={"me": 4, "other": 9})
        self.assertEqual(st.slots_for_user("me"), [4])
        self.assertEqual(st.slots_for_user("other"), [9])
        self.assertEqual(st.my_next_pick("me"), 4)
        self.assertEqual(st.my_next_pick("other"), 9)

    def test_user_id_accepted_as_int_or_str(self) -> None:
        st = self.state(0, draft_order={"123456": 4})
        self.assertEqual(st.my_next_pick(123456), 4)  # type: ignore[arg-type]
        self.assertEqual(st.my_next_pick("123456"), 4)

    def test_unknown_user_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            self.state(0).my_next_pick("nobody")

    def test_slot_without_roster_raises(self) -> None:
        st = self.state(0, draft_order={"me": 99}, slot_to_roster_id={s: s for s in range(1, 13)})
        with self.assertRaises(sleeper.DraftShapeError):
            st.my_next_pick("me")

    def test_teams_disagreement_rejected(self) -> None:
        with self.assertRaises(sleeper.DraftShapeError):
            DraftMeta.from_payload({
                "draft_id": "d", "type": "snake",
                "settings": {"teams": 12, "rounds": 15},
                "slot_to_roster_id": {"1": 1, "2": 2},
            })

    def test_traded_pick_changes_ownership(self) -> None:
        # Slot 4 = roster 4. Trade roster 9's round-2 pick to roster 4.
        traded = [{"season": "2025", "round": 2, "roster_id": 9, "owner_id": 4}]
        meta = synthetic_meta(season="2025")
        st = DraftState(meta, [], traded)
        self.assertEqual(st.owner_of(2, 9), 4)      # changed hands
        self.assertEqual(st.owner_of(2, 4), 4)      # untouched
        self.assertEqual(st.owner_of(1, 9), 9)      # different round
        # Slot 9 in round 2 is overall pick 16, which now falls to roster 4
        # ahead of its own round-2 pick at 21.
        self.assertIn(16, st.pick_numbers_for_user("me"))
        self.assertEqual(DraftState(meta, [{"pick_no": 8}], traded).my_next_pick("me"), 16)

    def test_pick_traded_away_is_not_mine(self) -> None:
        traded = [{"season": "2025", "round": 1, "roster_id": 4, "owner_id": 9}]
        st = DraftState(synthetic_meta(season="2025"), [], traded)
        self.assertNotIn(4, st.pick_numbers_for_user("me"))
        self.assertEqual(st.my_next_pick("me"), 21)

    def test_no_traded_picks_is_original_ownership(self) -> None:
        st = DraftState(synthetic_meta(season="2025"), [], [])
        self.assertEqual(st.my_next_pick("me"), 4)

    def test_traded_map_ignores_other_seasons_and_junk(self) -> None:
        rows = [
            {"season": "2026", "round": 1, "roster_id": 4, "owner_id": 9},
            {"season": "2025", "round": None, "roster_id": 4, "owner_id": 9},
            {"season": "2025", "round": 1, "roster_id": None, "owner_id": 9},
        ]
        self.assertEqual(sleeper.build_traded_map(rows, "2025"), {})

    def test_ownership_partitions_the_draft(self) -> None:
        traded = [{"season": "2025", "round": 2, "roster_id": 9, "owner_id": 4},
                  {"season": "2025", "round": 5, "roster_id": 1, "owner_id": 12}]
        meta = synthetic_meta(season="2025")
        st = DraftState(meta, [], traded)
        seen: list[int] = []
        for r in set(meta.slot_to_roster_id.values()):
            seen.extend(st.pick_numbers_for_roster(r))
        self.assertEqual(sorted(seen), list(range(1, meta.n_teams * meta.rounds + 1)))

    def test_drafted_ids_and_available(self) -> None:
        st = DraftState(synthetic_meta(), [
            {"pick_no": 1, "player_id": "4046"},
            {"pick_no": 2, "player_id": None},
        ])
        self.assertEqual(st.drafted_player_ids, {"4046"})
        pool = [vorp.Player(name="A", pos="QB", player_id="4046"),
                vorp.Player(name="B", pos="RB", player_id="9999")]
        self.assertEqual([p.name for p in st.available(pool)], ["B"])


@unittest.skipUnless(have("_ground_truth"), "draft fixtures not recorded yet")
class TestDraftStateAgainstRealDraft(unittest.TestCase):
    """
    The load-bearing test: ordering checked against a real completed draft,
    not against its own arithmetic.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = fixture_for("_ground_truth")
        cls.draft = fixture_for(f"draft/{cls.truth['draft_id']}")
        cls.picks = fixture_for(f"draft/{cls.truth['draft_id']}/picks")
        cls.traded = fixture_for(f"league/{cls.truth['previous_league_id']}/traded_picks")
        cls.meta = DraftMeta.from_payload(cls.draft)
        cls.state = DraftState(cls.meta, cls.picks, cls.traded)

    def test_ownership_reproduces_every_recorded_pick(self) -> None:
        # For all 192 picks: (slot_to_roster_id + traded_picks) must predict
        # the roster that actually made the pick.
        for p in self.picks:
            self.assertEqual(
                self.state.owner_of(p["round"], p["draft_slot"]),
                p["roster_id"],
                f"pick {p['pick_no']} (round {p['round']}, slot {p['draft_slot']})",
            )

    def test_traded_picks_are_not_a_rounding_error(self) -> None:
        # Guard against someone deciding trades are negligible and dropping
        # the extra request: without them, this user's own board is wrong.
        naive = DraftState(self.meta, self.picks)  # no traded picks
        self.assertNotEqual(
            naive.pick_numbers_for_user(self.truth["my_user_id"]),
            self.state.pick_numbers_for_user(self.truth["my_user_id"]),
        )
        self.assertGreater(len(self.state.traded), 0)

    def test_season_filter_excludes_other_years(self) -> None:
        # A dynasty league trades picks years out; applying a 2027 trade to
        # the 2025 draft would hand picks to the wrong roster.
        polluted = list(self.traded) + [
            {"season": "2027", "round": 1, "roster_id": 1, "owner_id": 99}
        ]
        st = DraftState(self.meta, self.picks, polluted)
        self.assertEqual(st.traded, self.state.traded)

    def test_formula_reproduces_every_recorded_pick(self) -> None:
        # ~180 real (round, draft_slot) -> pick_no triples. If the snake or
        # reversal handling is wrong anywhere, this fails.
        meta = self.state.meta
        snake = meta.type == "snake"
        checked = 0
        for p in self.picks:
            if p.get("round") is None or p.get("draft_slot") is None:
                continue
            self.assertEqual(
                pick_number(p["round"], p["draft_slot"], meta.n_teams,
                            snake=snake, reversal_round=meta.reversal_round),
                p["pick_no"],
                f"round {p['round']} slot {p['draft_slot']}",
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_my_pick_numbers_match_ground_truth(self) -> None:
        # The load-bearing assertion: every pick number this user actually
        # held in a real completed draft, derived from draft structure alone.
        expected = self.truth["my_actual_pick_numbers"]
        if not expected:
            self.skipTest("recorded user made no picks in this draft")
        got = self.state.pick_numbers_for_user(self.truth["my_user_id"])
        self.assertEqual(got, expected)

    def test_roster_resolution(self) -> None:
        self.assertEqual(
            self.state.roster_for_user(self.truth["my_user_id"]),
            self.truth["my_roster_id"],
        )

    def test_every_pick_owned_by_exactly_one_roster(self) -> None:
        # Ownership must partition the draft: no pick unowned, none double-owned.
        meta = self.state.meta
        rosters = set(meta.slot_to_roster_id.values())
        seen: list[int] = []
        for r in rosters:
            seen.extend(self.state.pick_numbers_for_roster(r))
        self.assertEqual(sorted(seen), list(range(1, meta.n_teams * meta.rounds + 1)))

    def test_my_next_pick_walks_the_real_draft(self) -> None:
        mine = self.truth["my_actual_pick_numbers"]
        if not mine:
            self.skipTest("recorded user made no picks in this draft")
        meta = self.state.meta
        total = meta.n_teams * meta.rounds
        for made in range(0, total + 1):
            st = DraftState(meta, [{"pick_no": i} for i in range(1, made + 1)],
                            self.traded)
            expected = next((n for n in mine if n > made), None)
            self.assertEqual(st.my_next_pick(self.truth["my_user_id"]), expected,
                             f"after {made} picks")

    def test_fixtures_carry_no_real_identities(self) -> None:
        # Scrubbed ids are synthetic and uniform in shape; display names are
        # generated. This is a regression guard on record_fixtures.py.
        for uid in self.state.meta.draft_order:
            self.assertRegex(uid, r"^\d{18}$")


@unittest.skipUnless(have("_ground_truth"), "draft fixtures not recorded yet")
class TestScrubbedUsers(unittest.TestCase):
    def test_users_fixture_is_synthetic(self) -> None:
        truth = fixture_for("_ground_truth")
        users = fixture_for(f"league/{truth['previous_league_id']}/users")
        self.assertTrue(users)
        for u in users:
            self.assertRegex(str(u["user_id"]), r"^\d{18}$")
            self.assertRegex(u["display_name"], r"^manager_\d{2}$")
            if u.get("avatar") is not None:
                self.assertRegex(u["avatar"], r"^[0-9a-f]{32}$")

    def test_structure_preserved(self) -> None:
        truth = fixture_for("_ground_truth")
        users = fixture_for(f"league/{truth['previous_league_id']}/users")
        for u in users:
            self.assertIn("user_id", u)
            self.assertIn("display_name", u)
            self.assertIsInstance(u["user_id"], str)

    def test_no_user_authored_free_text_in_rosters(self) -> None:
        # roster.metadata carries p_nick_<player_id> keys holding nicknames
        # typed by leaguemates. These are arbitrary free text and have no
        # business in a git-tracked fixture; the recorder replaces every value
        # outside its allowlist with a synthetic placeholder.
        import record_fixtures

        truth = fixture_for("_ground_truth")
        rosters = fixture_for(f"league/{truth['previous_league_id']}/rosters")
        checked = 0
        for r in rosters:
            for key, value in (r.get("metadata") or {}).items():
                if key in record_fixtures.Scrubber.SAFE_ROSTER_META:
                    continue
                if isinstance(value, str):
                    self.assertRegex(value, r"^text_\d+$",
                                     f"unscrubbed free text in metadata[{key!r}]")
                    checked += 1
        self.assertGreater(checked, 0, "fixture should contain metadata to scrub")

    def test_index_does_not_key_on_a_real_username(self) -> None:
        # The /user/{username} endpoint path *is* the username, so the index
        # key leaks it even when the payload is scrubbed.
        for endpoint in INDEX:
            if endpoint.startswith("user/"):
                self.assertRegex(endpoint, r"^user/manager_\d{2}$")


# ---------------------------------------------------------------------------
# Player normalization
# ---------------------------------------------------------------------------


@unittest.skipUnless(have("players/nfl"), "no players fixture")
class TestNormalizePlayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blob = fixture_for("players/nfl")
        cls.players = normalize_players(cls.blob)
        cls.by_id = {p.player_id: p for p in cls.players}

    def test_returns_vorp_players(self) -> None:
        self.assertTrue(self.players)
        for p in self.players:
            self.assertIsInstance(p, vorp.Player)

    def test_only_fantasy_positions(self) -> None:
        self.assertTrue({p.pos for p in self.players} <= sleeper.FANTASY_POSITIONS)

    def test_non_fantasy_positions_dropped(self) -> None:
        raw_positions = {p.get("position") for p in self.blob.values()}
        self.assertTrue(raw_positions - sleeper.FANTASY_POSITIONS,
                        "fixture should contain a non-fantasy player to drop")
        for p in self.players:
            self.assertIn(p.pos, sleeper.FANTASY_POSITIONS)

    def test_retired_players_dropped(self) -> None:
        # Retirement shows up as active=False, NOT as a "Retired" status.
        retired = [pid for pid, r in self.blob.items() if r.get("active") is False]
        self.assertTrue(retired, "fixture should contain an active=False player")
        for pid in retired:
            self.assertNotIn(pid, self.by_id)

    def test_stale_active_inactive_dropped(self) -> None:
        # The cohort the active flag alone misses.
        stale = [pid for pid, r in self.blob.items()
                 if r.get("active") is True and r.get("status") == "Inactive"]
        self.assertTrue(stale, "fixture should contain an active+Inactive player")
        for pid in stale:
            self.assertNotIn(pid, self.by_id)

    def test_defenses_survive(self) -> None:
        # status is None for DEF; a `status == "Active"` filter would delete
        # every defense in the league.
        defs = [p for p in self.players if p.pos == "DEF"]
        self.assertTrue(defs, "defenses must survive normalization")
        for d in defs:
            self.assertTrue(d.name.strip())
            self.assertNotEqual(d.name, d.player_id)

    def test_injured_reserve_kept(self) -> None:
        ir = [pid for pid, r in self.blob.items()
              if r.get("active") is True and r.get("status") == "Injured Reserve"
              and r.get("position") in sleeper.FANTASY_POSITIONS]
        for pid in ir:
            self.assertIn(pid, self.by_id, "IR players are still draftable")

    def test_names_never_empty(self) -> None:
        for p in self.players:
            self.assertTrue(p.name and p.name.strip())

    def test_fields_carried(self) -> None:
        for p in self.players:
            self.assertIsInstance(p.player_id, str)
            raw = self.blob[p.player_id]
            self.assertEqual(p.team, raw.get("team") or None)
            self.assertEqual(p.injury_status, raw.get("injury_status") or None)

    def test_injury_status_actually_present_somewhere(self) -> None:
        self.assertTrue([p for p in self.players if p.injury_status],
                        "fixture should include a player carrying injury_status")

    def test_no_projections_invented(self) -> None:
        for p in self.players:
            self.assertEqual(p.proj, 0.0)
            self.assertIsNone(p.adp)

    def test_free_agents_kept_with_null_team(self) -> None:
        fas = [p for p in self.players if p.team is None]
        self.assertTrue(fas, "unsigned players should survive with team=None")

    def test_ids_unique(self) -> None:
        ids = [p.player_id for p in self.players]
        self.assertEqual(len(ids), len(set(ids)))

    def test_feeds_the_engine_unchanged(self) -> None:
        # The whole point of returning vorp.Player: the math layer takes these
        # as-is. Projections are absent, so this only checks it runs clean.
        league = vorp.LeagueConfig(
            n_teams=12,
            roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                              "K", "DEF"] + ["BN"] * 6,
        )
        vorp.assign_vorp(list(self.players), league)


if __name__ == "__main__":
    unittest.main(verbosity=2)
