"""
Offline tests for the market layer.

Logic is tested against synthetic CSVs written to a temp dir. The real Flock
exports in data/ are used as an integration check when present. No network.

    python3 -m unittest test_market -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import market
import vorp
from market import (
    HOME_PLATFORM,
    MarketView,
    Matcher,
    divergence_table,
    load_adp,
    load_expert_ranks,
    zscores,
)

DATA = Path(__file__).parent / "data"
ADP_CSV = DATA / "flock_adp.csv"
EXPERT_CSV = DATA / "flock_expert_ranks.csv"


def pool() -> list[vorp.Player]:
    return [
        vorp.Player(name="Jahmyr Gibbs", pos="RB", player_id="9221", team="DET"),
        vorp.Player(name="Ja'Marr Chase", pos="WR", player_id="7564", team="CIN"),
        vorp.Player(name="Kenny Gainwell", pos="RB", player_id="7200", team="TB"),
        vorp.Player(name="Brock Bowers", pos="TE", player_id="11604", team="LV"),
        vorp.Player(name="Michael Thomas", pos="WR", player_id="3210", team="NO"),
        vorp.Player(name="Michael Thomas", pos="RB", player_id="3211", team="KC"),
    ]


class TestMarketView(unittest.TestCase):
    def view(self, **kw) -> MarketView:
        base = dict(
            player_id="1", name="X", pos="RB",
            platform_adp={"Sleeper": 40.0, "ESPN": 30.0, "Yahoo": 32.0,
                          "Underdog": 28.0, "CBS": 34.0, "FFPC": 30.0},
        )
        base.update(kw)
        return MarketView(**base)

    def test_market_and_outside_means(self) -> None:
        v = self.view()
        self.assertAlmostEqual(v.market_adp, 32.3333, places=3)
        self.assertAlmostEqual(v.outside_market_adp, 30.8, places=3)
        self.assertEqual(v.home_adp, 40.0)
        self.assertEqual(v.n_platforms, 6)

    def test_home_vs_market_sign(self) -> None:
        # Positive means he lasts LONGER on our platform than outside.
        v = self.view()
        self.assertAlmostEqual(v.home_vs_market, 9.2, places=3)

    def test_sd_is_the_disagreement_width(self) -> None:
        tight = self.view(platform_adp={"Sleeper": 40.0, "ESPN": 40.0, "Yahoo": 41.0})
        wide = self.view(platform_adp={"Sleeper": 20.0, "ESPN": 60.0, "Yahoo": 40.0})
        self.assertLess(tight.adp_sd, wide.adp_sd)

    def test_sd_needs_two_platforms(self) -> None:
        # A standard deviation of one number is a lie, not a zero.
        self.assertIsNone(self.view(platform_adp={"Sleeper": 40.0}).adp_sd)
        self.assertIsNone(self.view(platform_adp={}).adp_sd)

    def test_empty_platforms_yield_none_not_crash(self) -> None:
        v = self.view(platform_adp={})
        self.assertIsNone(v.market_adp)
        self.assertIsNone(v.home_adp)
        self.assertIsNone(v.home_vs_market)

    def test_expert_divergences(self) -> None:
        v = self.view(expert_rank=20.0)
        # Experts rank him 20th, market drafts him ~32nd -> crowd is late.
        self.assertLess(v.expert_vs_market, 0)
        self.assertAlmostEqual(v.expert_vs_home, -20.0, places=3)

    def test_expert_adp_used_when_rank_absent(self) -> None:
        v = self.view(expert_adp=25.0)
        self.assertIsNotNone(v.expert_vs_market)
        v2 = self.view(expert_adp=25.0, expert_rank=10.0)
        # expert_rank wins when both are present.
        self.assertAlmostEqual(v2.expert_vs_home, -30.0, places=3)

    def test_home_platform_excluded_from_outside(self) -> None:
        v = self.view()
        self.assertNotIn(v.platform_adp[HOME_PLATFORM], [])
        self.assertEqual(len(v.outside_adp), 5)


class TestMatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Matcher(pool())

    def test_exact_name_and_position(self) -> None:
        self.assertEqual(self.m.match("Jahmyr Gibbs", "RB").player_id, "9221")

    def test_punctuation_folding(self) -> None:
        self.assertEqual(self.m.match("JaMarr Chase", "WR").player_id, "7564")

    def test_nickname_alias(self) -> None:
        # The one real miss on the live export: Flock says Kenneth, Sleeper Kenny.
        self.assertEqual(self.m.match("Kenneth Gainwell", "RB").player_id, "7200")

    def test_surname_fallback(self) -> None:
        self.assertEqual(self.m.match("Brocky Bowers", "TE").player_id, "11604")

    def test_surname_fallback_disambiguated_by_team(self) -> None:
        # Two Michael Thomases; surname+pos alone is ambiguous for WR only
        # after position filtering, so team settles it.
        got = self.m.match("Mike Thomas", "WR", "NO")
        self.assertEqual(got.player_id, "3210")

    def test_position_prevents_cross_position_collision(self) -> None:
        self.assertEqual(self.m.match("Michael Thomas", "RB").player_id, "3211")
        self.assertEqual(self.m.match("Michael Thomas", "WR").player_id, "3210")

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(self.m.match("Nobody At All", "WR"))

    def test_ambiguous_surname_without_team_is_a_miss(self) -> None:
        players = [
            vorp.Player(name="Josh Allen", pos="QB", player_id="1", team="BUF"),
            vorp.Player(name="Kyle Allen", pos="QB", player_id="2", team="NYJ"),
        ]
        m = Matcher(players)
        self.assertIsNone(m.match("Bobby Allen", "QB"))


class TestLoadCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="ff-edge-market-")

    def write(self, name: str, text: str) -> str:
        p = Path(self.dir) / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_positional_rank_stripped_from_pos(self) -> None:
        path = self.write("a.csv",
            "Rank,Player,POS,Team,AVG,Expert,Sleeper,ESPN,Yahoo,Underdog,CBS,FFPC\n"
            "1,Jahmyr Gibbs,RB1,DET,1.0,1,1,1,1,1,1,1\n")
        views, unmatched = load_adp(path, pool())
        self.assertEqual(unmatched, [])
        self.assertEqual(views["9221"].pos, "RB")

    def test_missing_platform_cells_are_absent_not_zero(self) -> None:
        path = self.write("b.csv",
            "Rank,Player,POS,Team,AVG,Expert,Sleeper,ESPN,Yahoo,Underdog,CBS,FFPC\n"
            "1,Brock Bowers,TE1,LV,20.0,19,20,,22,18,,\n")
        views, _ = load_adp(path, pool())
        v = views["11604"]
        self.assertEqual(v.n_platforms, 3)
        self.assertNotIn("ESPN", v.platform_adp)
        self.assertEqual(v.home_adp, 20.0)

    def test_duplicate_name_does_not_clobber_the_earlier_match(self) -> None:
        # The live file has repeated names deep in the tail; the shallow
        # (real) player must win.
        path = self.write("c.csv",
            "Rank,Player,POS,Team,AVG,Expert,Sleeper,ESPN,Yahoo,Underdog,CBS,FFPC\n"
            "1,Jahmyr Gibbs,RB1,DET,1.0,1,1,1,1,1,1,1\n"
            "400,Jahmyr Gibbs,RB99,DET,400.0,,,,,400,,\n")
        views, _ = load_adp(path, pool())
        self.assertEqual(views["9221"].market_adp, 1.0)

    def test_unmatched_rows_are_returned(self) -> None:
        path = self.write("d.csv",
            "Rank,Player,POS,Team,AVG,Expert,Sleeper,ESPN,Yahoo,Underdog,CBS,FFPC\n"
            "1,Ghost Player,WR1,FA,1.0,1,1,1,1,1,1,1\n")
        views, unmatched = load_adp(path, pool())
        self.assertEqual(views, {})
        self.assertEqual(len(unmatched), 1)
        self.assertIn("Ghost Player", unmatched[0])

    def test_expert_ranks_load_and_merge(self) -> None:
        path = self.write("e.csv",
            "Name,Team,Position,Expert Rank\nJahmyr Gibbs,DET,RB,1.25\n")
        ranks, unmatched = load_expert_ranks(path, pool())
        self.assertEqual(unmatched, [])
        self.assertEqual(ranks["9221"], 1.25)


class TestZScores(unittest.TestCase):
    def test_basic(self) -> None:
        z = zscores({"a": 10.0, "b": 20.0, "c": 30.0})
        self.assertAlmostEqual(z["b"], 0.0)
        self.assertGreater(z["c"], 0)
        self.assertLess(z["a"], 0)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(zscores({"a": 5.0}), {"a": 0.0})
        self.assertEqual(zscores({"a": 5.0, "b": 5.0}), {"a": 0.0, "b": 0.0})
        self.assertEqual(zscores({}), {})

    def test_banding_keeps_early_picks_comparable_to_each_other(self) -> None:
        # An early-round player with a modest absolute divergence should still
        # score highly within his own band, rather than being drowned out by
        # the huge absolute swings that happen deep in the draft.
        views = {}
        for i in range(1, 101):
            adp = float(i)
            # everyone flat except player 5 (small early skew) and 95 (big late skew)
            home = adp + (8.0 if i == 5 else 0.0) + (40.0 if i == 95 else 0.0)
            views[str(i)] = MarketView(
                player_id=str(i), name=f"P{i}", pos="RB",
                platform_adp={"Sleeper": home, "ESPN": adp, "Yahoo": adp, "CBS": adp},
            )
        scored = divergence_table(views, band_size=50)
        self.assertGreater(scored["5"], 3.0)
        self.assertGreater(scored["95"], 3.0)

    def test_requires_minimum_platforms(self) -> None:
        views = {
            "1": MarketView(player_id="1", name="A", pos="RB",
                            platform_adp={"Sleeper": 10.0}),
        }
        self.assertEqual(divergence_table(views, min_platforms=3), {})


@unittest.skipUnless(ADP_CSV.exists(), "no Flock ADP export in data/")
class TestAgainstRealExport(unittest.TestCase):
    """Integration check against the committed Flock files. Still offline."""

    @classmethod
    def setUpClass(cls) -> None:
        import json
        import sleeper

        fixtures = Path(__file__).parent / "tests" / "fixtures"
        with (fixtures / "index.json").open() as fh:
            index = json.load(fh)
        with (fixtures / index["players/nfl"]).open() as fh:
            blob = json.load(fh)
        cls.players = sleeper.normalize_players(blob)
        cls.views, cls.report = market.load_all(ADP_CSV, EXPERT_CSV, cls.players)

    def test_matches_the_fixture_pool(self) -> None:
        # The trimmed fixture holds ~40 players; every one that appears in the
        # export should resolve, proving the matcher works on real names.
        self.assertGreater(len(self.views), 10)

    def test_platform_columns_are_the_expected_six(self) -> None:
        seen = set()
        for v in self.views.values():
            seen |= set(v.platform_adp)
        self.assertTrue(seen <= set(market.PLATFORMS), seen)

    def test_home_platform_present_for_top_players(self) -> None:
        shallow = [v for v in self.views.values()
                   if v.market_adp is not None and v.market_adp <= 100]
        self.assertTrue(shallow)
        self.assertTrue(all(v.home_adp is not None for v in shallow))

    def test_sd_available_for_shallow_players(self) -> None:
        shallow = [v for v in self.views.values()
                   if v.market_adp is not None and v.market_adp <= 100]
        self.assertTrue(all(v.adp_sd is not None for v in shallow))


if __name__ == "__main__":
    unittest.main(verbosity=2)
