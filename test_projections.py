"""
Offline tests for the projection layer.

Same contract as test_sleeper.py: fixtures only, no network, unrecorded
endpoints raise instead of falling through to a request.

    python3 -m unittest test_projections -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import projections
import sleeper
import vorp
from projections import (
    AttachReport,
    ProjectedLine,
    attach,
    build_name_index,
    from_csv,
    from_sleeper,
    normalize_name,
    scoring_kind,
)

FIXTURES = Path(__file__).parent / "tests" / "fixtures"


def fixture_index() -> dict[str, str]:
    path = FIXTURES / "index.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


INDEX = fixture_index()


def load_fixture(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        return json.load(fh)


class FixtureTransport:
    def __call__(self, endpoint: str) -> Any:
        name = INDEX.get(endpoint)
        if name is None:
            raise AssertionError(
                f"test requested unrecorded endpoint {endpoint!r}. "
                f"Recorded: {sorted(INDEX)}"
            )
        return load_fixture(name)


def fixture_client() -> sleeper.SleeperClient:
    tmp = tempfile.mkdtemp(prefix="ff-edge-proj-")
    return sleeper.SleeperClient(tmp, transport=FixtureTransport())


def projections_endpoint() -> str | None:
    for ep in INDEX:
        if ep.startswith("projections/nfl/"):
            return ep
    return None


# ---------------------------------------------------------------------------
# Scoring derivation
# ---------------------------------------------------------------------------


class TestScoringKind(unittest.TestCase):
    def test_derived_from_league_not_hardcoded(self) -> None:
        self.assertEqual(scoring_kind({"scoring_settings": {"rec": 1.0}}), "ppr")
        self.assertEqual(scoring_kind({"scoring_settings": {"rec": 0.5}}), "half_ppr")
        self.assertEqual(scoring_kind({"scoring_settings": {"rec": 0}}), "std")

    def test_missing_or_junk_defaults_to_ppr(self) -> None:
        self.assertEqual(scoring_kind({}), "ppr")
        self.assertEqual(scoring_kind({"scoring_settings": {}}), "ppr")
        self.assertEqual(scoring_kind({"scoring_settings": {"rec": "x"}}), "ppr")

    @unittest.skipUnless(INDEX.get("league/1379937879198822401"), "no league fixture")
    def test_real_league_is_full_ppr(self) -> None:
        league = load_fixture(INDEX["league/1379937879198822401"])
        self.assertEqual(scoring_kind(league), "ppr")


# ---------------------------------------------------------------------------
# Name normalization (the CSV upgrade path)
# ---------------------------------------------------------------------------


class TestNormalizeName(unittest.TestCase):
    def test_punctuation_and_case(self) -> None:
        self.assertEqual(normalize_name("D.J. Moore"), normalize_name("DJ Moore"))
        self.assertEqual(normalize_name("Ja'Marr Chase"), "jamarr chase")

    def test_generational_suffixes(self) -> None:
        self.assertEqual(normalize_name("Odell Beckham Jr."),
                         normalize_name("Odell Beckham"))
        self.assertEqual(normalize_name("Michael Pittman Jr"),
                         normalize_name("Michael Pittman"))

    def test_accents(self) -> None:
        self.assertEqual(normalize_name("Amon-Ra St. Brown"),
                         normalize_name("Amon Ra St Brown"))

    def test_empty_input(self) -> None:
        self.assertEqual(normalize_name(""), "")

    def test_index_keys_on_position_too(self) -> None:
        # Real collision risk: same name, different position.
        a = vorp.Player(name="Michael Thomas", pos="WR", player_id="1")
        b = vorp.Player(name="Michael Thomas", pos="DEF", player_id="2")
        idx = build_name_index([a, b])
        self.assertEqual(idx[("michael thomas", "WR")].player_id, "1")
        self.assertEqual(idx[("michael thomas", "DEF")].player_id, "2")


# ---------------------------------------------------------------------------
# Sleeper source
# ---------------------------------------------------------------------------


@unittest.skipUnless(projections_endpoint(), "no projections fixture")
class TestFromSleeper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_fixture(INDEX[projections_endpoint()])
        cls.client = fixture_client()
        cls.proj = from_sleeper(cls.client, 2026)

    def test_keyed_by_player_id(self) -> None:
        self.assertTrue(self.proj)
        for pid, line in self.proj.items():
            self.assertIsInstance(pid, str)
            self.assertIsInstance(line, ProjectedLine)
            self.assertGreater(line.proj, 0)

    def test_rows_without_points_are_skipped_not_zeroed(self) -> None:
        # Treating "no projection" as 0.0 would plant deep-bench players
        # exactly at replacement level instead of below it.
        missing = [r for r in self.rows
                   if (r.get("stats") or {}).get("pts_ppr") is None]
        self.assertTrue(missing, "fixture should contain a row with no projection")
        for r in missing:
            self.assertNotIn(str(r["player_id"]), self.proj)

    def test_scoring_kind_selects_different_numbers(self) -> None:
        ppr = from_sleeper(self.client, 2026, kind="ppr")
        std = from_sleeper(self.client, 2026, kind="std")
        # At least one RB/WR must differ between PPR and standard.
        self.assertTrue(any(ppr[k].proj != std[k].proj for k in ppr if k in std))

    def test_adp_carried(self) -> None:
        self.assertTrue([l for l in self.proj.values() if l.adp is not None],
                        "ADP should come through — survival math depends on it")

    def test_unrecorded_season_raises_rather_than_calling_out(self) -> None:
        with self.assertRaises(AssertionError):
            from_sleeper(self.client, 1999)


# ---------------------------------------------------------------------------
# Attaching
# ---------------------------------------------------------------------------


class TestAttach(unittest.TestCase):
    def players(self) -> list[vorp.Player]:
        return [
            vorp.Player(name="A", pos="RB", player_id="1"),
            vorp.Player(name="B", pos="WR", player_id="2"),
            vorp.Player(name="C", pos="TE", player_id="3"),
        ]

    def test_sets_proj_and_adp(self) -> None:
        ps = self.players()
        rep = attach(ps, {"1": ProjectedLine(250.0, 12.5),
                          "2": ProjectedLine(200.0)})
        self.assertEqual(rep.matched, 2)
        self.assertEqual(ps[0].proj, 250.0)
        self.assertEqual(ps[0].adp, 12.5)
        self.assertEqual(ps[1].proj, 200.0)
        self.assertIsNone(ps[1].adp)

    def test_unprojected_stay_at_zero_and_are_reported(self) -> None:
        ps = self.players()
        rep = attach(ps, {"1": ProjectedLine(250.0)})
        self.assertEqual(ps[2].proj, 0.0)
        self.assertEqual([p.name for p in rep.unmatched_players], ["B", "C"])

    def test_unused_source_rows_counted(self) -> None:
        rep = attach(self.players(), {"1": ProjectedLine(1.0),
                                      "999": ProjectedLine(2.0)})
        self.assertEqual(rep.unused_source_ids, 1)

    def test_summary_is_readable(self) -> None:
        rep = attach(self.players(), {"1": ProjectedLine(1.0)})
        self.assertIn("1 players projected", rep.summary())

    def test_players_without_ids_are_not_matched(self) -> None:
        ps = [vorp.Player(name="X", pos="RB")]
        rep = attach(ps, {"None": ProjectedLine(100.0)})
        self.assertEqual(rep.matched, 0)
        self.assertEqual(ps[0].proj, 0.0)


# ---------------------------------------------------------------------------
# CSV source
# ---------------------------------------------------------------------------


class TestFromCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="ff-edge-csv-")
        self.players = [
            vorp.Player(name="Ja'Marr Chase", pos="WR", player_id="100"),
            vorp.Player(name="Odell Beckham Jr.", pos="WR", player_id="101"),
            vorp.Player(name="Bijan Robinson", pos="RB", player_id="102"),
        ]

    def write(self, text: str) -> str:
        path = Path(self.dir) / "p.csv"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_matches_through_normalization(self) -> None:
        path = self.write(
            "name,pos,points\n"
            "JaMarr Chase,WR,311.1\n"
            "Odell Beckham,WR,90.0\n"
            "Bijan Robinson,RB,324.9\n"
        )
        proj, unmatched = from_csv(path, self.players)
        self.assertEqual(unmatched, [])
        self.assertEqual(proj["100"].proj, 311.1)
        self.assertEqual(proj["101"].proj, 90.0)

    def test_unmatched_rows_returned_not_swallowed(self) -> None:
        path = self.write("name,pos,points\nNobody Here,WR,100\n")
        proj, unmatched = from_csv(path, self.players)
        self.assertEqual(proj, {})
        self.assertEqual(unmatched, ["Nobody Here (WR)"])

    def test_position_mismatch_is_a_miss(self) -> None:
        path = self.write("name,pos,points\nBijan Robinson,WR,300\n")
        _, unmatched = from_csv(path, self.players)
        self.assertEqual(unmatched, ["Bijan Robinson (WR)"])

    def test_missing_column_raises_with_the_columns_it_found(self) -> None:
        path = self.write("player,position,fpts\nX,WR,1\n")
        with self.assertRaises(ValueError) as ctx:
            from_csv(path, self.players)
        self.assertIn("player", str(ctx.exception))

    def test_thousands_separator_and_blank_rows(self) -> None:
        path = self.write(
            "name,pos,points\nBijan Robinson,RB,\"1,024.5\"\nJa'Marr Chase,WR,\n"
        )
        proj, _ = from_csv(path, self.players)
        self.assertEqual(proj["102"].proj, 1024.5)
        self.assertNotIn("100", proj)

    def test_adp_column_optional(self) -> None:
        path = self.write("name,pos,points,adp\nBijan Robinson,RB,324.9,1.8\n")
        proj, _ = from_csv(path, self.players, adp_col="adp")
        self.assertEqual(proj["102"].adp, 1.8)


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------


@unittest.skipUnless(projections_endpoint() and INDEX.get("players/nfl"),
                     "fixtures missing")
class TestEndToEnd(unittest.TestCase):
    def test_fixture_pool_produces_a_real_board(self) -> None:
        client = fixture_client()
        players = sleeper.normalize_players(client.get_players())
        proj = from_sleeper(client, 2026)
        report = attach(players, proj)
        self.assertGreater(report.matched, 0)

        league = vorp.LeagueConfig(
            n_teams=12,
            roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                              "K", "DEF"] + ["BN"] * 6,
        )
        vorp.assign_vorp(players, league)
        projected = [p for p in players if p.proj > 0]
        self.assertTrue(projected)
        # A board with real numbers has spread; the all-zeros state does not.
        self.assertGreater(max(p.vorp for p in projected),
                           min(p.vorp for p in projected))

        vorp.assign_tiers(players, "RB")
        rbs = [p for p in players if p.pos == "RB" and p.proj > 0]
        if len(rbs) >= 3:
            self.assertTrue(all(p.tier >= 1 for p in rbs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
