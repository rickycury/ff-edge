"""
Offline tests for survival.py and roster.py.

Both modules are pure math over synthetic inputs — no network, no fixtures.
(Kept in one file since the two modules are two halves of the same draft-day
decision; split them if that ever stops being true.)

    python3 -m unittest test_draft_math -v
"""

from __future__ import annotations

import unittest

import roster
import survival
import vorp
from roster import RosterValuer, optimal_lineup, replacement_players
from survival import (
    Candidate,
    cost_of_waiting,
    expected_best_available,
    expected_survivors,
    normal_cdf,
    survival_probability,
)


def league(**kw) -> vorp.LeagueConfig:
    positions = kw.pop("roster_positions", None) or [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"
    ] + ["BN"] * 6
    return vorp.LeagueConfig(n_teams=kw.pop("n_teams", 12),
                             roster_positions=positions)


def player(name, pos, proj, vorp_=0.0) -> vorp.Player:
    p = vorp.Player(name=name, pos=pos, proj=proj, player_id=name)
    p.vorp = vorp_
    return p


def cand(name, pos, vorp_, adp, sd, gone=False) -> Candidate:
    return Candidate(player=player(name, pos, 0.0, vorp_), adp=adp, sd=sd, gone=gone)


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


class TestNormalCdf(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=6)
        self.assertAlmostEqual(normal_cdf(1.96), 0.975, places=3)
        self.assertAlmostEqual(normal_cdf(-1.96), 0.025, places=3)


class TestSurvivalProbability(unittest.TestCase):
    def test_at_his_own_adp_it_is_a_coin_flip(self) -> None:
        self.assertAlmostEqual(survival_probability(50, 10, 50), 0.5, places=6)

    def test_monotonic_in_pick(self) -> None:
        probs = [survival_probability(50, 10, n) for n in range(20, 81, 10)]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_spread_is_what_separates_equal_adps(self) -> None:
        # The whole point of carrying SD: same ADP, different risk.
        tight = survival_probability(76, 4, 90)
        loose = survival_probability(76, 25, 90)
        self.assertLess(tight, loose)
        self.assertLess(tight, 0.01)
        self.assertGreater(loose, 0.2)

    def test_drafted_player_never_survives(self) -> None:
        self.assertEqual(survival_probability(200, 5, 10, already_gone=True), 0.0)

    def test_unpriced_player_survives(self) -> None:
        # No ADP anywhere means the market has no opinion, not that he's hot.
        self.assertEqual(survival_probability(None, None, 50), 1.0)

    def test_missing_sd_falls_back_proportionally(self) -> None:
        p = survival_probability(100, None, 100)
        self.assertAlmostEqual(p, 0.5, places=6)
        # Deeper players get a wider assumed spread.
        self.assertGreater(survival.fallback_sd(200), survival.fallback_sd(20))

    def test_sd_floor_prevents_step_function(self) -> None:
        p = survival_probability(50, 0.0, 51)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 0.5)

    def test_probability_stays_in_range(self) -> None:
        for pick in (1, 50, 500):
            for sd in (1, 10, 60):
                p = survival_probability(50, sd, pick)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)


class TestExpectedSurvivors(unittest.TestCase):
    def test_sums_probabilities(self) -> None:
        pool = [cand(f"p{i}", "TE", 10.0, 50, 10) for i in range(4)]
        self.assertAlmostEqual(expected_survivors(pool, 50), 2.0, places=6)

    def test_drafted_players_do_not_count(self) -> None:
        pool = [cand("a", "TE", 10, 50, 10, gone=True),
                cand("b", "TE", 10, 50, 10)]
        self.assertAlmostEqual(expected_survivors(pool, 50), 0.5, places=6)


class TestExpectedBestAvailable(unittest.TestCase):
    def test_certain_survivor_sets_the_floor(self) -> None:
        pool = [cand("stud", "RB", 100.0, 1, 1), cand("safe", "RB", 40.0, 400, 5)]
        exp, _ = expected_best_available(pool, 100)
        # The stud is long gone; the safe one is certain -> expect ~his value.
        self.assertAlmostEqual(exp, 40.0, places=2)

    def test_probabilities_of_being_best_sum_sensibly(self) -> None:
        pool = [cand("a", "WR", 50, 30, 8), cand("b", "WR", 40, 40, 8),
                cand("c", "WR", 30, 200, 8)]
        _, breakdown = expected_best_available(pool, 45)
        total = sum(p for _, p in breakdown)
        self.assertLessEqual(total, 1.0 + 1e-9)
        self.assertGreater(total, 0.9)  # someone is almost surely available

    def test_empty_pool(self) -> None:
        exp, breakdown = expected_best_available([], 20)
        self.assertEqual(exp, 0.0)
        self.assertEqual(breakdown, [])

    def test_better_player_ranked_first_regardless_of_input_order(self) -> None:
        pool = [cand("low", "RB", 10, 300, 5), cand("high", "RB", 90, 300, 5)]
        _, breakdown = expected_best_available(pool, 400)
        self.assertEqual(breakdown[0][0].player.name, "high")


class TestCostOfWaiting(unittest.TestCase):
    def test_certain_survivor_costs_nothing(self) -> None:
        # He'll obviously still be there -> no reason to spend the pick now.
        c = cand("safe", "TE", 50.0, 300, 5)
        self.assertAlmostEqual(cost_of_waiting(c, [c], 100), 0.0, places=2)

    def test_about_to_vanish_costs_his_edge_over_the_next_guy(self) -> None:
        target = cand("vanishing", "TE", 60.0, 20, 3)
        backup = cand("backup", "TE", 20.0, 300, 5)
        cost = cost_of_waiting(target, [target, backup], 60)
        self.assertGreater(cost, 35.0)

    def test_deep_position_is_cheap_to_wait_on(self) -> None:
        # Five interchangeable players -> losing one costs almost nothing.
        pool = [cand(f"wr{i}", "WR", 50.0 - i, 40 + i * 3, 8) for i in range(5)]
        shallow = [cand("te1", "TE", 50.0, 40, 8), cand("te2", "TE", 5.0, 300, 8)]
        deep_cost = cost_of_waiting(pool[0], pool, 60)
        thin_cost = cost_of_waiting(shallow[0], shallow, 60)
        self.assertLess(deep_cost, thin_cost)


class TestRecommend(unittest.TestCase):
    def test_ranks_by_cost_and_skips_drafted(self) -> None:
        pool = [
            cand("gone", "RB", 200.0, 1, 2, gone=True),
            cand("urgent", "TE", 60.0, 22, 3),
            cand("patient", "WR", 58.0, 300, 10),
        ]
        rows = survival.recommend(pool, next_pick=60)
        names = [r["player"].name for r in rows]
        self.assertNotIn("gone", names)
        self.assertEqual(names[0], "urgent")

    def test_position_filter(self) -> None:
        pool = [cand("a", "TE", 60.0, 22, 3), cand("b", "WR", 60.0, 22, 3)]
        rows = survival.recommend(pool, 60, positions=["TE"])
        self.assertEqual({r["pos"] for r in rows}, {"TE"})


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


class TestOptimalLineup(unittest.TestCase):
    def test_fills_dedicated_then_flex(self) -> None:
        lg = league()
        ps = [player("qb1", "QB", 300), player("rb1", "RB", 200),
              player("rb2", "RB", 180), player("rb3", "RB", 170),
              player("wr1", "WR", 190), player("wr2", "WR", 150),
              player("te1", "TE", 120), player("k1", "K", 100),
              player("d1", "DEF", 90)]
        lu = optimal_lineup(ps, lg)
        # RB3 (170) is the best leftover and should take the flex.
        self.assertIn("FLEX", lu.assignment)
        self.assertEqual(lu.assignment["FLEX"][0].name, "rb3")
        self.assertAlmostEqual(lu.points, sum(p.proj for p in ps), places=6)

    def test_best_players_start(self) -> None:
        lg = league(roster_positions=["RB", "BN"])
        ps = [player("good", "RB", 200), player("bad", "RB", 50)]
        lu = optimal_lineup(ps, lg)
        self.assertEqual(lu.assignment["RB"][0].name, "good")
        self.assertEqual([b.name for b in lu.bench], ["bad"])

    def test_narrow_flex_filled_before_wide_one(self) -> None:
        # A SUPER_FLEX would happily eat the only WR; REC_FLEX can't use a QB,
        # so it must be served first or a slot goes empty.
        lg = league(roster_positions=["SUPER_FLEX", "REC_FLEX"])
        ps = [player("qb", "QB", 300), player("wr", "WR", 200)]
        lu = optimal_lineup(ps, lg)
        self.assertAlmostEqual(lu.points, 500.0, places=6)
        self.assertEqual(lu.assignment["REC_FLEX"][0].name, "wr")
        self.assertEqual(lu.assignment["SUPER_FLEX"][0].name, "qb")

    def test_empty_roster(self) -> None:
        self.assertEqual(optimal_lineup([], league()).points, 0.0)


class TestReplacementPlayers(unittest.TestCase):
    def test_covers_every_starting_slot(self) -> None:
        lg = league()
        base = {"QB": 250.0, "RB": 120.0, "WR": 130.0, "TE": 90.0,
                "K": 100.0, "DEF": 80.0}
        fillers = replacement_players(base, lg)
        starters = sum(lg.dedicated.values()) + sum(lg.flex_slots.values())
        self.assertEqual(len(fillers), starters)

    def test_flex_filler_uses_best_eligible_replacement(self) -> None:
        lg = league()
        base = {"QB": 250.0, "RB": 120.0, "WR": 130.0, "TE": 90.0,
                "K": 100.0, "DEF": 80.0}
        flex = [f for f in replacement_players(base, lg) if "FLEX" in f.name]
        self.assertEqual(flex[0].proj, 130.0)  # WR is the best flex-eligible


class TestRosterValuer(unittest.TestCase):
    BASE = {"QB": 250.0, "RB": 120.0, "WR": 130.0, "TE": 90.0,
            "K": 100.0, "DEF": 80.0}

    def test_empty_roster_reduces_to_vorp(self) -> None:
        # The property that keeps this consistent with the rest of the board.
        lg = league()
        v = RosterValuer([], lg, self.BASE)
        rb = player("elite rb", "RB", 300)
        self.assertAlmostEqual(v.marginal_value(rb), 300 - 120, places=6)

    def test_value_decays_as_the_position_fills(self) -> None:
        lg = league()
        empty = RosterValuer([], lg, self.BASE)
        stacked = RosterValuer(
            [player("rb1", "RB", 320), player("rb2", "RB", 310),
             player("rb3", "RB", 300)],
            lg, self.BASE,
        )
        rb4 = player("rb4", "RB", 250)
        self.assertGreater(empty.marginal_value(rb4), stacked.marginal_value(rb4))

    def test_scarce_slot_outranks_a_better_player_at_a_full_one(self) -> None:
        # The whole reason this module exists: raw points say take the RB.
        lg = league()
        v = RosterValuer(
            [player("rb1", "RB", 320), player("rb2", "RB", 310),
             player("rb3", "RB", 305), player("wr1", "WR", 300),
             player("wr2", "WR", 290)],
            lg, self.BASE,
        )
        rb4 = player("rb4", "RB", 260)   # higher raw projection
        te1 = player("te1", "TE", 200)   # lower, but TE slot is empty
        self.assertGreater(v.marginal_value(te1), v.marginal_value(rb4))

    def test_marginal_value_never_negative(self) -> None:
        lg = league()
        v = RosterValuer([player("te1", "TE", 300)], lg, self.BASE)
        scrub = player("scrub", "TE", 1.0)
        self.assertGreaterEqual(v.marginal_value(scrub), 0.0)

    def test_positional_need_flags_the_hole(self) -> None:
        lg = league()
        v = RosterValuer(
            [player("rb1", "RB", 320), player("rb2", "RB", 310),
             player("rb3", "RB", 300)],
            lg, self.BASE,
        )
        pool = [player("rbX", "RB", 280), player("teX", "TE", 210),
                player("wrX", "WR", 280)]
        need = v.positional_need(pool)
        self.assertGreater(need["TE"], need["RB"])

    def test_starters_includes_replacement_fillers(self) -> None:
        lg = league()
        v = RosterValuer([player("qb1", "QB", 400)], lg, self.BASE)
        lu = v.starters()
        self.assertGreater(lu.points, 400)
        self.assertEqual(lu.assignment["QB"][0].name, "qb1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
