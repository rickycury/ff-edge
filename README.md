# ff-edge: League-Specific Fantasy Football Valuation Engine

A draft, waiver, and trade valuation tool built on the Sleeper API. It doesn't
try to out-project the consensus. It looks for edge in the places a generic
ranking misses: **this league's** replacement level, **this room's** draft
behavior, and **this roster's** marginal needs.

## What it does

| Module | What it solves |
|---|---|
| `vorp.py` | Value over replacement player with a **flex-aware, iteratively solved** replacement level, plus tier detection |
| `survival.py` | Probability a player is still available at your next pick, and the expected cost of waiting |
| `roster.py` | Marginal value of a player against *your* current starting lineup |
| `market.py` | Cross-platform ADP (6 sites) and expert consensus; flags where the room misprices players |
| `projections.py` | Source-agnostic projection ingest (Sleeper feed or any CSV) |
| `sleeper.py` | Cached, rate-limit-aware Sleeper client; parses league settings and live draft state |
| `dashboard.py` | Streamlit draft board tying it all together |

## The interesting parts

**Replacement level is solved, not assumed.** The textbook approach says that
in a 12-team league with 2 RB slots, RB24 is replacement. That ignores flex
slots, which pull extra RB/WR/TE into starting lineups. The solver starts from
dedicated slots, lets the highest-value leftovers claim flex spots, observes
which positions they came from, and repeats until it converges (2–3 passes).
In 12-team full PPR it lands near RB28 / WR41 / TE15 instead of RB24 / WR36 / TE12.

**Tiers are fungibility bands.** A new tier opens when either the cumulative
drop from the tier leader exceeds a fraction of the positional range (WIDTH)
or a single gap is a statistical outlier (CLIFF). Gap-only detection
manufactures fake singleton tiers at the top of the board, which is exactly
where decisions matter most.

**Survival curves come from measured market data.** P(available at pick N) is
modeled as Normal(ADP, σ), where σ is the cross-platform ADP dispersion for
that player rather than an assumed constant, with a floor so it never collapses
to a step function.

**Draft-pick ownership handles trades.** Pick owners are resolved from slot
order *plus* the league's traded-picks history, filtered by season. In the real
league this was tested on, 62 of 192 picks had changed hands; using the
original slot order alone was wrong for a third of the board.

**Everything derives from league settings.** Roster slots, scoring, and team
count are parsed from the Sleeper league endpoint, so the same engine serves
redraft and dynasty leagues with no special cases.

## Quick start

```bash
pip install -r requirements.txt
python3 -m unittest test_draft_math test_market test_projections test_sleeper test_synthetic   # 160 tests, no network
export SLEEPER_LEAGUE_ID=<your league id>
python3 status.py              # league summary in the terminal
streamlit run dashboard.py     # full draft board
```

The math layer (`vorp.py`, `sleeper.py`, `projections.py`) is standard-library
only. Tests run offline against recorded, identity-scrubbed fixtures
(`record_fixtures.py` regenerates them).

**Market data:** ADP and expert ranks come from a Flock Fantasy export, which
isn't redistributed here. Drop `flock_adp.csv` and `flock_expert_ranks.csv`
into `data/` to enable the market and survival views; the rest works without them.

## Design notes

`CLAUDE.md` documents the design invariants (why each of the choices above was
made and what was tried first) along with Sleeper API constraints.
