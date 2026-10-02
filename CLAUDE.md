# ff-edge

A fantasy football draft/waiver/trade edge tool. Two leagues are targeted:
a 12-team full-PPR redraft league and a dynasty league. Same engine, different
projection inputs.

The goal is **not** better projections — we will never beat consensus by much.
The edge lives in three places Sleeper's own UI is blind to:

1. League-specific VORP with a correctly-solved replacement level
2. FAAB bids priced as expected points added over the current worst starter
3. Trade evaluation against actual roster construction and playoff schedule

---

## Design invariants

These are deliberate decisions. Do not "simplify" them without asking.

### Replacement level is solved iteratively, not hardcoded

`vorp.solve_replacement_levels()` does not use a static positional cutoff.
The naive approach (12 teams x 2 RB slots => RB24 is replacement) is wrong
because flex slots pull extra RB/WR/TE into starting lineups. We solve for it:

- Start with dedicated-slot-only starter counts
- Compute provisional baselines and VORP
- Let the highest-VORP flex-eligible leftovers claim flex slots
- Observe which positions they came from, expand starter counts, repeat
- Converges in 2-3 passes

In 12-team full PPR this typically lands around RB28 / WR41 / TE15 rather than
RB24 / WR36 / TE12. **Never replace this with a fixed rank cutoff.**

### Tiers use a two-rule detector, not gap-only

`vorp.assign_tiers()` opens a new tier on either:

- **WIDTH** — cumulative VORP drop from the tier's leader exceeds
  `width_frac * positional VORP range`
- **CLIFF** — a single gap exceeds `mean + cliff_z * stdev` of gaps

Gap-only detection was tried and failed: on a smoothly decaying curve the
early gaps all sit slightly above the mean, so it manufactures a run of
singleton tiers at the top of the board, which is exactly where the decisions
matter. The width rule is what makes tiers behave as fungibility bands.

### League config is parsed from Sleeper, never hardcoded

`LeagueConfig` takes `roster_positions` verbatim from the Sleeper league
endpoint. Scoring, team count, and flex types all derive from the league
itself so the same code serves both leagues. Do not add league-specific
branches; add config.

### Draft pick ownership is per-roster, not per-slot

`DraftState` resolves who owns pick N through `slot_to_roster_id` **plus**
`/league/{id}/traded_picks`, keyed `(round, original_roster_id) -> owner_id`
and filtered to the draft's season.

`draft_order` gives a user's *original* slot and is fixed at draft creation.
It does not survive trades. In the recorded 2025 RE Fantasy draft, 62 of 192
picks had changed hands and 7 of the user's own 16 picks were acquired — 
computing from `draft_order` alone put round 4 at pick 41 instead of 37, and
was wrong for a third of the board. This is a normal league, not an edge case.

The season filter is load-bearing for dynasty, where picks are traded years
out; applying a 2027 trade to the 2025 draft hands picks to the wrong team.

### Dynasty is a projection-input problem, not an engine problem

Single-season VORP is the wrong objective for dynasty. When we get there it
becomes a discounted multi-year points stream with an age curve applied,
fed into the *same* replacement-level solver. Do not fork the engine.

---

## Sleeper API notes

Base URL `https://api.sleeper.app/v1`. Read-only, no auth, no API key.

- `GET /league/{league_id}` — settings, `roster_positions`, `total_rosters`,
  `scoring_settings`
- `GET /league/{league_id}/rosters`, `/users`, `/matchups/{week}`,
  `/transactions/{week}`
- `GET /league/{league_id}/drafts` -> `GET /draft/{draft_id}` and
  `GET /draft/{draft_id}/picks`
- `GET /players/nfl` — ~5MB, cache to disk, refresh at most once per day
- `GET /user/{username}` -> `user_id`, then
  `GET /user/{user_id}/leagues/nfl/{season}`

Constraints:

- **Stay under 1000 calls/min** or the IP gets blocked. Draft polling should
  be a few seconds between calls at most, never a tight loop.
- Draft picks return a `metadata` object carrying `position`, `team`, and
  `injury_status`. Use it — it saves a join against the players blob.
- There is an undocumented projections endpoint floating around in community
  wrappers. Do not depend on it as the primary projection source.
- Verify response shapes against a live call before trusting any structure
  described here or in training data.

---

## Conventions

- Python 3.10+ (builtin generics and `X | None` unions are used throughout)
- Stdlib-only in `vorp.py` — keep the math layer dependency-free and testable
- Network code lives behind a cache layer; the math layer never makes calls
- Every new module gets a synthetic-data test like `test_synthetic.py` that
  runs with no network access
- Add deps to `requirements.txt` as they are introduced

## Status

- [x] `vorp.py` — flex-aware replacement level + tier detection
- [x] `sleeper.py` — cached client, league config, live draft state
      (`record_fixtures.py` re-records `tests/fixtures/`; it scrubs identity,
      so run it rather than hand-editing fixtures)
- [x] `projections.py` — source-agnostic ingest; Sleeper feed + generic CSV
      (Sleeper's feed is rotowire only, i.e. one provider, not a consensus —
      swapping in FantasyPros is a new function here, not a change elsewhere)
- [x] `market.py` — Flock ADP (6 platforms) + expert consensus, keyed by
      player_id. Sleeper ADP predicts our room and is kept SEPARATE from the
      cross-platform mean on purpose; the per-player ADP std dev is the
      survival-curve width. Files live in `data/`, not committed to fixtures.
- [x] `survival.py` — P(available at pick N) from Normal(ADP, cross-platform
      SD). Both parameters measured from the Flock export, never assumed;
      SD_FLOOR keeps it from becoming a step function. `cost_of_waiting` is
      the draft-day number: VORP now minus expected best available at that
      position next pick.
- [x] `roster.py` — marginal value against OUR starting lineup. Empty slots
      are filled with replacement-level players built from the solved
      baselines, so marginal value reduces exactly to VORP on an empty roster
      and decays as slots fill. Do not replace this with a positional-need
      fudge factor.
- [x] Streamlit dashboard — `streamlit run dashboard.py`
