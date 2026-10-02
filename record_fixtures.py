"""
Record scrubbed test fixtures from live Sleeper responses.

Run this by hand when an endpoint's shape needs re-verifying. The test suite
never calls it — tests read tests/fixtures/ and never touch the network.

    python3 record_fixtures.py --username <sleeper-username>
    python3 record_fixtures.py --players-only

Two things happen on the way to disk:

1. IDENTITY SCRUBBING. Fixtures live in git forever, and league endpoints
   carry leaguemates' user_ids, display names, usernames and avatar hashes.
   Every real user_id is replaced with a synthetic one via a single mapping
   applied across *all* fixtures — scrubbing users.json alone would be
   pointless (the same ids appear in draft_order, picks[].picked_by and
   rosters[].owner_id) and would also break the join my_next_pick depends on.
   Structure and key types are preserved exactly; only identity changes.

2. PLAYER TRIMMING. The real blob is 12k players / ~14.6MB. We keep a few
   dozen, with every field of each kept player intact, chosen to exercise the
   filter branches rather than just the happy path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sleeper

FIXTURE_DIR = Path(__file__).parent / "tests" / "fixtures"
INDEX_NAME = "index.json"


# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------


class Scrubber:
    """
    Stable real-identity -> synthetic-identity mapping.

    Synthetic values are derived from the *synthetic* id, never from the real
    one, so nothing about the original survives the transform — not even as a
    hash preimage.
    """

    def __init__(self) -> None:
        self._users: dict[str, str] = {}
        self.me: str | None = None

    def user_id(self, real: Any) -> Any:
        if real is None:
            return None
        real = str(real)
        if real not in self._users:
            self._users[real] = f"{100000000000000000 + len(self._users) + 1}"
        return self._users[real]

    def display_name(self, real_id: Any) -> str:
        synthetic = self.user_id(real_id)
        return f"manager_{list(self._users.values()).index(synthetic) + 1:02d}"

    def avatar(self, real_id: Any) -> str:
        # 32 hex chars, matching Sleeper's avatar hash shape.
        return hashlib.sha256(f"avatar:{self.user_id(real_id)}".encode()).hexdigest()[:32]

    def scrub_user(self, user: dict[str, Any]) -> dict[str, Any]:
        real = user.get("user_id")
        out = dict(user)
        out["user_id"] = self.user_id(real)
        out["display_name"] = self.display_name(real)
        if "username" in out:
            out["username"] = self.display_name(real)
        if out.get("avatar") is not None:
            out["avatar"] = self.avatar(real)
        meta = out.get("metadata")
        if isinstance(meta, dict):
            # Same allowlist reasoning as scrub_roster: user.metadata holds
            # team names, avatar urls and other self-authored strings.
            scrubbed: dict[str, Any] = {}
            for i, (k, v) in enumerate(sorted(meta.items())):
                if k == "team_name":
                    scrubbed[k] = f"Team {self.display_name(real)}"
                elif k == "avatar" and v:
                    scrubbed[k] = f"https://sleepercdn.com/avatars/{self.avatar(real)}"
                elif isinstance(v, str) and not v.startswith(("on", "off", "true", "false")):
                    scrubbed[k] = f"text_{i:03d}"
                else:
                    scrubbed[k] = v
            out["metadata"] = scrubbed
        return out

    # roster.metadata keys whose values are league-generated and safe to keep
    # verbatim. Everything else is treated as user-authored free text.
    SAFE_ROSTER_META = {
        "streak", "record",
        "allow_pn_inactive_starters", "allow_pn_player_injury_status",
        "allow_pn_scoring", "restrict_pn_scoring_starters_only",
    }

    def scrub_roster(self, roster: dict[str, Any]) -> dict[str, Any]:
        out = dict(roster)
        if "owner_id" in out:
            out["owner_id"] = self.user_id(out["owner_id"])
        if isinstance(out.get("co_owners"), list):
            out["co_owners"] = [self.user_id(c) for c in out["co_owners"]]

        meta = out.get("metadata")
        if isinstance(meta, dict):
            # Allowlist, not denylist. roster.metadata carries p_nick_<player_id>
            # keys holding nicknames typed by leaguemates — arbitrary free text,
            # frequently crude, occasionally worse. The key names are public
            # player ids and are kept so the shape survives; every value outside
            # the allowlist is replaced with a synthetic placeholder of the same
            # type. A denylist here is the wrong shape of defense: it fails open
            # on whatever Sleeper adds next.
            scrubbed: dict[str, Any] = {}
            for i, (k, v) in enumerate(sorted(meta.items())):
                if k in self.SAFE_ROSTER_META:
                    scrubbed[k] = v
                elif isinstance(v, str):
                    scrubbed[k] = f"text_{i:03d}"
                else:
                    scrubbed[k] = v
            out["metadata"] = scrubbed
        return out

    def scrub_draft(self, draft: dict[str, Any]) -> dict[str, Any]:
        out = dict(draft)
        if isinstance(out.get("draft_order"), dict):
            out["draft_order"] = {
                self.user_id(uid): slot for uid, slot in out["draft_order"].items()
            }
        if isinstance(out.get("creators"), list):
            out["creators"] = [self.user_id(c) for c in out["creators"]]
        if out.get("creator") is not None:
            out["creator"] = self.user_id(out["creator"])
        meta = out.get("metadata")
        if isinstance(meta, dict):
            meta = dict(meta)
            meta.pop("name", None)
            meta.pop("description", None)
            out["metadata"] = meta
        return out

    def scrub_pick(self, pick: dict[str, Any]) -> dict[str, Any]:
        out = dict(pick)
        if "picked_by" in out:
            # Empty string means autopick; preserve that distinction.
            out["picked_by"] = self.user_id(out["picked_by"]) if out["picked_by"] else out["picked_by"]
        # pick.metadata carries the NFL player (public info) — left intact.
        return out

    def scrub_league(self, league: dict[str, Any]) -> dict[str, Any]:
        out = dict(league)
        if out.get("avatar") is not None:
            out["avatar"] = hashlib.sha256(b"league-avatar").hexdigest()[:32]
        return out


# ---------------------------------------------------------------------------
# Player trimming
# ---------------------------------------------------------------------------

# Deliberate edge cases, so the fixture exercises the filter rather than just
# the happy path. Each is a (label, predicate) checked against the live blob.
EDGE_CASES = [
    ("def_entry", lambda p: p.get("position") == "DEF"),
    ("retired_inactive_flag", lambda p: p.get("position") in {"QB", "RB", "WR", "TE"}
        and p.get("active") is False),
    ("stale_active_inactive_status", lambda p: p.get("position") in {"QB", "RB", "WR", "TE"}
        and p.get("active") is True and p.get("status") == "Inactive"),
    ("injured_reserve", lambda p: p.get("position") in {"RB", "WR"}
        and p.get("active") is True and p.get("status") == "Injured Reserve"),
    ("has_injury_status", lambda p: p.get("position") in {"QB", "RB", "WR", "TE"}
        and p.get("active") is True and bool(p.get("injury_status"))),
    ("free_agent_no_team", lambda p: p.get("position") in {"RB", "WR"}
        and p.get("active") is True and p.get("status") == "Active" and not p.get("team")),
    ("non_fantasy_position", lambda p: p.get("position") in {"OL", "CB", "P"}),
]

TOP_N_PER_POS = {"QB": 8, "RB": 10, "WR": 10, "TE": 6, "K": 3}


def trim_players(blob: dict[str, Any]) -> dict[str, Any]:
    """
    A few dozen players, every field of each preserved verbatim.

    Selection is top-N by search_rank per position (so the fixture contains
    recognizable, genuinely draftable players) plus a handful of deliberate
    oddities that the normalizer's filter has to get right.
    """
    kept: dict[str, Any] = {}
    inf = float("inf")

    for pos, n in TOP_N_PER_POS.items():
        pool = [
            p for p in blob.values()
            if p.get("position") == pos and p.get("active") and p.get("status") == "Active"
        ]
        pool.sort(key=lambda p: (p.get("search_rank") if p.get("search_rank") is not None else inf))
        for p in pool[:n]:
            kept[str(p["player_id"])] = p

    for label, pred in EDGE_CASES:
        matches = [p for p in blob.values() if pred(p)]
        matches.sort(key=lambda p: (p.get("search_rank") if p.get("search_rank") is not None else inf))
        take = 2 if label == "def_entry" else 1
        if not matches:
            print(f"  !! edge case {label!r} matched nothing in the live blob")
            continue
        for p in matches[:take]:
            kept[str(p["player_id"])] = p

    return kept


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def write_fixture(index: dict[str, str], endpoint: str, name: str, payload: Any) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    index[endpoint] = f"{name}.json"
    size = path.stat().st_size
    print(f"  wrote {path.name:<28} {size:>9,} bytes   <- /{endpoint}")


def load_index() -> dict[str, str]:
    path = FIXTURE_DIR / INDEX_NAME
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_index(index: dict[str, str]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with (FIXTURE_DIR / INDEX_NAME).open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
        fh.write("\n")


def record_players(client: sleeper.SleeperClient, index: dict[str, str]) -> None:
    print("players/nfl ...")
    blob = client.get_players()
    print(f"  live blob: {len(blob):,} players")
    trimmed = trim_players(blob)
    print(f"  trimmed to {len(trimmed)} players, all fields intact")
    write_fixture(index, "players/nfl", "players_nfl", trimmed)
    return trimmed


def record_projections(index: dict[str, str], trimmed: dict[str, Any],
                       season: str) -> None:
    """
    Projections for exactly the players in the trimmed blob, so the two
    fixtures join. Rows are NFL projections — no identity to scrub.
    """
    import projections as proj_mod

    print(f"\nprojections/nfl/{season} ...")
    pc = proj_mod.projections_client()
    endpoint = proj_mod.sleeper_projections_endpoint(
        season, ("QB", "RB", "WR", "TE", "K", "DEF")
    )
    rows = pc.get(endpoint, proj_mod.TTL_PROJECTIONS) or []
    print(f"  live: {len(rows)} rows")
    keep_ids = set(trimmed)
    kept = [r for r in rows if str(r.get("player_id")) in keep_ids]
    with_pts = sum(1 for r in kept if (r.get("stats") or {}).get("pts_ppr") is not None)
    print(f"  trimmed to {len(kept)} rows ({with_pts} carrying pts_ppr)")
    write_fixture(index, endpoint, "projections", kept)


def record_league_and_draft(
    client: sleeper.SleeperClient,
    index: dict[str, str],
    username: str,
    season: str,
    league_id: str | None,
) -> None:
    scrub = Scrubber()

    print(f"user/{username} ...")
    user = client.get_user(username)
    real_me = user["user_id"]
    my_synthetic = scrub.user_id(real_me)
    # Recorded under the synthetic handle: the endpoint path is itself the
    # username, so keying the index by the real one would put it in git.
    scrubbed_me = scrub.scrub_user(user)
    write_fixture(index, f"user/{scrubbed_me['display_name']}", "user_me", scrubbed_me)

    if league_id is None:
        leagues = client.get_user_leagues(real_me, season)
        if not leagues:
            raise SystemExit(f"no {season} leagues for {username!r}")
        print(f"\n{season} leagues for this user:")
        for lg in leagues:
            print(f"  {lg['league_id']}  {lg.get('total_rosters')}-team  {lg.get('name')!r}")
        if len(leagues) > 1:
            raise SystemExit(
                "more than one league — re-run with --league-id to pick one"
            )
        league_id = leagues[0]["league_id"]

    print(f"\ncurrent season league {league_id} ...")
    current = client.get_league(league_id)
    write_fixture(index, f"league/{league_id}", "league_current",
                  scrub.scrub_league(current))

    prev_id = current.get("previous_league_id")
    if not prev_id or prev_id == "0":
        raise SystemExit(
            f"league {league_id} has no previous_league_id; cannot reach a "
            f"completed draft to validate ordering against"
        )

    print(f"\nprevious season league {prev_id} (this is the one with a real draft) ...")
    prev = client.get_league(prev_id)
    write_fixture(index, f"league/{prev_id}", "league_previous",
                  scrub.scrub_league(prev))

    users = client.get_users(prev_id)
    write_fixture(index, f"league/{prev_id}/users", "league_users",
                  [scrub.scrub_user(u) for u in users])

    rosters = client.get_rosters(prev_id)
    write_fixture(index, f"league/{prev_id}/rosters", "league_rosters",
                  [scrub.scrub_roster(r) for r in rosters])

    drafts = client.get_drafts(prev_id)
    write_fixture(index, f"league/{prev_id}/drafts", "league_drafts",
                  [scrub.scrub_draft(d) for d in drafts])

    # Traded picks reference roster_ids only, never user_ids — nothing to scrub.
    traded = client.get_traded_picks(prev_id)
    write_fixture(index, f"league/{prev_id}/traded_picks", "traded_picks", traded)

    completed = [d for d in drafts if d.get("status") == "complete"]
    if not completed:
        raise SystemExit(
            f"no completed draft in league {prev_id}; statuses were "
            f"{[d.get('status') for d in drafts]}"
        )
    draft_id = completed[0]["draft_id"]

    print(f"\ndraft {draft_id} ...")
    draft = client.get_draft(draft_id)
    settings = draft.get("settings") or {}
    print(f"  type={draft.get('type')!r} status={draft.get('status')!r} "
          f"teams={settings.get('teams')} rounds={settings.get('rounds')} "
          f"reversal_round={settings.get('reversal_round')!r}")
    write_fixture(index, f"draft/{draft_id}", "draft", scrub.scrub_draft(draft))

    picks = client.get_draft_picks(draft_id, ttl=0)
    print(f"  {len(picks)} picks")
    write_fixture(index, f"draft/{draft_id}/picks", "draft_picks",
                  [scrub.scrub_pick(p) for p in picks])

    # Ground truth for the tests: which synthetic user is "me", and which
    # overall picks actually landed on my roster. Ownership is taken from each
    # pick's roster_id rather than picked_by — picked_by is empty on autopicks,
    # and roster_id is what actually determines who got the player.
    my_slot = (draft.get("draft_order") or {}).get(str(real_me))
    my_roster = (draft.get("slot_to_roster_id") or {}).get(str(my_slot))
    my_picks = sorted(
        p["pick_no"] for p in picks
        if p.get("roster_id") == my_roster and p.get("pick_no")
    )
    traded_mine = sum(
        1 for p in picks
        if p.get("roster_id") == my_roster
        and (draft.get("slot_to_roster_id") or {}).get(str(p.get("draft_slot"))) != my_roster
    )
    meta = {
        "note": "recorded by record_fixtures.py; identities are synthetic",
        "my_user_id": my_synthetic,
        "my_draft_slot": my_slot,
        "my_roster_id": my_roster,
        "my_actual_pick_numbers": my_picks,
        "traded_pick_count": len(traded),
        "current_league_id": league_id,
        "previous_league_id": prev_id,
        "draft_id": draft_id,
    }
    write_fixture(index, "_ground_truth", "ground_truth", meta)
    print(f"\n  my slot: {my_slot} -> roster {my_roster}")
    print(f"  my {len(my_picks)} picks: {my_picks[:8]}...")
    print(f"  {traded_mine} of them acquired by trade; {len(traded)} traded "
          f"picks league-wide")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--username", help="Sleeper username")
    ap.add_argument("--league-id", default=None, help="current-season league_id")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--players-only", action="store_true")
    args = ap.parse_args()

    client = sleeper.SleeperClient()
    index = load_index()

    trimmed = record_players(client, index)
    record_projections(index, trimmed, args.season)
    if not args.players_only:
        if not args.username:
            ap.error("--username is required unless --players-only")
        record_league_and_draft(client, index, args.username, args.season, args.league_id)

    save_index(index)
    print(f"\nindex: {len(index)} endpoints -> {FIXTURE_DIR}/{INDEX_NAME}")


if __name__ == "__main__":
    main()
