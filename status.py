"""
Show current state of a league through the data layer. Read-only.

    python3 status.py                    # league from $SLEEPER_LEAGUE_ID
    python3 status.py <league_id>
    python3 status.py <league_id> --offline   # cache only, no network
"""

from __future__ import annotations

import os
import sys

import sleeper

DEFAULT_LEAGUE = os.environ.get("SLEEPER_LEAGUE_ID", "")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    league_id = args[0] if args else DEFAULT_LEAGUE
    if not league_id:
        sys.exit("usage: python3 status.py <league_id>  (or set SLEEPER_LEAGUE_ID)")
    offline = "--offline" in sys.argv

    client = sleeper.SleeperClient(offline=offline)

    league = client.get_league(league_id)
    cfg = sleeper.build_league_config(client, league_id)

    print(f"{league.get('name')}  |  {league.get('season')} "
          f"|  status: {league.get('status')}")
    print(f"  {cfg.n_teams} teams")
    print(f"  dedicated : {cfg.dedicated}")
    print(f"  flex      : {cfg.flex_slots}")
    print(f"  scored    : {sorted(cfg.scored_positions)}")

    players = sleeper.normalize_players(client.get_players())
    by_pos: dict[str, int] = {}
    for p in players:
        by_pos[p.pos] = by_pos.get(p.pos, 0) + 1
    print(f"\nplayer pool: {len(players)}")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(by_pos.items())))
    projected = sum(1 for p in players if p.proj)
    print(f"  with projections: {projected}"
          + ("   <- projections.py not built yet, so VORP is all zeros"
             if not projected else ""))

    drafts = client.get_drafts(league_id)
    print(f"\ndrafts: {len(drafts)}")
    for d in drafts:
        s = d.get("settings") or {}
        print(f"  {d['draft_id']}  {d.get('type')}  {d.get('status')}  "
              f"{s.get('teams')} teams x {s.get('rounds')} rounds"
              + (f"  reversal_round={s['reversal_round']}"
                 if s.get("reversal_round") else ""))

        state = sleeper.DraftState.from_client(client, d["draft_id"])
        made = state.picks_made
        total = state.meta.n_teams * state.meta.rounds
        print(f"     picks made: {made}/{total}   traded picks: {len(state.traded)}")

        if state.meta.draft_order:
            print(f"     {'user':<20}{'slot':>6}{'roster':>8}{'next pick':>11}"
                  f"{'picks left':>12}")
            rows = sorted(state.meta.draft_order.items(), key=lambda kv: kv[1])
            for user_id, slot in rows:
                nxt = state.my_next_pick(user_id)
                remaining = [n for n in state.pick_numbers_for_user(user_id) if n > made]
                print(f"     {user_id:<20}{slot:>6}"
                      f"{state.roster_for_user(user_id):>8}"
                      f"{str(nxt) if nxt else '-':>11}{len(remaining):>12}")
        else:
            print("     draft order not set yet (pre-draft)")

    print(f"\nnetwork calls: {client.network_calls}   cache hits: {client.cache_hits}"
          f"   rate-limit budget used: {client.limiter.in_window}"
          f"/{client.limiter.max_calls} per minute")


if __name__ == "__main__":
    main()
