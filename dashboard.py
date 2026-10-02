"""
ff-edge draft board.

    streamlit run dashboard.py

Shows the three things Sleeper's own UI cannot:
  1. VORP against this league's solved replacement level, not generic ranks
  2. Tiers as fungibility bands — who is interchangeable with whom
  3. Value vs ADP — who the room is systematically mispricing

Survival math ("will he last until my next pick?") is the next milestone and
is deliberately absent rather than faked.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import market
import projections
import roster as roster_mod
import sleeper
import survival
import vorp

DEFAULT_LEAGUE = os.environ.get("SLEEPER_LEAGUE_ID", "")  # set your own league id
ADP_CSV = "data/flock_adp.csv"
EXPERT_CSV = "data/flock_expert_ranks.csv"
POS_COLORS = {
    "QB": "#8b5cf6", "RB": "#10b981", "WR": "#3b82f6",
    "TE": "#f59e0b", "K": "#6b7280", "DEF": "#ef4444",
}

st.set_page_config(page_title="ff-edge draft board", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def load_board(league_id: str, players_ttl: int):
    """League config + projected, VORP'd, tiered player pool."""
    client = sleeper.SleeperClient()
    league = client.get_league(league_id)
    cfg = sleeper.build_league_config(client, league_id)
    players = sleeper.normalize_players(client.get_players(ttl=players_ttl))
    report, kind = projections.load_for_league(client, league, players)

    vorp.assign_vorp(players, cfg)
    for pos in sorted(cfg.scored_positions):
        vorp.assign_tiers(players, pos, max_rank=60)
    baselines, starters = vorp.solve_replacement_levels(players, cfg)

    # Market layer: Flock ADP (6 platforms) + expert consensus, keyed by
    # player_id. Optional — the board still works on rotowire alone if the
    # CSVs are absent.
    views: dict = {}
    market_note = "no Flock CSV loaded"
    if Path(ADP_CSV).exists():
        exp = EXPERT_CSV if Path(EXPERT_CSV).exists() else None
        views, mreport = market.load_all(ADP_CSV, exp, players)
        market_note = (
            f"{len(views)} players priced · "
            f"{len(mreport['adp_unmatched'])} ADP rows unmatched (all deep)"
        )

    rows = [
        {
            "player_id": p.player_id, "Player": p.name, "Pos": p.pos,
            "Team": p.team or "FA", "Proj": p.proj, "VORP": p.vorp,
            "Tier": p.tier, "Injury": p.injury_status or "",
        }
        for p in players if p.proj > 0
    ]
    df = pd.DataFrame(rows).sort_values("VORP", ascending=False).reset_index(drop=True)
    df.insert(0, "VORP Rank", df.index + 1)

    # Attach market columns. Sleeper ADP predicts OUR room; expert rank is the
    # value opinion; SD is how much the market disagrees with itself (the
    # survival-curve width, once that math lands).
    def col(pid, attr):
        v = views.get(pid)
        return getattr(v, attr) if v else None

    df["Sleeper ADP"] = df["player_id"].map(lambda p: col(p, "home_adp"))
    df["Market ADP"] = df["player_id"].map(lambda p: col(p, "market_adp"))
    # Baseline excluding our own platform. Round buckets key off THIS rather
    # than Market ADP so the bucket and the deviation aren't both measured
    # against a number Sleeper is itself part of.
    df["Outside ADP"] = df["player_id"].map(lambda p: col(p, "outside_market_adp"))
    df["Platforms"] = df["player_id"].map(lambda p: col(p, "n_platforms"))
    df["Expert"] = df["player_id"].map(lambda p: col(p, "expert_rank"))
    df["Disagreement"] = df["player_id"].map(lambda p: col(p, "adp_sd"))
    # The steal: how many picks later our room drafts him than he's worth here.
    # Positive = he falls to you below his value. This is the Flock-powered
    # replacement for the old rotowire ADP-minus-rank metric.
    df["Steal"] = df["Sleeper ADP"] - df["VORP Rank"]
    # How far our room diverges from everyone else on this player.
    df["Room vs Market"] = df["player_id"].map(lambda p: col(p, "home_vs_market"))

    return (league, cfg, df, baselines, starters, report.summary(), kind,
            market_note, players, views)


@st.cache_data(ttl=5, show_spinner=False)
def load_draft(draft_id: str):
    client = sleeper.SleeperClient()
    state = sleeper.DraftState.from_client(client, draft_id, picks_ttl=5)
    return (state.meta, state.picks, sorted(state.drafted_player_ids),
            state.picks_made, dict(state.traded))


@st.cache_data(ttl=600, show_spinner=False)
def load_drafts(league_id: str):
    return sleeper.SleeperClient().get_drafts(league_id)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("ff-edge")
league_id = st.sidebar.text_input("League ID", DEFAULT_LEAGUE)
stale_players = st.sidebar.checkbox(
    "Force refresh player blob", value=False,
    help="Normally cached 24h — Sleeper asks not to refetch the 14MB blob more often.",
)

if not league_id:
    st.info("Enter your Sleeper league ID in the sidebar (or set SLEEPER_LEAGUE_ID).")
    st.stop()

try:
    (league, cfg, df, baselines, starters, proj_summary, kind,
     market_note, all_players, market_views) = load_board(
        league_id, 0 if stale_players else sleeper.TTL_PLAYERS
    )
except Exception as exc:  # noqa: BLE001 - surface any config/network error in-app
    st.error(f"Could not load league {league_id}: {exc}")
    st.stop()

st.sidebar.markdown(f"**{league.get('name')}** · {league.get('season')}")
st.sidebar.caption(f"{cfg.n_teams} teams · {kind.replace('_', ' ')} scoring")
st.sidebar.caption(f"Market: {market_note}")
st.sidebar.caption(proj_summary)

drafts = load_drafts(league_id)
draft_options = {
    f"{d.get('status')} · {d.get('type')} · {d['draft_id'][-6:]}": d["draft_id"]
    for d in drafts
}
selected_draft = None
if draft_options:
    label = st.sidebar.selectbox("Draft", list(draft_options))
    selected_draft = draft_options[label]

hide_drafted = st.sidebar.checkbox("Hide drafted players", value=True)
positions = st.sidebar.multiselect(
    "Positions", sorted(cfg.scored_positions),
    default=[p for p in ["QB", "RB", "WR", "TE"] if p in cfg.scored_positions],
)

# ---------------------------------------------------------------------------
# Draft state
# ---------------------------------------------------------------------------

drafted: set[str] = set()
my_next = None
picks_until = None
meta = None
picks: list = []
picks_made = 0
user_label: str | None = None
my_pick_numbers: list[int] = []

if selected_draft:
    try:
        meta, picks, drafted_ids, picks_made, traded = load_draft(selected_draft)
        drafted = set(drafted_ids)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Draft unavailable: {exc}")
        meta = None

    if meta and meta.draft_order:
        user_label = st.sidebar.selectbox(
            "You are", sorted(meta.draft_order, key=lambda u: meta.draft_order[u])
        )
        state = sleeper.DraftState(meta, [{"pick_no": picks_made}] if picks_made else [],
                                   [{"season": meta.season, "round": r, "roster_id": o,
                                     "owner_id": w} for (r, o), w in traded.items()])
        try:
            my_next = state.my_next_pick(user_label)
            picks_until = state.picks_until(user_label)
            my_pick_numbers = state.pick_numbers_for_user(user_label)
        except NotImplementedError as exc:
            st.sidebar.info(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Pick order unavailable: {exc}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Draft board")

cols = st.columns(5)
cols[0].metric("Teams", cfg.n_teams)
cols[1].metric("Players on board", len(df),
               help="Players carrying a projection above zero.")
if meta:
    cols[2].metric("Picks made", f"{meta and picks_made}/{meta.n_teams * meta.rounds}")
    cols[3].metric("Your next pick", my_next if my_next else "—")
    cols[4].metric("Picks until you're up", picks_until if picks_until is not None else "—")
else:
    cols[2].metric("Draft", "not started")

with st.expander("Replacement level (solved, flex-aware)", expanded=False):
    st.caption(
        "Naive replacement is teams x dedicated slots. Flex slots pull extra "
        "players into starting lineups, so the real baseline sits deeper. "
        "This is where league-specific value comes from."
    )
    rep = pd.DataFrame([
        {
            "Pos": pos,
            "Naive starters": cfg.n_teams * cfg.dedicated.get(pos, 0),
            "Effective starters": starters.get(pos, 0),
            "Baseline pts": round(baselines.get(pos, 0.0), 1),
        }
        for pos in sorted(cfg.scored_positions)
    ])
    st.dataframe(rep, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

view = df[df["Pos"].isin(positions)] if positions else df
if hide_drafted and drafted:
    view = view[~view["player_id"].isin(drafted)]

tab_clock, tab_board, tab_tiers, tab_value, tab_room = st.tabs(
    ["On the clock", "Big board", "Tiers", "Value vs ADP", "Room vs Market"]
)

with tab_clock:
    st.caption(
        "**Cost of waiting** is the number to draft by: his value now, minus "
        "what we expect to salvage at his position when we pick again. A stud "
        "who will obviously last is not a good pick right now; a solid player "
        "about to disappear is. **Marginal** is what he adds to *our* starting "
        "lineup — it collapses once a position is covered."
    )

    auto_next = my_next if my_next else None

    ca, cb, cc = st.columns(3)
    with ca:
        pick_now = st.number_input(
            "Your pick", min_value=1, max_value=400,
            value=int(auto_next or 8), step=1,
            help="Auto-filled from the live draft once the order is set.",
        )
    with cb:
        # Use the real pick sequence when we have it. At a snake turn the gap
        # alternates wildly — slot 2 waits 21 picks, then only 3 — so an
        # estimated round-trip would badly misprice how long you actually wait.
        after = [n for n in my_pick_numbers if n > pick_now]
        default_next = after[0] if after else int(pick_now) + cfg.n_teams
        pick_next = st.number_input(
            "Your pick after that", min_value=int(pick_now) + 1, max_value=500,
            value=int(default_next), step=1,
            help="Auto-filled from your actual pick sequence, including trades.",
        )
    with cc:
        clock_pos = st.multiselect(
            "Positions", sorted(cfg.scored_positions),
            default=[p for p in ["QB", "RB", "WR", "TE"] if p in cfg.scored_positions],
            key="clock_pos",
        )

    # Who's already off the board, and what we already own.
    my_roster_players: list = []
    if meta and drafted and user_label:
        try:
            my_rid = sleeper.DraftState(meta, [], []).roster_for_user(user_label)
            mine = {str(p["player_id"]) for p in picks
                    if p.get("roster_id") == my_rid and p.get("player_id")}
            my_roster_players = [p for p in all_players if p.player_id in mine]
        except Exception:  # noqa: BLE001 - roster resolution is best-effort here
            my_roster_players = []

    pool_players = [p for p in all_players
                    if p.proj > 0 and p.player_id not in drafted]
    cands = survival.build_candidates(pool_players, market_views, drafted)

    valuer = roster_mod.RosterValuer(my_roster_players, cfg, baselines)
    shortlist = sorted(pool_players, key=lambda p: -p.vorp)[:80]
    need = valuer.positional_need(shortlist)

    st.markdown("**Positional need** — points the best available adds to our lineup")
    st.caption(
        "Not a depth chart guess: a position where the best available adds 60 "
        "points is a hole; one where he adds 4 is already covered."
        + (f"  Roster: {len(my_roster_players)} players."
           if my_roster_players else "  Roster is empty, so this equals VORP.")
    )
    need_df = pd.DataFrame(
        [{"Pos": k, "Marginal pts": v} for k, v in need.items()]
    ).sort_values("Marginal pts", ascending=False)
    st.dataframe(
        need_df, hide_index=True, width="stretch",
        column_config={"Marginal pts": st.column_config.NumberColumn(format="%.1f")},
    )

    st.divider()
    recs = survival.recommend(
        cands, next_pick=int(pick_next),
        positions=clock_pos or None, top_n=20,
    )
    if not recs:
        st.info("No available players match those filters.")
    else:
        rec_df = pd.DataFrame([{
            "Player": r["player"].name,
            "Pos": r["pos"],
            "Tier": r["player"].tier,
            "VORP": r["vorp"],
            "Marginal": valuer.marginal_value(r["player"]),
            "ADP": r["adp"],
            "Spread": r["sd"],
            "Lasts to next pick": r["survives"],
            "Cost of waiting": r["cost_of_waiting"],
        } for r in recs])
        st.markdown(
            f"**Take now, or lose him** — ranked by cost of waiting from pick "
            f"{int(pick_now)} to {int(pick_next)}"
        )
        st.dataframe(
            rec_df, hide_index=True, width="stretch", height=560,
            column_config={
                "VORP": st.column_config.NumberColumn(format="%.1f"),
                "Marginal": st.column_config.NumberColumn(
                    format="%.1f",
                    help="Points added to OUR starting lineup, given what we "
                         "already roster."),
                "ADP": st.column_config.NumberColumn(format="%.1f"),
                "Spread": st.column_config.NumberColumn(
                    format="%.0f",
                    help="Cross-platform ADP std dev — the width of his "
                         "survival curve."),
                "Lasts to next pick": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0.0, max_value=1.0),
                "Cost of waiting": st.column_config.NumberColumn(
                    format="%.1f",
                    help="VORP surrendered by passing on him now."),
            },
        )

        st.divider()
        st.markdown("**Tier survival** — expected players left at your next pick")
        st.caption(
            "When a tier's expected survivors drops below 1, the tier does not "
            "reach you. That is the moment to take one."
        )
        tier_rows = []
        for pos in (clock_pos or sorted(cfg.scored_positions)):
            pos_cands = [c for c in cands if c.player.pos == pos and c.player.tier]
            tiers: dict[int, list] = {}
            for c in pos_cands:
                tiers.setdefault(c.player.tier, []).append(c)
            for tier in sorted(tiers)[:4]:
                group = tiers[tier]
                tier_rows.append({
                    "Pos": pos, "Tier": tier, "Available": len(group),
                    "Expected survivors": survival.expected_survivors(
                        group, int(pick_next)),
                    "Best VORP": max(c.vorp for c in group),
                })
        if tier_rows:
            tdf = pd.DataFrame(tier_rows).sort_values(
                ["Expected survivors", "Best VORP"], ascending=[True, False])
            st.dataframe(
                tdf, hide_index=True, width="stretch",
                column_config={
                    "Expected survivors": st.column_config.NumberColumn(format="%.2f"),
                    "Best VORP": st.column_config.NumberColumn(format="%.1f"),
                },
            )

with tab_board:
    st.caption(
        "Ranked by value over replacement **in this league** — not by raw points. "
        "Sleeper ADP predicts *our* room; Expert is the 10-analyst consensus; "
        "Steal = how many picks past his value the room lets him fall."
    )
    st.dataframe(
        view[["VORP Rank", "Player", "Pos", "Team", "Tier", "Proj", "VORP",
              "Sleeper ADP", "Expert", "Disagreement", "Steal", "Injury"]],
        hide_index=True, width="stretch", height=620,
        column_config={
            "VORP": st.column_config.ProgressColumn(
                "VORP", format="%.1f", min_value=0.0,
                max_value=float(max(df["VORP"].max(), 1.0)),
            ),
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Sleeper ADP": st.column_config.NumberColumn(format="%.1f"),
            "Expert": st.column_config.NumberColumn(format="%.0f"),
            "Disagreement": st.column_config.NumberColumn(
                format="%.0f",
                help="Cross-platform ADP spread (std dev). High = the market "
                     "can't agree on him, so he could go much earlier than his "
                     "ADP — you can't safely wait.",
            ),
            "Steal": st.column_config.NumberColumn(
                format="%+.0f",
                help="Sleeper ADP minus VORP rank. Positive = the room drafts "
                     "him later than he's worth here.",
            ),
        },
    )

with tab_tiers:
    st.caption(
        "Tiers are fungibility bands: within a tier, take the cheapest. "
        "The gap between tiers is what you actually lose by waiting."
    )
    tier_cols = st.columns(len(positions) if positions else 1)
    for col, pos in zip(tier_cols, positions):
        with col:
            st.subheader(pos)
            pool = view[view["Pos"] == pos].head(30)
            for tier, group in pool.groupby("Tier", sort=True):
                st.markdown(
                    f"<div style='border-left:3px solid {POS_COLORS.get(pos,'#888')};"
                    f"padding-left:8px;margin:6px 0;'><b>Tier {tier}</b></div>",
                    unsafe_allow_html=True,
                )
                for _, r in group.iterrows():
                    st.write(f"{r['Player']} · {r['VORP']:.0f}")

with tab_value:
    st.caption(
        "Where the room is wrong. Positive means his average draft position is "
        "later than his value here — that is the pick you take."
    )

    # K and DEF are excluded on purpose. Their VORP spread is tiny (the whole
    # kicker position spans ~13 points against ~330 for RB), so a trivial value
    # difference swings their overall rank by dozens of places, while their ADP
    # sits at 170+ because everyone drafts them last. ADP-minus-rank then
    # explodes and they crowd out every real signal — the first build of this
    # tab confidently recommended a kicker as the biggest value in the draft.
    # Steal is now driven by Sleeper ADP (from Flock's 6-platform export),
    # which predicts our actual room, rather than rotowire's own ADP.
    # K and DEF stay excluded: their VORP spread is tiny, so trivial value
    # gaps swing their rank dozens of places and they crowd out real signal.
    val = view[~view["Pos"].isin(["K", "DEF"])].dropna(subset=["Sleeper ADP"]).copy()
    draftable = cfg.n_teams * (meta.rounds if meta else 16)
    val = val[(val["Sleeper ADP"] <= draftable) & (val["VORP Rank"] <= draftable)]
    val = val.sort_values("Steal", ascending=False)

    by_pos = (
        val.groupby("Pos")["Steal"].agg(["mean", "count"])
        .sort_values("mean", ascending=False).reset_index()
    )
    by_pos.columns = ["Pos", "Avg steal (picks)", "Players"]
    st.markdown("**Which positions the room misprices**")
    st.caption(
        "Positive = this whole position goes later than it's worth in your "
        "league. This is the structural edge; individual players are noise "
        "around it."
    )
    st.dataframe(
        by_pos, hide_index=True, width="stretch",
        column_config={"Avg steal (picks)": st.column_config.NumberColumn(format="%+.1f")},
    )

    st.divider()
    cols_v = ["Player", "Pos", "Tier", "VORP Rank", "Sleeper ADP", "Expert", "Steal"]
    fmt = {
        "Sleeper ADP": st.column_config.NumberColumn(format="%.1f"),
        "Expert": st.column_config.NumberColumn(format="%.0f"),
        "Steal": st.column_config.NumberColumn(format="%+.0f"),
    }
    left, right = st.columns(2)
    with left:
        st.subheader("Underpriced here")
        st.dataframe(val.head(15)[cols_v], hide_index=True, width="stretch",
                     column_config=fmt)
    with right:
        st.subheader("Overpriced here")
        st.dataframe(val.tail(15).iloc[::-1][cols_v], hide_index=True,
                     width="stretch", column_config=fmt)

with tab_room:
    st.caption(
        "Where **your Sleeper room** drafts differently from the rest of the "
        "market. Rounds are cut from the outside-market ADP (the five non-Sleeper "
        f"platforms), {cfg.n_teams} picks per round. "
        "**Negative = your room reaches** — he goes earlier here than everywhere "
        "else, so waiting loses him. **Positive = he falls** — the market says "
        "he's gone but your room lets him slide."
    )

    room = df.dropna(subset=["Room vs Market", "Outside ADP"]).copy()
    # Two outside platforms minimum: a "consensus" built from one other site
    # is not a consensus, and its deviation is noise.
    room = room[room["Platforms"] >= 3]
    room["Round"] = ((room["Outside ADP"] - 1) // cfg.n_teams + 1).astype(int)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pos_pick = st.multiselect(
            "Positions", sorted(df["Pos"].unique()),
            default=[p for p in ["QB", "RB", "WR", "TE"] if p in set(df["Pos"])],
            key="room_pos",
        )
    with c2:
        max_round = int(room["Round"].max()) if len(room) else 1
        rounds = st.slider("Rounds", 1, min(max_round, 20), (1, 8), key="room_rounds")
    with c3:
        per_round = st.slider("Players per round", 3, 15, 6, key="room_n")

    if pos_pick:
        room = room[room["Pos"].isin(pos_pick)]
    room = room[(room["Round"] >= rounds[0]) & (room["Round"] <= rounds[1])]

    if room.empty:
        st.info("No players match those filters.")
    else:
        # Biggest absolute deviation within each round.
        room["absdev"] = room["Room vs Market"].abs()
        top = (
            room.sort_values("absdev", ascending=False)
            .groupby("Round", group_keys=False)
            .head(per_round)
            .copy()
        )
        top["Label"] = top["Player"] + "  (" + top["Pos"] + ")"
        top["Direction"] = top["Room vs Market"].map(
            lambda v: "Falls to you" if v > 0 else "Room reaches"
        )

        span = float(top["absdev"].max()) * 1.15 or 1.0
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Room vs Market:Q",
                    title="Sleeper ADP − outside market (picks)",
                    scale=alt.Scale(domain=[-span, span]),
                ),
                y=alt.Y(
                    "Label:N", title=None,
                    sort=alt.EncodingSortField(
                        field="Room vs Market", order="descending"),
                    # Names run long; without this Vega truncates at ~180px.
                    axis=alt.Axis(labelLimit=220),
                ),
                color=alt.Color(
                    "Pos:N", title="Position",
                    scale=alt.Scale(
                        domain=[p for p in ["QB", "RB", "WR", "TE", "K", "DEF"]
                                if p in set(top["Pos"])],
                        range=[POS_COLORS[p]
                               for p in ["QB", "RB", "WR", "TE", "K", "DEF"]
                               if p in set(top["Pos"])],
                    ),
                ),
                tooltip=[
                    "Player", "Pos", "Round",
                    alt.Tooltip("Sleeper ADP:Q", format=".1f"),
                    alt.Tooltip("Outside ADP:Q", format=".1f", title="Outside market"),
                    alt.Tooltip("Room vs Market:Q", format="+.1f", title="Deviation"),
                    alt.Tooltip("Disagreement:Q", format=".1f", title="Platform spread"),
                    alt.Tooltip("VORP Rank:Q", title="Our VORP rank"),
                ],
            )
            .properties(height=alt.Step(20), width="container")
            .facet(
                row=alt.Row(
                    "Round:O", title=None,
                    header=alt.Header(
                        labelExpr='"Round " + datum.value',
                        labelAngle=0, labelAlign="left", labelAnchor="start",
                        labelFontWeight="bold", labelFontSize=13,
                        labelOrient="top",
                    ),
                ),
            )
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart, use_container_width=True)

        st.divider()
        st.markdown("**How unpredictable your room is, by round**")
        st.caption(
            "Mean absolute deviation. High rounds are where Sleeper ADP stops "
            "predicting the outside market — plan those picks off your own board."
        )
        summary = (
            room.groupby("Round")
            .agg(**{
                "Mean abs deviation": ("absdev", "mean"),
                "Biggest reach": ("Room vs Market", "min"),
                "Biggest faller": ("Room vs Market", "max"),
                "Players": ("absdev", "size"),
            })
            .reset_index()
        )
        st.dataframe(
            summary, hide_index=True, width="stretch",
            column_config={
                "Mean abs deviation": st.column_config.NumberColumn(format="%.1f"),
                "Biggest reach": st.column_config.NumberColumn(format="%+.0f"),
                "Biggest faller": st.column_config.NumberColumn(format="%+.0f"),
            },
        )

st.divider()
st.caption(
    "Points: rotowire (via Sleeper), scoring from this league's settings. "
    "ADP + expert ranks: Flock Fantasy (6 platforms + 10-analyst consensus). "
    "Survival treats draft position as Normal(ADP, cross-platform SD) — both "
    "measured, not assumed. Runs correlate in real drafts, so treat "
    "probabilities as well-calibrated guidance, not guarantees."
)
