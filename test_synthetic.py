"""Sanity-check the engine on synthetic full-PPR-shaped projections."""

from vorp import LeagueConfig, Player, assign_vorp, assign_tiers, solve_replacement_levels

# Rough full-PPR season-point curves: top-end value and decay rate per position.
CURVES = {
    "QB": (390, 0.0075, 40),
    "RB": (330, 0.0210, 80),
    "WR": (335, 0.0155, 100),
    "TE": (250, 0.0290, 30),
}


def synth() -> list[Player]:
    players = []
    for pos, (top, decay, count) in CURVES.items():
        for i in range(count):
            pts = top * pow(2.718281828, -decay * i)
            players.append(Player(name=f"{pos}{i+1}", pos=pos, proj=round(pts, 1)))
    return players


league = LeagueConfig(
    n_teams=12,
    roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                      "K", "DEF"] + ["BN"] * 6,
)

players = synth()
baselines, starters = solve_replacement_levels(players, league)

print("league:", league.n_teams, "teams | dedicated:", league.dedicated,
      "| flex:", league.flex_slots)
print()
print(f"{'pos':<5}{'naive starters':>16}{'effective':>12}{'baseline pts':>15}")
for pos in ["QB", "RB", "WR", "TE"]:
    naive = league.n_teams * league.dedicated.get(pos, 0)
    print(f"{pos:<5}{naive:>16}{starters[pos]:>12}{baselines[pos]:>15.1f}")

assign_vorp(players, league)
for pos in ["QB", "RB", "WR", "TE"]:
    assign_tiers(players, pos, width_frac=0.08, cliff_z=2.0, max_rank=48)

print("\ntop 15 overall by VORP:")
top = sorted(players, key=lambda p: p.vorp, reverse=True)[:15]
for r, p in enumerate(top, 1):
    print(f"{r:>3}. {p.name:<6} {p.pos:<3} proj {p.proj:>6.1f}  "
          f"vorp {p.vorp:>6.1f}  tier {p.tier}")

print("\nWR tier structure:")
wrs = sorted([p for p in players if p.pos == "WR" and p.tier],
             key=lambda p: p.vorp, reverse=True)
cur = None
for p in wrs[:30]:
    if p.tier != cur:
        cur = p.tier
        print(f"  -- tier {cur} --")
    print(f"     {p.name:<6} vorp {p.vorp:>6.1f}")
