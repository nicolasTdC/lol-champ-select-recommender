from __future__ import annotations

import csv
import sys
from pathlib import Path

from .modeling.draft_inference import load_champion_blacklist
from .modeling.player_pruning import (
    MAX_LOSSES_FOR_LOW_SAMPLE,
    MIN_GAMES,
    MIN_WIN_RATE,
    PruneStats,
    load_player_prune_index,
)
from .roles import POSITION_ORDER, ROLE_NAMES


def run_stats_only(args) -> int:
    if args.recommendation_count < 1:
        print("Error: --recommendation-count must be positive.", file=sys.stderr)
        return 1
    index = load_player_prune_index(args.player_stats, profiles=args.profiles)
    if index is None or not index.overall_by_champion:
        print(f"Error: no player stats found for the selected profiles in {args.player_stats}.", file=sys.stderr)
        return 1

    features = {}
    feature_path = Path(args.champion_features)
    if feature_path.is_file():
        with feature_path.open(encoding="utf-8", newline="") as file:
            features = {int(row["champion_id"]): row for row in csv.DictReader(file)}
    with Path(args.player_stats).open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            champion_id = int(row["champion_id"])
            features.setdefault(champion_id, {"champion_name": row.get("champion_name") or str(champion_id)})
    blacklist = load_champion_blacklist(args.champion_blacklist, features)
    candidates = set(features) | set(index.overall_by_champion)
    zero = PruneStats(0, 0, 0)

    def stats_text(stats):
        rate = f"{stats.win_rate:.1%}" if stats.games else "n/a"
        return f"{stats.games} games, {stats.wins}W/{stats.losses}L, WR {rate}"

    lines = [
        "Player Stats Recommendations",
        f"Profiles: {', '.join(args.profiles) if args.profiles else 'all profiles in CSV'} (combined)",
        f"Source: {args.player_stats}",
        f"Soft: {MIN_GAMES}+ games and {MIN_WIN_RATE:.0%}+ WR across all roles",
        "Hard: Soft plus the same threshold in the listed role",
        f"Extrapolated: also allow <{MIN_GAMES} games with losses <{MAX_LOSSES_FOR_LOW_SAMPLE:g}; includes zero games",
        "Whitelisted: excludes global and role-specific blacklist entries",
        "Ranking: WR descending, then games descending; zero-game champions last. WR is historical, not a model score.",
        "", "Lane",
    ]
    for label, rows in (("Hard", index.hard_lane_recommendations()), ("Soft", index.soft_lane_recommendations())):
        lines.append(f"  {label}: " + ("; ".join(f"{ROLE_NAMES[role]} ({stats_text(stats)})" for role, stats in rows) or "-"))

    for role in POSITION_ORDER:
        lines.extend(["", ROLE_NAMES[role]])
        groups = (
            ("Soft", index.passes_soft, False),
            ("Hard", lambda cid: index.passes_soft(cid) and index.passes_hard(cid, role), True),
            ("Extrapolated Soft", index.passes_soft_extrapolated, False),
            ("Extrapolated Hard", lambda cid: index.passes_soft_extrapolated(cid) and index.passes_hard_extrapolated(cid, role), True),
        )
        for whitelisted in (False, True):
            for label, passes, role_specific in groups:
                stats_for = index.role_stats if role_specific else lambda cid, _: index.overall_stats(cid)
                kept = [cid for cid in candidates if passes(cid) and (not whitelisted or not blacklist.blocks(cid, role))]
                kept.sort(key=lambda cid: (
                    -(stats_for(cid, role) or zero).win_rate,
                    -(stats_for(cid, role) or zero).games,
                    features.get(cid, {}).get("champion_name", str(cid)),
                ))
                lines.append(f"  {'Whitelisted ' if whitelisted else ''}{label}:")
                if not kept:
                    lines.append("    -")
                for cid in kept[:args.recommendation_count]:
                    name = features.get(cid, {}).get("champion_name", str(cid))
                    lines.append(f"    {name}: {stats_text(stats_for(cid, role) or zero)}")
    print("\n".join(lines))
    return 0
