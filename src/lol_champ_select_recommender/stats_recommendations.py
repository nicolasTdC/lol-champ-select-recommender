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
    index = load_player_prune_index(args.player_stats, profiles=args.profiles, ranked_queue=args.ranked_queue)
    if index is None or not index.overall_by_champion:
        print(f"Error: no player stats found for the selected profiles/queue ({args.ranked_queue}) in {args.player_stats}. Refresh the CSV with the requested queues.", file=sys.stderr)
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
        f"Ranked queue: {args.ranked_queue}",
        f"Soft: {MIN_GAMES}+ games and {MIN_WIN_RATE:.0%}+ WR across all roles",
        "Hard: Soft plus the same threshold in the listed role",
        f"Extrapolated: also allow <{MIN_GAMES} games with losses <{MAX_LOSSES_FOR_LOW_SAMPLE:g}; includes zero games",
        "Whitelisted: excludes global and role-specific blacklist entries",
        "Not Recommended: exact complement of each filter, including insufficient samples; whitelisted complements also include blacklist exclusions.",
        "Ranking: games descending, then WR descending. WR is historical, not a model score.",
        "", "Lane",
    ]
    for label, rows in (("Hard", index.hard_lane_recommendations()), ("Soft", index.soft_lane_recommendations())):
        lines.append(f"  {label}: " + ("; ".join(f"{ROLE_NAMES[role]} ({stats_text(stats)})" for role, stats in rows) or "-"))

    recommendation_start = len(lines)
    not_recommended = ["", "Not Recommended"]
    for role in POSITION_ORDER:
        lines.extend(["", ROLE_NAMES[role]])
        not_recommended.extend(["", ROLE_NAMES[role]])
        groups = (
            ("Soft", index.passes_soft, False),
            ("Hard", lambda cid: index.passes_soft(cid) and index.passes_hard(cid, role), True),
            ("Extrapolated Soft", index.passes_soft_extrapolated, False),
            ("Extrapolated Hard", lambda cid: index.passes_soft_extrapolated(cid) and index.passes_hard_extrapolated(cid, role), True),
        )
        for whitelisted in (False, True):
            for label, passes, role_specific in groups:
                stats_for = index.role_stats if role_specific else lambda cid, _: index.overall_stats(cid)
                kept = {cid for cid in candidates if passes(cid) and (not whitelisted or not blacklist.blocks(cid, role))}
                for destination, champion_ids in ((lines, kept), (not_recommended, candidates - kept)):
                    ranked = sorted(champion_ids, key=lambda cid: (
                        -(stats_for(cid, role) or zero).games,
                        -(stats_for(cid, role) or zero).win_rate,
                        features.get(cid, {}).get("champion_name", str(cid)),
                    ))
                    destination.append(f"  {'Whitelisted ' if whitelisted else ''}{label}:")
                    if not ranked:
                        destination.append("    -")
                    for cid in ranked[:args.recommendation_count]:
                        name = features.get(cid, {}).get("champion_name", str(cid))
                        destination.append(f"    {name}: {stats_text(stats_for(cid, role) or zero)}")
    lines.insert(recommendation_start, "\nRecommended")
    lines.extend(not_recommended)
    lines.extend(ban_recommendation_lines(args, features))
    print("\n".join(lines))
    return 0


def ban_recommendation_lines(args, features) -> list[str]:
    index = load_player_prune_index(args.opponent_stats, profiles=args.profiles, ranked_queue=args.ranked_queue)
    lines = ["", "Ban Recommendations"]
    if index is None or not index.overall_by_champion:
        return lines + [f"  No enemy matchup stats for these profiles/queue. Refresh with collect_player_stats.py ({args.opponent_stats})."]
    lines.extend([
        "  WR is your profiles' win rate against the enemy champion; roles refer to the enemy's lane.",
        "  Soft: 20+ encounters with personal WR below 52% across all roles; Hard: the same rule using only the listed enemy role.",
        "  Extrapolated: also flag <20 encounters with losses >=10. Unseen champions are not ban candidates.",
        "  Ranked by encounters descending, then personal WR ascending; pick blacklists do not filter bans.",
    ])

    def poor(stats, extrapolated):
        if stats is None:
            return False
        if stats.games >= MIN_GAMES:
            return stats.win_rate < MIN_WIN_RATE
        return extrapolated and stats.losses >= MAX_LOSSES_FOR_LOW_SAMPLE

    for role in (None, *POSITION_ORDER):
        lines.extend(["", f"  {ROLE_NAMES[role] if role else 'All Roles'}"])
        stats_by_champion = index.by_role_by_champion.get(role, {}) if role else index.overall_by_champion
        base_label = "Hard" if role else "Soft"
        for label, extrapolated in ((base_label, False), (f"Extrapolated {base_label}", True)):
            kept = [cid for cid, stats in stats_by_champion.items() if poor(stats, extrapolated)]
            kept.sort(key=lambda cid: (-stats_by_champion[cid].games, stats_by_champion[cid].win_rate, cid))
            lines.append(f"    {label}:")
            if not kept:
                lines.append("      -")
            for cid in kept[:args.recommendation_count]:
                stats = stats_by_champion[cid]
                name = features.get(cid, {}).get("champion_name", str(cid))
                scope = "enemy role" if role else "all roles"
                lines.append(f"      {name}: {stats.games} games, {stats.wins}W/{stats.losses}L, personal WR {stats.win_rate:.1%} ({scope})")
    return lines
