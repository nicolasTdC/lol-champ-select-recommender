from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregate_matches import QUEUE_SR, _as_int, _participant_roles, _patch
from .ddragon import StaticData, load_static_data
from .riot_api import RiotApiError, region_for_platform
from .roles import POSITION_ORDER


TEAM_TO_SIDE = {100: "blue", 200: "red"}
SIDE_TO_TEAM = {"blue": 100, "red": 200}
UNKNOWN_RANK = {
    "tier": "UNKNOWN",
    "division": None,
    "rank_bucket": "UNKNOWN",
}
TIER_STRENGTH = {
    "UNKNOWN": 0,
    "IRON": 1,
    "BRONZE": 2,
    "SILVER": 3,
    "GOLD": 4,
    "PLATINUM": 5,
    "EMERALD": 6,
    "DIAMOND": 7,
    "MASTER": 8,
    "GRANDMASTER": 9,
    "CHALLENGER": 10,
}
DIVISION_STRENGTH = {
    None: 0,
    "": 0,
    "IV": 1,
    "III": 2,
    "II": 3,
    "I": 4,
}


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    sources_path = Path(args.match_sources)

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    static_data = load_static_data(args.language)
    match_sources = load_match_sources(sources_path)
    rows = build_draft_dataset_rows(
        input_dir,
        static_data,
        match_sources,
        include_non_sr=args.include_non_sr,
        allow_incomplete=args.allow_incomplete,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} draft rows to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw normalized draft JSONL from downloaded Match-V5 files.")
    parser.add_argument(
        "--input-dir",
        default="data/raw/matches",
        help="Directory containing raw Match-V5 JSON files. Default: data/raw/matches",
    )
    parser.add_argument(
        "--match-sources",
        default="data/raw/match_sources.jsonl",
        help="Optional match source/rank sidecar. Default: data/raw/match_sources.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/processed/draft_dataset.jsonl",
        help="Output JSONL path. Default: data/processed/draft_dataset.jsonl",
    )
    parser.add_argument(
        "--language",
        default="en_US",
        help="Data Dragon language for champion names. Default: en_US",
    )
    parser.add_argument(
        "--include-non-sr",
        action="store_true",
        help="Include queues outside standard Summoner's Rift queues.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit rows even if one or both teams are missing normalized roles.",
    )
    return parser.parse_args()


def build_draft_dataset_rows(
    input_dir: Path,
    static_data: StaticData,
    match_sources: dict[str, list[dict[str, Any]]] | None = None,
    *,
    include_non_sr: bool = False,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    match_sources = match_sources or {}

    for path in sorted(input_dir.glob("*.json")):
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping unreadable match file {path}: {exc}", file=sys.stderr)
            continue

        row = normalized_draft_row(
            match,
            static_data,
            source_records=match_sources.get(path.stem, []),
            fallback_match_id=path.stem,
            include_non_sr=include_non_sr,
            allow_incomplete=allow_incomplete,
        )
        if row:
            rows.append(row)

    return rows


def normalized_draft_row(
    match: dict[str, Any],
    static_data: StaticData,
    *,
    source_records: list[dict[str, Any]] | None = None,
    fallback_match_id: str = "",
    include_non_sr: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any] | None:
    metadata = match.get("metadata", {})
    info = match.get("info", {})
    if not isinstance(info, dict):
        return None

    queue_id = _as_int(info.get("queueId")) or 0
    if not include_non_sr and queue_id not in QUEUE_SR:
        return None

    match_id = str(metadata.get("matchId") or fallback_match_id)
    platform = platform_from_match_id(match_id)
    region = safe_region_for_platform(platform)
    teams = [team for team in info.get("teams", []) if isinstance(team, dict)]
    participants = [participant for participant in info.get("participants", []) if isinstance(participant, dict)]

    winning_team_id = winning_team(teams, participants)
    if winning_team_id not in TEAM_TO_SIDE:
        return None

    participant_roles = _participant_roles(participants, static_data)
    side_drafts = side_role_champions(participants, participant_roles)
    if not allow_incomplete and not complete_drafts(side_drafts):
        return None

    blue_bans = bans_for_team(teams, SIDE_TO_TEAM["blue"])
    red_bans = bans_for_team(teams, SIDE_TO_TEAM["red"])
    seed_ranks = seed_ranks_from_sources(source_records or [])
    chosen_rank = choose_rank(seed_ranks)

    return {
        "schema_version": 1,
        "match_id": match_id,
        "platform": platform,
        "region": region,
        "patch": _patch(info.get("gameVersion")),
        "game_version": info.get("gameVersion"),
        "queue_id": queue_id,
        "game_creation": info.get("gameCreation"),
        "game_duration": info.get("gameDuration"),
        "winning_team_id": winning_team_id,
        "winning_side": TEAM_TO_SIDE[winning_team_id],
        "blue_win": winning_team_id == SIDE_TO_TEAM["blue"],
        "rank_tier": chosen_rank["tier"],
        "rank_division": chosen_rank["division"],
        "rank_bucket": chosen_rank["rank_bucket"],
        "seed_ranks": seed_ranks,
        "blue": side_drafts["blue"],
        "red": side_drafts["red"],
        "blue_names": champion_names(side_drafts["blue"], static_data),
        "red_names": champion_names(side_drafts["red"], static_data),
        "blue_bans": blue_bans,
        "red_bans": red_bans,
        "blue_ban_names": [static_data.champion_name(champion_id) for champion_id in blue_bans],
        "red_ban_names": [static_data.champion_name(champion_id) for champion_id in red_bans],
    }


def load_match_sources(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}

    sources: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        match_id = str(record.get("match_id") or "")
        if not match_id:
            continue
        sources.setdefault(match_id, []).append(record)
    return sources


def platform_from_match_id(match_id: str) -> str:
    if "_" not in match_id:
        return ""
    return match_id.split("_", 1)[0].lower()


def safe_region_for_platform(platform: str) -> str:
    if not platform:
        return ""
    try:
        return region_for_platform(platform)
    except RiotApiError:
        return ""


def winning_team(teams: list[dict[str, Any]], participants: list[dict[str, Any]]) -> int | None:
    for team in teams:
        if bool(team.get("win")):
            return _as_int(team.get("teamId"))

    for participant in participants:
        if bool(participant.get("win")):
            return _as_int(participant.get("teamId"))

    return None


def side_role_champions(
    participants: list[dict[str, Any]],
    participant_roles: dict[int, str],
) -> dict[str, dict[str, int | None]]:
    drafts: dict[str, dict[str, int | None]] = {
        "blue": {position: None for position in POSITION_ORDER},
        "red": {position: None for position in POSITION_ORDER},
    }

    for participant in participants:
        participant_id = _as_int(participant.get("participantId"))
        team_id = _as_int(participant.get("teamId"))
        champion_id = _as_int(participant.get("championId"))
        if participant_id is None or team_id is None or not champion_id:
            continue
        side = TEAM_TO_SIDE.get(team_id)
        role = participant_roles.get(participant_id)
        if not side or role not in POSITION_ORDER:
            continue
        drafts[side][role] = champion_id

    return drafts


def complete_drafts(side_drafts: dict[str, dict[str, int | None]]) -> bool:
    return all(side_drafts[side].get(position) for side in ("blue", "red") for position in POSITION_ORDER)


def champion_names(draft: dict[str, int | None], static_data: StaticData) -> dict[str, str]:
    return {role: static_data.champion_name(champion_id) for role, champion_id in draft.items()}


def bans_for_team(teams: list[dict[str, Any]], team_id: int) -> list[int]:
    for team in teams:
        if _as_int(team.get("teamId")) != team_id:
            continue
        bans = [ban for ban in team.get("bans", []) if isinstance(ban, dict)]
        bans.sort(key=lambda ban: _as_int(ban.get("pickTurn")) or 0)
        return [_as_int(ban.get("championId")) or -1 for ban in bans]
    return []


def seed_ranks_from_sources(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str]] = set()

    for record in source_records:
        seed_rank = record.get("seed_rank", {})
        if not isinstance(seed_rank, dict):
            continue
        tier = str(seed_rank.get("tier") or "UNKNOWN").upper()
        division = seed_rank.get("division")
        division = str(division).upper() if division else None
        rank_bucket = str(seed_rank.get("rank_bucket") or rank_bucket_from_parts(tier, division))
        key = (tier, division, rank_bucket)
        if key in seen:
            continue
        seen.add(key)
        ranks.append(
            {
                "tier": tier,
                "division": division,
                "rank_bucket": rank_bucket,
                "league_points": seed_rank.get("league_points"),
                "wins": seed_rank.get("wins"),
                "losses": seed_rank.get("losses"),
            }
        )

    return ranks


def choose_rank(seed_ranks: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_ranks:
        return dict(UNKNOWN_RANK)
    return dict(max(seed_ranks, key=rank_sort_key))


def rank_sort_key(rank: dict[str, Any]) -> tuple[int, int, int]:
    tier = str(rank.get("tier") or "UNKNOWN").upper()
    division = rank.get("division")
    division = str(division).upper() if division else None
    lp = _as_int(rank.get("league_points")) or 0
    return (TIER_STRENGTH.get(tier, 0), DIVISION_STRENGTH.get(division, 0), lp)


def rank_bucket_from_parts(tier: str, division: str | None) -> str:
    if tier in {"MASTER", "GRANDMASTER", "CHALLENGER"}:
        return tier
    return f"{tier}_{division}" if division else tier
