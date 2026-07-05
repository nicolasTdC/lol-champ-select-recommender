from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from .env import riot_api_key
from .riot_api import RiotApiClient, RiotApiError, region_for_platform


DEFAULT_DIVISIONS = ("I", "II", "III", "IV")
DEFAULT_TIERS = ("EMERALD",)
APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}


def main() -> int:
    args = parse_args()

    try:
        platform = args.platform.lower()
        region = args.region or region_for_platform(platform)
        client = RiotApiClient(api_key=riot_api_key())
        entries = collect_ladder_entries(client, args)
    except (RuntimeError, RiotApiError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    players = select_seed_players(entries, args.max_players, args.seed)
    if not players:
        print("No ladder entries found for the requested filters.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    matches_dir = output_dir / "matches"
    manifests_dir = output_dir / "manifests"
    match_sources_path = output_dir / "match_sources.jsonl"
    matches_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Platform:       {platform}")
    print(f"Region:         {region}")
    print(f"Queue:          {args.queue}")
    print(f"Match queue:    {args.match_queue}")
    print(f"Ladder entries: {len(entries)}")
    print(f"Seed players:   {len(players)}")
    print(f"Matches/player: {args.matches_per_player}")
    print(f"Output:         {matches_dir}")

    seen_match_ids = existing_match_ids(matches_dir)
    discovered_match_ids: list[str] = []
    downloaded = 0
    skipped_existing = 0
    skipped_duplicate = 0
    player_errors = 0
    match_errors = 0

    for player_index, entry in enumerate(players, start=1):
        puuid = entry.get("puuid")
        if not puuid:
            puuid = resolve_puuid(client, platform, entry)
        if not puuid:
            player_errors += 1
            print(f"[player {player_index:>3}/{len(players)}] missing PUUID; skipped")
            continue

        try:
            match_ids = client.match_ids_by_puuid(
                str(puuid),
                region,
                count=args.matches_per_player,
                queue=args.match_queue,
                match_type=args.match_type,
            )
        except RiotApiError as exc:
            player_errors += 1
            print(f"[player {player_index:>3}/{len(players)}] match IDs error: {exc}", file=sys.stderr)
            sleep(args.sleep)
            continue

        new_for_player = 0
        for match_id in match_ids:
            append_match_source(
                match_sources_path,
                match_id=match_id,
                platform=platform,
                region=region,
                queue=args.queue,
                match_queue=args.match_queue,
                match_type=args.match_type,
                entry=entry,
            )

            if match_id in discovered_match_ids:
                skipped_duplicate += 1
                continue
            discovered_match_ids.append(match_id)

            match_path = matches_dir / f"{match_id}.json"
            if match_id in seen_match_ids and not args.force:
                skipped_existing += 1
                continue

            try:
                match = client.match_by_id(match_id, region)
            except RiotApiError as exc:
                match_errors += 1
                print(f"  error {match_id}: {exc}", file=sys.stderr)
                sleep(args.sleep)
                continue

            write_json(match_path, match)
            seen_match_ids.add(match_id)
            downloaded += 1
            new_for_player += 1
            sleep(args.sleep)

        print(
            f"[player {player_index:>3}/{len(players)}] "
            f"{entry_label(entry)} -> {len(match_ids)} ids, {new_for_player} downloaded"
        )
        sleep(args.sleep)

    manifest = {
        "platform": platform,
        "region": region,
        "queue": args.queue,
        "match_queue": args.match_queue,
        "match_type": args.match_type,
        "tiers": args.tiers,
        "divisions": args.divisions,
        "pages": args.pages,
        "max_players": args.max_players,
        "matches_per_player": args.matches_per_player,
        "seed": args.seed,
        "ladder_entries": len(entries),
        "seed_players": len(players),
        "unique_match_ids_discovered": len(discovered_match_ids),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "skipped_duplicate": skipped_duplicate,
        "player_errors": player_errors,
        "match_errors": match_errors,
        "match_sources": str(match_sources_path),
    }
    manifest_path = manifests_dir / f"ranked-{platform}-{int(time.time())}.json"
    write_json(manifest_path, manifest)

    print("")
    print(f"Unique match IDs: {len(discovered_match_ids)}")
    print(f"Downloaded:       {downloaded}")
    print(f"Existing skipped: {skipped_existing}")
    print(f"Duplicate skipped:{skipped_duplicate}")
    print(f"Player errors:    {player_errors}")
    print(f"Match errors:     {match_errors}")
    print(f"Manifest:         {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect server-level ranked Match-V5 data from League-V4 ladder entries."
    )
    parser.add_argument(
        "--platform",
        default="br1",
        help="Riot platform routing value. Examples: br1, na1, euw1, kr. Default: br1",
    )
    parser.add_argument(
        "--region",
        help="Riot regional routing value for Match-V5. Defaults from --platform.",
    )
    parser.add_argument(
        "--queue",
        default="RANKED_SOLO_5x5",
        help="League-V4 queue. Default: RANKED_SOLO_5x5",
    )
    parser.add_argument(
        "--match-queue",
        type=int,
        default=420,
        help="Match-V5 queue ID. Default: 420 for ranked solo/duo.",
    )
    parser.add_argument(
        "--match-type",
        default="ranked",
        choices=["ranked", "normal", "tourney", "tutorial"],
        help="Match-V5 match type filter. Default: ranked",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(DEFAULT_TIERS),
        help="League tiers to sample. Supports MASTER, GRANDMASTER, CHALLENGER. Default: EMERALD",
    )
    parser.add_argument(
        "--divisions",
        nargs="+",
        default=list(DEFAULT_DIVISIONS),
        help="Divisions for DIAMOND and below. Ignored for MASTER+. Default: I II III IV",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="League-V4 pages per tier/division for DIAMOND and below. Ignored for MASTER+. Default: 1",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=25,
        help="Maximum ladder players to seed from. Default: 25",
    )
    parser.add_argument(
        "--matches-per-player",
        type=int,
        default=5,
        choices=range(1, 101),
        metavar="[1-100]",
        help="Recent matches to request per seed player. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for selecting ladder players. Default: 1",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between Riot requests. Default: 0.05",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for raw downloaded JSON. Default: data/raw",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download matches even if local JSON already exists.",
    )
    return parser.parse_args()


def collect_ladder_entries(client: RiotApiClient, args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_puuids_or_summoner_ids: set[str] = set()

    for tier in args.tiers:
        tier = tier.upper()
        if tier in APEX_TIERS:
            try:
                apex_entries = client.apex_league_entries(args.platform, tier=tier, queue=args.queue)
            except RiotApiError as exc:
                print(f"Warning: {tier} failed: {exc}", file=sys.stderr)
                continue

            print(f"Loaded {len(apex_entries)} entries from {tier}")
            entries.extend(_unique_entries(apex_entries, seen_puuids_or_summoner_ids))
            continue

        for division in args.divisions:
            division = division.upper()
            for page in range(1, args.pages + 1):
                entries.extend(
                    _collect_standard_ladder_page(
                        client,
                        args,
                        tier=tier,
                        division=division,
                        page=page,
                        seen_puuids_or_summoner_ids=seen_puuids_or_summoner_ids,
                    )
                )

    return entries


def _collect_standard_ladder_page(
    client: RiotApiClient,
    args: argparse.Namespace,
    *,
    tier: str,
    division: str,
    page: int,
    seen_puuids_or_summoner_ids: set[str],
) -> list[dict[str, Any]]:
    try:
        page_entries = client.league_entries(
            args.platform,
            queue=args.queue,
            tier=tier,
            division=division,
            page=page,
        )
    except RiotApiError as exc:
        print(f"Warning: {tier} {division} page {page} failed: {exc}", file=sys.stderr)
        return []

    print(f"Loaded {len(page_entries)} entries from {tier} {division} page {page}")
    return _unique_entries(page_entries, seen_puuids_or_summoner_ids)


def _unique_entries(entries: list[dict[str, Any]], seen_puuids_or_summoner_ids: set[str]) -> list[dict[str, Any]]:
    unique_entries: list[dict[str, Any]] = []
    for entry in entries:
        identity = entry.get("puuid") or entry.get("summonerId")
        if not identity or identity in seen_puuids_or_summoner_ids:
            continue
        seen_puuids_or_summoner_ids.add(str(identity))
        unique_entries.append(entry)
    return unique_entries


def select_seed_players(entries: list[dict[str, Any]], max_players: int, seed: int) -> list[dict[str, Any]]:
    candidates = list(entries)
    random.Random(seed).shuffle(candidates)
    return candidates[: max(0, max_players)]


def resolve_puuid(client: RiotApiClient, platform: str, entry: dict[str, Any]) -> str | None:
    summoner_id = entry.get("summonerId")
    if not summoner_id:
        return None
    try:
        summoner = client.summoner_by_id(platform, str(summoner_id))
    except RiotApiError as exc:
        print(f"  summoner lookup failed for {entry_label(entry)}: {exc}", file=sys.stderr)
        return None
    puuid = summoner.get("puuid")
    return str(puuid) if puuid else None


def existing_match_ids(matches_dir: Path) -> set[str]:
    return {path.stem for path in matches_dir.glob("*.json")}


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_match_source(
    path: Path,
    *,
    match_id: str,
    platform: str,
    region: str,
    queue: str,
    match_queue: int,
    match_type: str,
    entry: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tier = str(entry.get("tier") or "UNKNOWN").upper()
    division = None if tier in APEX_TIERS else str(entry.get("rank") or "").upper() or None
    payload = {
        "match_id": match_id,
        "platform": platform,
        "region": region,
        "queue": queue,
        "match_queue": match_queue,
        "match_type": match_type,
        "seed_rank": {
            "tier": tier,
            "division": division,
            "rank_bucket": rank_bucket(tier, division),
            "league_points": entry.get("leaguePoints"),
            "wins": entry.get("wins"),
            "losses": entry.get("losses"),
        },
        "collected_at": int(time.time()),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def rank_bucket(tier: str | None, division: str | None) -> str:
    tier = str(tier or "UNKNOWN").upper()
    if tier in APEX_TIERS:
        return tier
    division = str(division or "").upper()
    return f"{tier}_{division}" if division else tier


def entry_label(entry: dict[str, Any]) -> str:
    tier = entry.get("tier", "?")
    rank = entry.get("rank", "?")
    lp = entry.get("leaguePoints", "?")
    wins = entry.get("wins", "?")
    losses = entry.get("losses", "?")
    return f"{tier} {rank} {lp}LP {wins}W/{losses}L"


def sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
