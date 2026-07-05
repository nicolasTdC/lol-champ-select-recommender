from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .env import riot_api_key
from .riot_api import RiotApiClient, RiotApiError, parse_riot_id


def main() -> int:
    args = parse_args()

    try:
        game_name, tag_line = _riot_id_parts(args)
        client = RiotApiClient(api_key=riot_api_key())
        account = client.account_by_riot_id(game_name, tag_line, args.region)
        puuid = str(account["puuid"])
        match_ids = client.match_ids_by_puuid(
            puuid,
            args.region,
            count=args.count,
            queue=args.queue,
            match_type=args.match_type,
        )
    except (KeyError, RuntimeError, RiotApiError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    matches_dir = output_dir / "matches"
    accounts_dir = output_dir / "accounts"
    matches_dir.mkdir(parents=True, exist_ok=True)
    accounts_dir.mkdir(parents=True, exist_ok=True)

    account_path = accounts_dir / f"{_safe_filename(game_name)}-{_safe_filename(tag_line)}.json"
    _write_json(account_path, account)

    print(f"Account: {account.get('gameName', game_name)}#{account.get('tagLine', tag_line)}")
    print(f"PUUID:   {puuid}")
    print(f"Matches: {len(match_ids)}")
    print(f"Output:  {matches_dir}")

    downloaded = 0
    skipped = 0

    for index, match_id in enumerate(match_ids, start=1):
        match_path = matches_dir / f"{match_id}.json"
        if match_path.exists() and not args.force:
            skipped += 1
            print(f"[{index:>3}/{len(match_ids)}] skip {match_id}")
            continue

        try:
            match = client.match_by_id(match_id, args.region)
        except RiotApiError as exc:
            print(f"[{index:>3}/{len(match_ids)}] error {match_id}: {exc}", file=sys.stderr)
            continue

        _write_json(match_path, match)
        downloaded += 1
        print(f"[{index:>3}/{len(match_ids)}] saved {match_id}")

    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch recent Riot Match-V5 matches for a Riot ID.")
    parser.add_argument(
        "--riot-id",
        help='Riot ID in the form "GameName#TAG".',
    )
    parser.add_argument("--game-name", help="Riot ID game name. Use with --tag-line.")
    parser.add_argument("--tag-line", help="Riot ID tag line. Use with --game-name.")
    parser.add_argument(
        "--region",
        default="americas",
        choices=["americas", "asia", "europe", "sea"],
        help="Riot regional routing value for Account-V1 and Match-V5. Default: americas",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        choices=range(1, 101),
        metavar="[1-100]",
        help="Number of recent match IDs to fetch. Default: 20",
    )
    parser.add_argument(
        "--queue",
        type=int,
        help="Optional queue filter. Example: 420 for ranked solo/duo, 440 for ranked flex.",
    )
    parser.add_argument(
        "--match-type",
        choices=["ranked", "normal", "tourney", "tutorial"],
        help="Optional Riot match type filter.",
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


def _riot_id_parts(args: argparse.Namespace) -> tuple[str, str]:
    if args.riot_id:
        return parse_riot_id(args.riot_id)
    if args.game_name and args.tag_line:
        return args.game_name, args.tag_line
    raise ValueError('Pass --riot-id "GameName#TAG" or both --game-name and --tag-line.')


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
