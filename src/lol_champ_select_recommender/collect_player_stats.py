from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

from .aggregate_matches import _participant_roles, _as_int
from .collect_ranked_matches import download_match, resolve_download_workers
from .ddragon import load_static_data
from .env import riot_api_key
from .riot_api import RiotApiClient, RiotApiError, parse_riot_id
from .roles import ROLE_NAMES


def main() -> int:
    args = parse_args()

    if not args.riot_id:
        print("Error: pass at least one --riot-id", file=sys.stderr)
        return 1

    try:
        resolve_download_workers(args.download_workers, 1)
        client = RiotApiClient(api_key=riot_api_key(), request_rate_limit=args.request_rate_limit, log_rate_limits=True)
        static_data = load_static_data(args.language)
        rows = collect_player_stats(
            client,
            static_data,
            riot_ids=args.riot_id,
            region=args.region,
            queue=args.queue,
            match_type=args.match_type,
            matches_per_player=args.matches_per_player,
            sleep_seconds=args.sleep,
            download_workers=args.download_workers,
            matches_dir=Path(args.matches_dir),
        )
    except (KeyError, RuntimeError, RiotApiError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "player",
                "riot_id",
                "champion_id",
                "champion_name",
                "role",
                "role_name",
                "queue_id",
                "games",
                "wins",
                "losses",
                "win_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} player-champion-role rows to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-player champion-role stats from Riot Match-V5 data.")
    parser.add_argument(
        "--riot-id",
        action="append",
        required=True,
        help='Riot ID in the form "GameName#TAG". Repeat for multiple players.',
    )
    parser.add_argument(
        "--region",
        default="americas",
        choices=["americas", "asia", "europe", "sea"],
        help="Riot regional routing value. Default: americas",
    )
    parser.add_argument(
        "--queue",
        type=int,
        nargs="+",
        choices=[420, 440],
        default=[420, 440],
        help="Ranked queues to sample: 420 Solo/Duo, 440 Flex. Default: both.",
    )
    parser.add_argument(
        "--match-type",
        default="ranked",
        choices=["ranked", "normal", "tourney", "tutorial"],
        help="Match type filter. Default: ranked",
    )
    parser.add_argument(
        "--matches-per-player",
        type=positive_int,
        default=100,
        metavar="N",
        help="Recent matches per player per queue, paginated in batches of 100. Default: 100",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0,
        help="Extra delay between requests per worker. Default: 0 (shared limiter controls pacing)",
    )
    parser.add_argument("--download-workers", default="auto", help="Concurrent downloads: auto or a positive integer. Default: auto (up to 8)")
    parser.add_argument("--request-rate-limit", type=float, default=5.0, help="Shared requests/sec limit; 0 disables pacing. Default: 5.0")
    parser.add_argument("--matches-dir", default="data/raw/matches", help="Shared raw match cache. Default: data/raw/matches")
    parser.add_argument(
        "--language",
        default="en_US",
        help="Data Dragon language for champion names. Default: en_US",
    )
    parser.add_argument(
        "--output",
        default="data/processed/player_champion_role_stats.csv",
        help="Output CSV path. Default: data/processed/player_champion_role_stats.csv",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def collect_player_stats(
    client: RiotApiClient,
    static_data: Any,
    *,
    riot_ids: list[str],
    region: str,
    queue: int | list[int],
    match_type: str | None,
    matches_per_player: int,
    sleep_seconds: float,
    download_workers: str = "1",
    matches_dir: Path | None = None,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    if matches_dir is not None:
        matches_dir.mkdir(parents=True, exist_ok=True)

    for index, riot_id in enumerate(riot_ids, start=1):
        game_name, tag_line = parse_riot_id(riot_id)
        account = client.account_by_riot_id(game_name, tag_line, region)
        puuid = str(account["puuid"])
        time.sleep(sleep_seconds)
        match_ids = []
        for queue_id in dict.fromkeys([queue] if isinstance(queue, int) else queue):
            ids = []
            for start in range(0, matches_per_player, 100):
                count = min(100, matches_per_player - start)
                page = client.match_ids_by_puuid(
                    puuid,
                    region,
                    start=start,
                    count=count,
                    queue=queue_id,
                    match_type=match_type,
                )
                ids.extend(page)
                time.sleep(sleep_seconds)
                if len(page) < count:
                    break
            match_ids.extend(ids)
            print(f"  queue {queue_id}: {len(ids)} matches", flush=True)
        match_ids = list(dict.fromkeys(match_ids))

        print(f"[player {index:>3}/{len(riot_ids)}] {game_name}#{tag_line} -> {len(match_ids)} matches", flush=True)
        started = time.monotonic()
        downloaded = 0
        cached = 0
        errors = 0
        workers = resolve_download_workers(download_workers, len(match_ids))
        print(f"  Download workers: {workers}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_load_match, client, match_id, region, matches_dir, sleep_seconds): match_id
                for match_id in match_ids
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                match_id = futures[future]
                try:
                    match, status = future.result()
                    if status == "existing":
                        cached += 1
                    else:
                        downloaded += 1
                    _accumulate_player_match(aggregates, match, puuid, riot_id, static_data)
                except (RiotApiError, OSError, ValueError) as exc:
                    errors += 1
                    print(f"  error {match_id}: {exc}", file=sys.stderr)
                if completed == 1 or completed % 10 == 0 or completed == len(match_ids):
                    print(f"  [match {completed}/{len(match_ids)}] downloaded={downloaded} cached={cached} errors={errors} elapsed={time.monotonic() - started:.0f}s", flush=True)
        print(f"  Finished: {downloaded} downloaded, {cached} cached, {errors} errors, {time.monotonic() - started:.0f}s", flush=True)

    rows = [
        {
            **row,
            "win_rate": round(row["wins"] / row["games"], 4) if row["games"] else 0,
        }
        for row in aggregates.values()
    ]
    return sorted(rows, key=lambda row: (row["player"], row["champion_name"], row["role"], row["queue_id"]))


def _load_match(client, match_id, region, matches_dir, sleep_seconds):
    if matches_dir is None:
        match = client.match_by_id(match_id, region)
        time.sleep(sleep_seconds)
        return match, "downloaded"
    path = matches_dir / f"{match_id}.json"
    status, error = download_match(client, match_id, region, path, force=False)
    if error:
        raise RiotApiError(error)
    if status == "downloaded":
        time.sleep(sleep_seconds)
    return json.loads(path.read_text(encoding="utf-8")), status


def _accumulate_player_match(
    aggregates: dict[tuple[str, int, str, int], dict[str, Any]],
    match: dict[str, Any],
    puuid: str,
    riot_id: str,
    static_data: Any,
) -> None:
    info = match.get("info", {})
    if not isinstance(info, dict):
        return
    queue_id = _as_int(info.get("queueId"))
    if queue_id not in (420, 440):
        return

    participants = [participant for participant in info.get("participants", []) if isinstance(participant, dict)]
    participant_roles = _participant_roles(participants, static_data)
    for participant in participants:
        if str(participant.get("puuid")) != puuid:
            continue

        champion_id = _as_int(participant.get("championId"))
        participant_id = _as_int(participant.get("participantId"))
        if champion_id is None or participant_id is None:
            return

        role = participant_roles.get(participant_id) or str(participant.get("teamPosition") or "").lower()
        if not role:
            return

        key = (riot_id, champion_id, role, queue_id)
        row = aggregates.setdefault(
            key,
            {
                "player": riot_id,
                "riot_id": riot_id,
                "champion_id": champion_id,
                "champion_name": static_data.champion_name(champion_id),
                "role": role,
                "role_name": ROLE_NAMES.get(role, role),
                "queue_id": queue_id,
                "games": 0,
                "wins": 0,
                "losses": 0,
            },
        )
        row["games"] += 1
        if bool(participant.get("win")):
            row["wins"] += 1
        else:
            row["losses"] += 1
        return


if __name__ == "__main__":
    raise SystemExit(main())
