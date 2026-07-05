from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .ddragon import load_static_data
from .roles import ROLE_NAMES, assign_roles


ROLE_TO_NAME = {key.upper(): value for key, value in ROLE_NAMES.items() if key}
QUEUE_SR = {400, 420, 430, 440, 700}


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    priors_output_path = Path(args.priors_output)

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    static_data = load_static_data(args.language)
    stats = aggregate_match_files(input_dir, static_data, include_non_sr=args.include_non_sr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, stats)
    print(f"Wrote {len(stats)} champion-role rows to {output_path}")

    priors = build_role_priors(stats)
    priors_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_role_priors_csv(priors_output_path, priors)
    print(f"Wrote {len(priors)} champion-role prior rows to {priors_output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate downloaded Match-V5 JSON into champion-role stats.")
    parser.add_argument(
        "--input-dir",
        default="data/raw/matches",
        help="Directory containing raw Match-V5 JSON files. Default: data/raw/matches",
    )
    parser.add_argument(
        "--output",
        default="data/processed/champion_role_stats.csv",
        help="Output CSV path. Default: data/processed/champion_role_stats.csv",
    )
    parser.add_argument(
        "--priors-output",
        default="data/processed/champion_role_priors.csv",
        help="Output role-priors CSV path. Default: data/processed/champion_role_priors.csv",
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
    return parser.parse_args()


def aggregate_match_files(input_dir: Path, static_data: Any, *, include_non_sr: bool = False) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, int, str, int], dict[str, Any]] = {}

    for path in sorted(input_dir.glob("*.json")):
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping unreadable match file {path}: {exc}", file=sys.stderr)
            continue

        info = match.get("info", {})
        if not isinstance(info, dict):
            continue

        queue_id = _as_int(info.get("queueId")) or 0
        if not include_non_sr and queue_id not in QUEUE_SR:
            continue

        patch = _patch(info.get("gameVersion"))
        participants = [p for p in info.get("participants", []) if isinstance(p, dict)]
        participant_roles = _participant_roles(participants, static_data)

        for participant in participants:
            champion_id = _as_int(participant.get("championId"))
            participant_id = _as_int(participant.get("participantId"))
            if not champion_id or participant_id is None:
                continue

            role = participant_roles.get(participant_id)
            if not role:
                continue

            key = (patch, queue_id, role, champion_id)
            row = aggregates.setdefault(
                key,
                {
                    "patch": patch,
                    "queue_id": queue_id,
                    "role": role,
                    "role_name": ROLE_NAMES.get(role, role),
                    "champion_id": champion_id,
                    "champion_name": static_data.champion_name(champion_id),
                    "games": 0,
                    "wins": 0,
                },
            )
            row["games"] += 1
            if bool(participant.get("win")):
                row["wins"] += 1

    rows = []
    for row in aggregates.values():
        games = row["games"]
        wins = row["wins"]
        rows.append(
            {
                **row,
                "win_rate": round(wins / games, 4) if games else 0,
            }
        )

    return sorted(rows, key=lambda row: (row["patch"], row["queue_id"], row["role"], row["champion_name"]))


def build_role_priors(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int], int] = {}

    for row in stats_rows:
        key = (str(row["patch"]), int(row["queue_id"]), int(row["champion_id"]))
        totals[key] = totals.get(key, 0) + int(row["games"])

    rows: list[dict[str, Any]] = []
    for row in stats_rows:
        key = (str(row["patch"]), int(row["queue_id"]), int(row["champion_id"]))
        total_games = totals[key]
        games = int(row["games"])
        rows.append(
            {
                "patch": row["patch"],
                "queue_id": row["queue_id"],
                "champion_id": row["champion_id"],
                "champion_name": row["champion_name"],
                "total_games": total_games,
                "role": row["role"],
                "role_name": row["role_name"],
                "games": games,
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "role_share": round(games / total_games, 4) if total_games else 0,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["patch"],
            row["queue_id"],
            row["champion_name"],
            -row["role_share"],
            row["role"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patch",
        "queue_id",
        "role",
        "role_name",
        "champion_id",
        "champion_name",
        "games",
        "wins",
        "win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_role_priors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patch",
        "queue_id",
        "champion_id",
        "champion_name",
        "total_games",
        "role",
        "role_name",
        "games",
        "wins",
        "win_rate",
        "role_share",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _participant_roles(participants: list[dict[str, Any]], static_data: Any) -> dict[int, str]:
    roles: dict[int, str] = {}

    for team_id in sorted({participant.get("teamId") for participant in participants}):
        team = [participant for participant in participants if participant.get("teamId") == team_id]
        adapted = []
        for participant in team:
            participant_id = _as_int(participant.get("participantId"))
            if participant_id is None:
                continue
            adapted.append(
                {
                    "cellId": participant_id,
                    "assignedPosition": _normalized_match_position(participant.get("teamPosition")),
                    "championId": participant.get("championId"),
                    "spell1Id": participant.get("summoner1Id"),
                    "spell2Id": participant.get("summoner2Id"),
                }
            )

        assignments = assign_roles(adapted, static_data, infer_missing=True)
        for participant_id, assignment in assignments.items():
            if assignment.position:
                roles[participant_id] = assignment.position

    return roles


def _normalized_match_position(value: Any) -> str:
    position = str(value or "").lower()
    if position == "invalid":
        return ""
    return position


def _patch(game_version: Any) -> str:
    version = str(game_version or "unknown")
    parts = version.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return version


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
