from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..roles import POSITION_ORDER


MIN_GAMES = 20
MIN_WIN_RATE = 0.52
MAX_LOSSES_FOR_LOW_SAMPLE = MIN_GAMES * (1 - MIN_WIN_RATE)
LANE_HARD_MIN_WIN_RATE = 0.53
LANE_SOFT_MAX_LOSSES_FOR_LOW_SAMPLE = MIN_GAMES * (1 - MIN_WIN_RATE)


@dataclass(frozen=True)
class PruneStats:
    games: int
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass(frozen=True)
class PlayerPruneIndex:
    source: str
    overall_by_champion: dict[int, PruneStats]
    by_role_by_champion: dict[str, dict[int, PruneStats]]
    by_role: dict[str, PruneStats]

    def overall_stats(self, champion_id: int | None) -> PruneStats | None:
        if not champion_id:
            return None
        return self.overall_by_champion.get(champion_id)

    def role_stats(self, champion_id: int | None, role: str) -> PruneStats | None:
        if not champion_id:
            return None
        role = normalize_role(role)
        if not role:
            return None
        return self.by_role_by_champion.get(role, {}).get(champion_id)

    def passes_soft(self, champion_id: int | None) -> bool:
        return passes_strict_threshold(self.overall_stats(champion_id))

    def passes_hard(self, champion_id: int | None, role: str) -> bool:
        return passes_strict_threshold(self.role_stats(champion_id, role))

    def passes_soft_extrapolated(self, champion_id: int | None) -> bool:
        return passes_extrapolated_threshold(self.overall_stats(champion_id))

    def passes_hard_extrapolated(self, champion_id: int | None, role: str) -> bool:
        return passes_extrapolated_threshold(self.role_stats(champion_id, role))

    def prune_reasons(self, champion_id: int | None, role: str) -> tuple[bool, bool]:
        return self.passes_soft(champion_id), self.passes_hard(champion_id, role)

    def hard_lane_recommendations(self) -> list[tuple[str, PruneStats]]:
        rows = [
            (role, stats)
            for role, stats in self.by_role.items()
            if stats.games >= MIN_GAMES and stats.win_rate >= LANE_HARD_MIN_WIN_RATE
        ]
        return sorted(rows, key=lambda item: (-item[1].win_rate, -item[1].games, item[0]))

    def soft_lane_recommendations(self) -> list[tuple[str, PruneStats]]:
        rows = [
            (role, stats)
            for role, stats in self.by_role.items()
            if stats.games < MIN_GAMES and stats.losses < LANE_SOFT_MAX_LOSSES_FOR_LOW_SAMPLE
        ]
        return sorted(rows, key=lambda item: (item[1].losses, -item[1].win_rate, -item[1].games, item[0]))


def load_player_prune_index(path: str | Path) -> PlayerPruneIndex | None:
    prune_path = Path(path)
    if not prune_path.is_file():
        return None

    try:
        with prune_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return None

    overall: dict[int, list[int]] = {}
    by_role: dict[str, dict[int, list[int]]] = {}
    role_totals: dict[str, list[int]] = {}

    for row in rows:
        champion_id = _as_int(row.get("champion_id"))
        role = normalize_role(row.get("role"))
        games = _as_int(row.get("games")) or 0
        wins = _as_int(row.get("wins")) or 0
        losses = _as_int(row.get("losses")) or max(0, games - wins)

        if champion_id is None or not role:
            continue

        _accumulate(overall, champion_id, games, wins, losses)
        role_bucket = by_role.setdefault(role, {})
        _accumulate(role_bucket, champion_id, games, wins, losses)
        _accumulate_role(role_totals, role, games, wins, losses)

    return PlayerPruneIndex(
        source=str(prune_path),
        overall_by_champion=_finalize(overall),
        by_role_by_champion={role: _finalize(bucket) for role, bucket in by_role.items()},
        by_role=_finalize_roles(role_totals),
    )


def prune_candidates(
    champion_ids: list[int],
    *,
    role: str,
    prune_index: PlayerPruneIndex | None,
) -> list[int]:
    if prune_index is None:
        return champion_ids

    kept: list[int] = []
    for champion_id in champion_ids:
        if prune_index.passes_soft(champion_id) and prune_index.passes_hard(champion_id, role):
            kept.append(champion_id)
    return kept


def soft_prune_candidates(
    champion_ids: list[int],
    *,
    prune_index: PlayerPruneIndex | None,
) -> list[int]:
    if prune_index is None:
        return champion_ids

    kept: list[int] = []
    for champion_id in champion_ids:
        if prune_index.passes_soft(champion_id):
            kept.append(champion_id)
    return kept


def hard_prune_candidates(
    champion_ids: list[int],
    *,
    role: str,
    prune_index: PlayerPruneIndex | None,
) -> list[int]:
    if prune_index is None:
        return champion_ids

    kept: list[int] = []
    for champion_id in champion_ids:
        if prune_index.passes_soft(champion_id) and prune_index.passes_hard(champion_id, role):
            kept.append(champion_id)
    return kept


def extrapolated_soft_prune_candidates(
    champion_ids: list[int],
    *,
    prune_index: PlayerPruneIndex | None,
) -> list[int]:
    if prune_index is None:
        return champion_ids

    kept: list[int] = []
    for champion_id in champion_ids:
        if prune_index.passes_soft_extrapolated(champion_id):
            kept.append(champion_id)
    return kept


def extrapolated_hard_prune_candidates(
    champion_ids: list[int],
    *,
    role: str,
    prune_index: PlayerPruneIndex | None,
) -> list[int]:
    if prune_index is None:
        return champion_ids

    kept: list[int] = []
    for champion_id in champion_ids:
        if prune_index.passes_soft_extrapolated(champion_id) and prune_index.passes_hard_extrapolated(champion_id, role):
            kept.append(champion_id)
    return kept


def passes_strict_threshold(stats: PruneStats | None) -> bool:
    if stats is None:
        return False
    return stats.games >= MIN_GAMES and stats.win_rate >= MIN_WIN_RATE


def passes_extrapolated_threshold(stats: PruneStats | None) -> bool:
    if stats is None:
        return False
    if stats.games >= MIN_GAMES:
        return stats.win_rate >= MIN_WIN_RATE
    return stats.losses < MAX_LOSSES_FOR_LOW_SAMPLE


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in POSITION_ORDER else ""


def _accumulate(bucket: dict[int, list[int]], champion_id: int, games: int, wins: int, losses: int) -> None:
    values = bucket.setdefault(champion_id, [0, 0, 0])
    values[0] += games
    values[1] += wins
    values[2] += losses


def _accumulate_role(bucket: dict[str, list[int]], role: str, games: int, wins: int, losses: int) -> None:
    values = bucket.setdefault(role, [0, 0, 0])
    values[0] += games
    values[1] += wins
    values[2] += losses


def _finalize(bucket: dict[int, list[int]]) -> dict[int, PruneStats]:
    return {champion_id: PruneStats(games=values[0], wins=values[1], losses=values[2]) for champion_id, values in bucket.items()}


def _finalize_roles(bucket: dict[str, list[int]]) -> dict[str, PruneStats]:
    return {role: PruneStats(games=values[0], wins=values[1], losses=values[2]) for role, values in bucket.items()}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
