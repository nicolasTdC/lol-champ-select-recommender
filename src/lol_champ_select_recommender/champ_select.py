from __future__ import annotations

from typing import Any


def bans_by_team(session: dict[str, Any]) -> tuple[list[int], list[int]]:
    bans = session.get("bans", {})
    my_bans = [_as_int(value) or -1 for value in bans.get("myTeamBans", [])] if isinstance(bans, dict) else []
    their_bans = [_as_int(value) or -1 for value in bans.get("theirTeamBans", [])] if isinstance(bans, dict) else []

    action_my_bans: list[int] = []
    action_their_bans: list[int] = []
    for action in flat_actions(session):
        if action.get("type") != "ban":
            continue
        champion_id = _as_int(action.get("championId")) or -1
        if champion_id <= 0:
            continue
        target = action_my_bans if action.get("isAllyAction") else action_their_bans
        if champion_id not in target:
            target.append(champion_id)

    return _merge_bans(my_bans, action_my_bans), _merge_bans(their_bans, action_their_bans)


def flat_actions(session: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for group in session.get("actions", []):
        if not isinstance(group, list):
            continue
        for action in group:
            if isinstance(action, dict):
                actions.append(action)
    return actions


def _merge_bans(primary: list[int], fallback: list[int]) -> list[int]:
    result = [champion_id for champion_id in primary if champion_id > 0]
    for champion_id in fallback:
        if champion_id > 0 and champion_id not in result:
            result.append(champion_id)
    return result


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
