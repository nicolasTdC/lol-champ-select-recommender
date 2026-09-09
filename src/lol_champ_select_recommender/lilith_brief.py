from __future__ import annotations

from typing import Any

from .champ_select import bans_by_team
from .ddragon import StaticData
from .roles import RolePriors, assign_roles


def format_lilith_brief(
    *,
    phase: str,
    session: dict[str, Any] | None,
    static_data: StaticData,
    role_priors: RolePriors | None = None,
) -> str:
    if phase != "ChampSelect" or not isinstance(session, dict):
        return f"League client phase: {phase or 'unknown'}. Not in champion select."

    local_cell = session.get("localPlayerCellId")
    lines = [
        "League champion select (live client data, not a screenshot).",
        f"You are cell {local_cell}.",
    ]

    you = _find_player(session.get("myTeam"), local_cell)
    ally_roles = assign_roles(
        session.get("myTeam", []),
        static_data,
        infer_missing=False,
        role_priors=role_priors,
    )
    enemy_roles = assign_roles(
        session.get("theirTeam", []),
        static_data,
        infer_missing=True,
        role_priors=role_priors,
    )
    if you:
        role = ally_roles.get(_as_int(you.get("cellId")))
        lines.append(
            "Your lane: "
            + (role.label if role else "-")
            + f" | pick {static_data.champion_name(_as_int(you.get('championId')))}"
            + f" | hover {static_data.champion_name(_as_int(you.get('championPickIntent')))}"
        )

    lines.append("Allies:")
    lines.extend(_team_lines(session.get("myTeam"), static_data, ally_roles, local_cell))
    lines.append("Enemies:")
    lines.extend(_team_lines(session.get("theirTeam"), static_data, enemy_roles, None))

    ally_bans, enemy_bans = bans_by_team(session)
    lines.append("Ally bans: " + _names(ally_bans, static_data))
    lines.append("Enemy bans: " + _names(enemy_bans, static_data))
    return "\n".join(lines)


def _team_lines(
    players: Any,
    static_data: StaticData,
    roles: dict[int, Any],
    local_cell: Any,
) -> list[str]:
    if not isinstance(players, list) or not players:
        return ["  (none visible yet)"]
    rows: list[str] = []
    for player in players:
        if not isinstance(player, dict):
            continue
        cell = player.get("cellId")
        marker = " YOU" if cell == local_cell else ""
        role = roles.get(_as_int(cell))
        role_label = role.label if role else "-"
        pick = static_data.champion_name(_as_int(player.get("championId")))
        hover = static_data.champion_name(_as_int(player.get("championPickIntent")))
        rows.append(f"  cell {cell}{marker}: {role_label} pick={pick} hover={hover}")
    return rows or ["  (none visible yet)"]


def _find_player(players: Any, cell_id: Any) -> dict[str, Any] | None:
    if not isinstance(players, list):
        return None
    for player in players:
        if isinstance(player, dict) and player.get("cellId") == cell_id:
            return player
    return None


def _names(ids: Any, static_data: StaticData) -> str:
    if not isinstance(ids, list) or not ids:
        return "-"
    names = [static_data.champion_name(_as_int(value)) for value in ids if (_as_int(value) or 0) > 0]
    return ", ".join(names) if names else "-"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
