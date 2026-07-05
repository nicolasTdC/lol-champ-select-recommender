from __future__ import annotations

import shutil
from typing import Any

from .ddragon import StaticData
from .roles import RolePriors, assign_roles


def render_session(
    *,
    phase: str,
    session: dict[str, Any] | None,
    static_data: StaticData,
    lockfile_label: str,
    role_priors: RolePriors | None = None,
) -> str:
    width = shutil.get_terminal_size((120, 30)).columns
    lines: list[str] = []
    lines.append("League Champ Select Watcher")
    lines.append("=" * min(width, 120))
    lines.append(f"LCU: {lockfile_label}")
    lines.append(f"Gameflow: {phase}")
    if static_data.version:
        lines.append(f"Data Dragon: {static_data.version}")
    else:
        lines.append("Data Dragon: unavailable, showing raw IDs")
    if role_priors:
        lines.append(
            "Role priors: "
            f"{role_priors.patch or 'all patches'} queue {role_priors.queue_id or 'all'} "
            f"({role_priors.champion_count} champions, min {role_priors.min_total_games} games)"
        )
    else:
        lines.append("Role priors: unavailable, using fallback map")
    lines.append("")

    if phase != "ChampSelect" or not session:
        lines.append("Not in champion select.")
        return "\n".join(lines)

    timer = session.get("timer", {})
    lines.append(_timer_line(timer))
    lines.append("")

    lines.extend(_current_actions(session, static_data))
    lines.append("")

    lines.append("Allies")
    ally_rows, _has_inferred_ally_roles = _team_table(
        session.get("myTeam", []),
        static_data,
        session.get("localPlayerCellId"),
        infer_missing_roles=False,
        role_priors=role_priors,
    )
    lines.extend(ally_rows)
    lines.append("")

    lines.append("Enemies")
    enemy_rows, has_inferred_roles = _team_table(
        session.get("theirTeam", []),
        static_data,
        session.get("localPlayerCellId"),
        infer_missing_roles=True,
        role_priors=role_priors,
    )
    lines.extend(enemy_rows)
    if has_inferred_roles:
        lines.append("  ? = inferred from champion/spells")
    lines.append("")

    bans = session.get("bans", {})
    lines.append(f"Ally bans:  {_name_list(bans.get('myTeamBans', []), static_data)}")
    lines.append(f"Enemy bans: {_name_list(bans.get('theirTeamBans', []), static_data)}")

    return "\n".join(lines)


def _timer_line(timer: dict[str, Any]) -> str:
    phase = timer.get("phase", "-")
    total = _milliseconds_to_seconds(timer.get("totalTimeInPhase"))
    left = _milliseconds_to_seconds(timer.get("adjustedTimeLeftInPhase"))
    return f"Draft phase: {phase} | Time left: {left}s | Phase length: {total}s"


def _team_table(
    players: Any,
    static_data: StaticData,
    local_cell_id: int | None,
    infer_missing_roles: bool,
    role_priors: RolePriors | None,
) -> tuple[list[str], bool]:
    if not isinstance(players, list) or not players:
        return ["  (none visible yet)"], False

    role_assignments = assign_roles(
        players,
        static_data,
        infer_missing=infer_missing_roles,
        role_priors=role_priors,
    )
    has_inferred_roles = any(assignment.inferred for assignment in role_assignments.values())

    rows = [
        "  {:<5} {:<9} {:<18} {:<18} {:<19} {:<19}".format(
            "Cell",
            "Role",
            "Pick",
            "Hover",
            "Spell 1",
            "Spell 2",
        )
    ]
    rows.append("  " + "-" * 93)

    for player in players:
        if not isinstance(player, dict):
            continue

        cell_id = player.get("cellId")
        marker = "*" if cell_id == local_cell_id else " "
        role = role_assignments.get(_as_int(cell_id))
        role_label = role.label if role else "-"
        pick = static_data.champion_name(_as_int(player.get("championId")))
        hover = static_data.champion_name(_as_int(player.get("championPickIntent")))
        spell_1 = static_data.spell_name(_as_int(player.get("spell1Id")))
        spell_2 = static_data.spell_name(_as_int(player.get("spell2Id")))

        rows.append(
            "{} {:<5} {:<9} {:<18} {:<18} {:<19} {:<19}".format(
                marker,
                str(cell_id if cell_id is not None else "-"),
                _clip(role_label, 9),
                _clip(pick, 18),
                _clip(hover, 18),
                _clip(spell_1, 19),
                _clip(spell_2, 19),
            )
        )

    return rows, has_inferred_roles


def _current_actions(session: dict[str, Any], static_data: StaticData) -> list[str]:
    actions = []
    for group in session.get("actions", []):
        if not isinstance(group, list):
            continue
        for action in group:
            if isinstance(action, dict):
                actions.append(action)

    in_progress = [action for action in actions if action.get("isInProgress")]
    if not in_progress:
        return ["Current action: -"]

    lines = ["Current action:"]
    local_cell_id = session.get("localPlayerCellId")
    for action in in_progress:
        actor = action.get("actorCellId")
        actor_label = "you" if actor == local_cell_id else f"cell {actor}"
        champion = static_data.champion_name(_as_int(action.get("championId")))
        action_type = action.get("type", "-")
        side = "ally" if action.get("isAllyAction") else "enemy"
        lines.append(f"  {side} {actor_label}: {action_type} {champion}")
    return lines


def _name_list(ids: Any, static_data: StaticData) -> str:
    if not isinstance(ids, list) or not ids:
        return "-"
    names = [static_data.champion_name(_as_int(value)) for value in ids if _as_int(value) and _as_int(value) > 0]
    return ", ".join(names) if names else "-"


def _milliseconds_to_seconds(value: Any) -> int:
    try:
        return max(0, int(value) // 1000)
    except (TypeError, ValueError):
        return 0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "."
