from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ddragon import StaticData


POSITION_ORDER = ("top", "jungle", "middle", "bottom", "utility")
SMITE_ID = 11

ROLE_NAMES = {
    "top": "Top",
    "jungle": "Jungle",
    "middle": "Mid",
    "bottom": "Bot",
    "utility": "Support",
    "": "-",
}


@dataclass(frozen=True)
class RoleAssignment:
    position: str | None
    inferred: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        if not self.position:
            return "-"
        label = ROLE_NAMES.get(self.position, self.position)
        return f"{label}?" if self.inferred else label


@dataclass(frozen=True)
class RolePriors:
    source: str
    patch: str | None
    queue_id: int | None
    min_total_games: int
    weights_by_champion: dict[int, dict[str, int]]

    @property
    def champion_count(self) -> int:
        return len(self.weights_by_champion)

    def role_weights(self, champion_id: int | None) -> dict[str, int]:
        if not champion_id:
            return {}
        return self.weights_by_champion.get(champion_id, {})


def load_role_priors(
    path: str | Path = "data/processed/champion_role_priors.csv",
    *,
    queue_id: int | None = 420,
    patch: str | None = None,
    min_total_games: int = 5,
) -> RolePriors | None:
    priors_path = Path(path)
    if not priors_path.is_file():
        return None

    try:
        with priors_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return None

    if queue_id is not None:
        rows = [row for row in rows if _as_int(row.get("queue_id")) == queue_id]

    if patch is None:
        patch = _latest_patch([str(row.get("patch", "")) for row in rows])
    if patch:
        rows = [row for row in rows if str(row.get("patch", "")) == patch]

    weights_by_champion: dict[int, dict[str, int]] = {}
    for row in rows:
        champion_id = _as_int(row.get("champion_id"))
        role = _normalized_position(row.get("role"))
        total_games = _as_int(row.get("total_games")) or 0
        if champion_id is None or role is None or total_games < min_total_games:
            continue

        role_share = _as_float(row.get("role_share")) or 0
        score = max(1, int(round(role_share * 1000)))
        weights_by_champion.setdefault(champion_id, {})[role] = score

    return RolePriors(
        source=str(priors_path),
        patch=patch,
        queue_id=queue_id,
        min_total_games=min_total_games,
        weights_by_champion=weights_by_champion,
    )


def assign_roles(
    players: Any,
    static_data: StaticData,
    infer_missing: bool,
    role_priors: RolePriors | None = None,
) -> dict[int, RoleAssignment]:
    if not isinstance(players, list):
        return {}

    assignments: dict[int, RoleAssignment] = {}
    taken_positions: set[str] = set()
    normalized_players = [player for player in players if isinstance(player, dict)]

    for player in normalized_players:
        cell_id = _cell_id(player)
        if cell_id is None:
            continue
        position = _normalized_position(player.get("assignedPosition"))
        if position:
            assignments[cell_id] = RoleAssignment(position=position, inferred=False, reason="api")
            taken_positions.add(position)

    if not infer_missing:
        for player in normalized_players:
            cell_id = _cell_id(player)
            if cell_id is not None and cell_id not in assignments:
                assignments[cell_id] = RoleAssignment(position=None)
        return assignments

    for player in normalized_players:
        cell_id = _cell_id(player)
        if cell_id is None or cell_id in assignments:
            continue
        if _has_smite(player) and "jungle" not in taken_positions:
            assignments[cell_id] = RoleAssignment(position="jungle", inferred=True, reason="smite")
            taken_positions.add("jungle")

    inferred_assignments = _best_inferred_assignments(
        normalized_players,
        static_data,
        assigned_cell_ids=set(assignments),
        taken_positions=taken_positions,
        role_priors=role_priors,
    )
    for cell_id, position in inferred_assignments.items():
        assignments[cell_id] = RoleAssignment(position=position, inferred=True, reason="champion-team-fit")
        taken_positions.add(position)

    for player in normalized_players:
        cell_id = _cell_id(player)
        if cell_id is not None and cell_id not in assignments:
            assignments[cell_id] = RoleAssignment(position=None)

    return assignments


def champion_position_weights(champion_key: str | None) -> dict[str, int]:
    if not champion_key:
        return {}
    return CHAMPION_POSITION_WEIGHTS.get(champion_key, {})


def champion_role_weights(
    *,
    champion_id: int | None,
    champion_key: str | None,
    role_priors: RolePriors | None = None,
) -> dict[str, int]:
    if role_priors:
        prior_weights = role_priors.role_weights(champion_id)
        if prior_weights:
            return prior_weights
    return champion_position_weights(champion_key)


def _best_inferred_assignments(
    players: list[dict[str, Any]],
    static_data: StaticData,
    assigned_cell_ids: set[int],
    taken_positions: set[str],
    role_priors: RolePriors | None,
) -> dict[int, str]:
    remaining_positions = set(POSITION_ORDER) - taken_positions
    remaining_players: list[tuple[int, int, dict[str, int]]] = []

    for order, player in enumerate(players):
        cell_id = _cell_id(player)
        if cell_id is None or cell_id in assigned_cell_ids:
            continue
        champion_id = _selected_champion_id(player)
        weights = {
            position: score
            for position, score in champion_role_weights(
                champion_id=champion_id,
                champion_key=static_data.champion_key(champion_id),
                role_priors=role_priors,
            ).items()
            if position in remaining_positions
        }
        remaining_players.append((order, cell_id, weights))

    best_key = (-1, -1)
    best_assignments: dict[int, str] = {}

    def search(
        index: int,
        used_positions: set[str],
        current_assignments: dict[int, str],
        assigned_count: int,
        total_score: int,
    ) -> None:
        nonlocal best_key, best_assignments

        if index >= len(remaining_players):
            key = (assigned_count, total_score)
            if key > best_key:
                best_key = key
                best_assignments = dict(current_assignments)
            return

        _order, cell_id, weights = remaining_players[index]

        search(index + 1, used_positions, current_assignments, assigned_count, total_score)

        for position, score in sorted(weights.items(), key=lambda item: (-item[1], POSITION_ORDER.index(item[0]))):
            if position in used_positions:
                continue
            current_assignments[cell_id] = position
            used_positions.add(position)
            search(
                index + 1,
                used_positions,
                current_assignments,
                assigned_count + 1,
                total_score + score,
            )
            used_positions.remove(position)
            del current_assignments[cell_id]

    search(0, set(), {}, 0, 0)
    return best_assignments


def _normalized_position(value: Any) -> str | None:
    position = str(value or "").lower()
    return position if position in POSITION_ORDER else None


def _cell_id(player: dict[str, Any]) -> int | None:
    try:
        return int(player.get("cellId"))
    except (TypeError, ValueError):
        return None


def _selected_champion_id(player: dict[str, Any]) -> int | None:
    for key in ("championId", "championPickIntent"):
        try:
            champion_id = int(player.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if champion_id > 0:
            return champion_id
    return None


def _has_smite(player: dict[str, Any]) -> bool:
    return _spell_id(player.get("spell1Id")) == SMITE_ID or _spell_id(player.get("spell2Id")) == SMITE_ID


def _spell_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_patch(patches: list[str]) -> str | None:
    valid_patches = [patch for patch in patches if patch]
    if not valid_patches:
        return None

    return max(valid_patches, key=_patch_sort_key)


def _patch_sort_key(patch: str) -> tuple[int, ...]:
    parts = []
    for part in patch.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _weights(*positions: str) -> dict[str, int]:
    return {position: max(10, 100 - index * 20) for index, position in enumerate(positions)}


CHAMPION_POSITION_WEIGHTS = {
    "Aatrox": _weights("top"),
    "Ahri": _weights("middle"),
    "Akali": _weights("middle", "top"),
    "Akshan": _weights("middle", "top"),
    "Alistar": _weights("utility"),
    "Ambessa": _weights("top", "middle"),
    "Amumu": _weights("jungle", "utility"),
    "Anivia": _weights("middle"),
    "Annie": _weights("middle", "utility"),
    "Aphelios": _weights("bottom"),
    "Ashe": _weights("bottom", "utility"),
    "AurelionSol": _weights("middle"),
    "Aurora": _weights("middle", "top"),
    "Azir": _weights("middle"),
    "Bard": _weights("utility"),
    "Belveth": _weights("jungle"),
    "Blitzcrank": _weights("utility"),
    "Brand": _weights("utility", "jungle", "middle"),
    "Braum": _weights("utility"),
    "Briar": _weights("jungle"),
    "Caitlyn": _weights("bottom"),
    "Camille": _weights("top", "utility"),
    "Cassiopeia": _weights("middle", "top"),
    "Chogath": _weights("top", "middle"),
    "Corki": _weights("middle", "bottom"),
    "Darius": _weights("top"),
    "Diana": _weights("jungle", "middle"),
    "DrMundo": _weights("top", "jungle"),
    "Draven": _weights("bottom"),
    "Ekko": _weights("jungle", "middle"),
    "Elise": _weights("jungle"),
    "Evelynn": _weights("jungle"),
    "Ezreal": _weights("bottom"),
    "Fiddlesticks": _weights("jungle", "utility"),
    "Fiora": _weights("top"),
    "Fizz": _weights("middle"),
    "Galio": _weights("middle", "utility"),
    "Gangplank": _weights("top"),
    "Garen": _weights("top"),
    "Gnar": _weights("top"),
    "Gragas": _weights("top", "jungle", "middle"),
    "Graves": _weights("jungle", "top"),
    "Gwen": _weights("top", "jungle"),
    "Hecarim": _weights("jungle"),
    "Heimerdinger": _weights("middle", "top", "utility"),
    "Hwei": _weights("middle", "utility"),
    "Illaoi": _weights("top"),
    "Irelia": _weights("top", "middle"),
    "Ivern": _weights("jungle"),
    "Janna": _weights("utility"),
    "JarvanIV": _weights("jungle", "top"),
    "Jax": _weights("top", "jungle"),
    "Jayce": _weights("top", "middle"),
    "Jhin": _weights("bottom"),
    "Jinx": _weights("bottom"),
    "KSante": _weights("top"),
    "Kaisa": _weights("bottom"),
    "Kalista": _weights("bottom"),
    "Karma": _weights("utility", "middle", "top"),
    "Karthus": _weights("jungle", "middle"),
    "Kassadin": _weights("middle"),
    "Katarina": _weights("middle"),
    "Kayle": _weights("top", "middle"),
    "Kayn": _weights("jungle"),
    "Kennen": _weights("top", "middle"),
    "Khazix": _weights("jungle"),
    "Kindred": _weights("jungle"),
    "Kled": _weights("top", "middle"),
    "KogMaw": _weights("bottom"),
    "Leblanc": _weights("middle"),
    "LeeSin": _weights("jungle"),
    "Leona": _weights("utility"),
    "Lillia": _weights("jungle", "top"),
    "Lissandra": _weights("middle"),
    "Locke": _weights("utility", "middle"),
    "Lucian": _weights("bottom", "middle"),
    "Lulu": _weights("utility"),
    "Lux": _weights("utility", "middle"),
    "Malphite": _weights("top", "middle", "utility"),
    "Malzahar": _weights("middle"),
    "Maokai": _weights("top", "utility", "jungle"),
    "MasterYi": _weights("jungle"),
    "Mel": _weights("middle", "utility"),
    "Milio": _weights("utility"),
    "MissFortune": _weights("bottom"),
    "Mordekaiser": _weights("top"),
    "Morgana": _weights("utility", "jungle", "middle"),
    "Naafiri": _weights("middle"),
    "Nami": _weights("utility"),
    "Nasus": _weights("top"),
    "Nautilus": _weights("utility", "jungle"),
    "Neeko": _weights("middle", "utility"),
    "Nidalee": _weights("jungle"),
    "Nilah": _weights("bottom"),
    "Nocturne": _weights("jungle", "middle"),
    "Nunu": _weights("jungle"),
    "Olaf": _weights("jungle", "top"),
    "Orianna": _weights("middle"),
    "Ornn": _weights("top"),
    "Pantheon": _weights("top", "middle", "utility"),
    "Poppy": _weights("top", "jungle", "utility"),
    "Pyke": _weights("utility"),
    "Qiyana": _weights("middle", "jungle"),
    "Quinn": _weights("top"),
    "Rakan": _weights("utility"),
    "Rammus": _weights("jungle"),
    "RekSai": _weights("jungle"),
    "Rell": _weights("utility", "jungle"),
    "Renata": _weights("utility"),
    "Renekton": _weights("top", "middle"),
    "Rengar": _weights("jungle", "top"),
    "Riven": _weights("top"),
    "Rumble": _weights("top", "middle", "jungle"),
    "Ryze": _weights("middle", "top"),
    "Samira": _weights("bottom"),
    "Sejuani": _weights("jungle", "top"),
    "Senna": _weights("utility", "bottom"),
    "Seraphine": _weights("utility", "bottom", "middle"),
    "Sett": _weights("top", "utility"),
    "Shaco": _weights("jungle", "utility"),
    "Shen": _weights("top", "utility"),
    "Shyvana": _weights("jungle", "top"),
    "Singed": _weights("top"),
    "Sion": _weights("top"),
    "Sivir": _weights("bottom"),
    "Skarner": _weights("jungle", "top"),
    "Smolder": _weights("bottom", "middle"),
    "Sona": _weights("utility"),
    "Soraka": _weights("utility"),
    "Swain": _weights("middle", "utility", "bottom"),
    "Sylas": _weights("middle", "top"),
    "Syndra": _weights("middle"),
    "TahmKench": _weights("top", "utility"),
    "Taliyah": _weights("middle", "jungle"),
    "Talon": _weights("middle", "jungle"),
    "Taric": _weights("utility"),
    "Teemo": _weights("top", "utility"),
    "Thresh": _weights("utility"),
    "Tristana": _weights("bottom", "middle"),
    "Trundle": _weights("jungle", "top"),
    "Tryndamere": _weights("top", "middle"),
    "TwistedFate": _weights("middle"),
    "Twitch": _weights("bottom", "jungle", "utility"),
    "Udyr": _weights("jungle", "top"),
    "Urgot": _weights("top"),
    "Varus": _weights("bottom"),
    "Vayne": _weights("bottom", "top"),
    "Veigar": _weights("middle", "bottom", "utility"),
    "Velkoz": _weights("utility", "middle"),
    "Vex": _weights("middle"),
    "Vi": _weights("jungle"),
    "Viego": _weights("jungle"),
    "Viktor": _weights("middle"),
    "Vladimir": _weights("middle", "top"),
    "Volibear": _weights("jungle", "top"),
    "Warwick": _weights("jungle", "top"),
    "MonkeyKing": _weights("jungle", "top"),
    "Xayah": _weights("bottom"),
    "Xerath": _weights("utility", "middle"),
    "XinZhao": _weights("jungle"),
    "Yasuo": _weights("middle", "bottom", "top"),
    "Yone": _weights("middle", "top"),
    "Yorick": _weights("top"),
    "Yunara": _weights("bottom"),
    "Yuumi": _weights("utility"),
    "Zaahen": _weights("top", "jungle"),
    "Zac": _weights("jungle", "top", "utility"),
    "Zed": _weights("middle", "jungle"),
    "Zeri": _weights("bottom"),
    "Ziggs": _weights("bottom", "middle"),
    "Zilean": _weights("utility", "middle"),
    "Zoe": _weights("middle"),
    "Zyra": _weights("utility", "jungle"),
}
