from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..champ_select import bans_by_team
from ..ddragon import StaticData
from ..roles import POSITION_ORDER, ROLE_NAMES, RolePriors, assign_roles
from .draft_data import (
    NOT_SELECTED_TOKEN,
    PAD_TOKEN,
    PICK_TOKEN,
    SPECIAL_CHAMPION_TOKENS,
    champion_features_by_id,
    load_champion_feature_rows,
    token_global_feature_ids,
)
from .draft_model import build_model_class, require_torch
from .player_pruning import (
    MAX_LOSSES_FOR_LOW_SAMPLE,
    MIN_GAMES,
    MIN_WIN_RATE,
    PlayerPruneIndex,
    extrapolated_hard_prune_candidates,
    extrapolated_soft_prune_candidates,
    hard_prune_candidates,
    load_player_prune_index,
    soft_prune_candidates,
)


@dataclass(frozen=True)
class DraftPickRecommendation:
    champion_id: int
    score: float


@dataclass(frozen=True)
class DraftRoleRecommendation:
    role: str
    raw: list[DraftPickRecommendation]
    soft: list[DraftPickRecommendation] | None
    hard: list[DraftPickRecommendation] | None
    extrapolated_soft: list[DraftPickRecommendation] | None
    extrapolated_hard: list[DraftPickRecommendation] | None
    whitelisted_soft: list[DraftPickRecommendation] | None
    whitelisted_hard: list[DraftPickRecommendation] | None
    whitelisted_extrapolated_soft: list[DraftPickRecommendation] | None
    whitelisted_extrapolated_hard: list[DraftPickRecommendation] | None

    @property
    def role_label(self) -> str:
        return ROLE_NAMES.get(self.role, self.role)


@dataclass(frozen=True)
class LiveDraftQuery:
    role: str
    feature_ids: list[list[int]]
    query_index: int
    blocked_champion_ids: set[int]


@dataclass(frozen=True)
class DecodedToken:
    index: int
    label: str
    values: list[tuple[str, str]]


@dataclass(frozen=True)
class ChampionBlacklist:
    global_champion_ids: set[int]
    by_role: dict[str, set[int]]

    @property
    def count(self) -> int:
        return len(self.global_champion_ids) + sum(len(champion_ids) for champion_ids in self.by_role.values())

    def is_empty(self) -> bool:
        return self.count == 0

    def blocks(self, champion_id: int, role: str) -> bool:
        return champion_id in self.global_champion_ids or champion_id in self.by_role.get(role, set())


class DraftRecommender:
    def __init__(
        self,
        model,
        model_vocab: dict[str, Any],
        champion_features: dict[int, dict[str, Any]],
        torch_module,
        player_prune_index: PlayerPruneIndex | None = None,
        champion_blacklist: ChampionBlacklist | set[int] | None = None,
    ) -> None:
        self.model = model
        self.model_vocab = model_vocab
        self.champion_features = champion_features
        self._torch = torch_module
        self.player_prune_index = player_prune_index
        self.champion_blacklist = normalize_champion_blacklist(champion_blacklist)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        champion_features_path: str | Path = "data/processed/champion_features.csv",
        player_stats_path: str | Path | None = None,
        champion_blacklist_path: str | Path | None = None,
        device: str | None = None,
    ) -> DraftRecommender:
        torch, _nn = require_torch()
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        model_vocab = checkpoint["model_vocab"]
        model_config = checkpoint["model_config"]
        champion_rows = load_champion_feature_rows(champion_features_path)
        champion_features = champion_features_by_id(champion_rows)
        player_prune_index = load_player_prune_index(player_stats_path) if player_stats_path else None
        champion_blacklist = (
            load_champion_blacklist(champion_blacklist_path, champion_features) if champion_blacklist_path else set()
        )

        SharedFeatureDraftTransformer = build_model_class()
        model = SharedFeatureDraftTransformer(
            shared_vocab_size=model_config["shared_vocab_size"],
            champion_vocab_size=model_config["champion_vocab_size"],
            coarse_bucket_size=model_config.get("coarse_bucket_size", 0),
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            use_role_heads=bool(model_config.get("use_role_heads", False)),
            use_hierarchy=bool(model_config.get("use_hierarchy", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(torch.device(resolved_device))
        model.eval()
        print(model)

        return cls(model, model_vocab, champion_features, torch, player_prune_index, champion_blacklist)

    def recommend(
        self,
        session: dict[str, Any],
        static_data: StaticData,
        *,
        role_priors: RolePriors | None = None,
        top_k: int = 3,
    ) -> list[DraftRoleRecommendation]:
        queries = build_live_queries(
            session,
            static_data,
            self.model_vocab,
            self.champion_features,
            role_priors=role_priors,
        )
        if not queries:
            return []

        torch = self._torch
        if torch is None:
            return []

        device = next(self.model.parameters()).device
        feature_ids = torch.tensor([query.feature_ids for query in queries], dtype=torch.long, device=device)
        query_index = torch.tensor([query.query_index for query in queries], dtype=torch.long, device=device)
        role_index = torch.tensor([POSITION_ORDER.index(query.role) for query in queries], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(feature_ids, query_index, role_index=role_index)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        token_id_to_champion_id = _token_id_to_champion_id(self.model_vocab)

        recommendations: list[DraftRoleRecommendation] = []
        for row_index, query in enumerate(queries):
            row_logits = logits[row_index].clone()
            ranked_token_ids = torch.argsort(row_logits, descending=True).tolist()
            ranked_candidates: list[tuple[int, float]] = []
            for token_id in ranked_token_ids:
                champion_id = token_id_to_champion_id.get(int(token_id))
                if champion_id is None:
                    continue
                if champion_id in query.blocked_champion_ids:
                    continue
                ranked_candidates.append((champion_id, float(row_logits[int(token_id)].item())))

            if not ranked_candidates:
                continue

            raw = _score_ranked_candidates(ranked_candidates, top_k=top_k, torch_module=torch)

            if self.player_prune_index is None:
                soft = None
                hard = None
                extrapolated_soft = None
                extrapolated_hard = None
                whitelisted_soft = None
                whitelisted_hard = None
                whitelisted_extrapolated_soft = None
                whitelisted_extrapolated_hard = None
            else:
                soft_candidates = soft_prune_candidates(
                    [champion_id for champion_id, _score in ranked_candidates],
                    prune_index=self.player_prune_index,
                )
                hard_candidates = hard_prune_candidates(
                    [champion_id for champion_id, _score in ranked_candidates],
                    role=query.role,
                    prune_index=self.player_prune_index,
                )
                soft = _score_ranked_candidates(
                    [(champion_id, _candidate_score(ranked_candidates, champion_id)) for champion_id in soft_candidates],
                    top_k=top_k,
                    torch_module=torch,
                )
                hard = _score_ranked_candidates(
                    [(champion_id, _candidate_score(ranked_candidates, champion_id)) for champion_id in hard_candidates],
                    top_k=top_k,
                    torch_module=torch,
                )
                extrapolated_soft_candidates = extrapolated_soft_prune_candidates(
                    [champion_id for champion_id, _score in ranked_candidates],
                    prune_index=self.player_prune_index,
                )
                extrapolated_hard_candidates = extrapolated_hard_prune_candidates(
                    [champion_id for champion_id, _score in ranked_candidates],
                    role=query.role,
                    prune_index=self.player_prune_index,
                )
                extrapolated_soft = _score_ranked_candidates(
                    [
                        (champion_id, _candidate_score(ranked_candidates, champion_id))
                        for champion_id in extrapolated_soft_candidates
                    ],
                    top_k=top_k,
                    torch_module=torch,
                )
                extrapolated_hard = _score_ranked_candidates(
                    [
                        (champion_id, _candidate_score(ranked_candidates, champion_id))
                        for champion_id in extrapolated_hard_candidates
                    ],
                    top_k=top_k,
                    torch_module=torch,
                )
                if not self.champion_blacklist.is_empty():
                    whitelist_candidates = [
                        champion_id
                        for champion_id, _score in ranked_candidates
                        if not self.champion_blacklist.blocks(champion_id, query.role)
                    ]
                    whitelisted_soft = _score_ranked_candidates(
                        [
                            (champion_id, _candidate_score(ranked_candidates, champion_id))
                            for champion_id in whitelist_candidates
                            if champion_id in soft_candidates
                        ],
                        top_k=top_k,
                        torch_module=torch,
                    )
                    whitelisted_hard = _score_ranked_candidates(
                        [
                            (champion_id, _candidate_score(ranked_candidates, champion_id))
                            for champion_id in whitelist_candidates
                            if champion_id in hard_candidates
                        ],
                        top_k=top_k,
                        torch_module=torch,
                    )
                    whitelisted_extrapolated_soft = _score_ranked_candidates(
                        [
                            (champion_id, _candidate_score(ranked_candidates, champion_id))
                            for champion_id in whitelist_candidates
                            if champion_id in extrapolated_soft_candidates
                        ],
                        top_k=top_k,
                        torch_module=torch,
                    )
                    whitelisted_extrapolated_hard = _score_ranked_candidates(
                        [
                            (champion_id, _candidate_score(ranked_candidates, champion_id))
                            for champion_id in whitelist_candidates
                            if champion_id in extrapolated_hard_candidates
                        ],
                        top_k=top_k,
                        torch_module=torch,
                    )
                else:
                    whitelisted_soft = None
                    whitelisted_hard = None
                    whitelisted_extrapolated_soft = None
                    whitelisted_extrapolated_hard = None

            if raw:
                recommendations.append(
                    DraftRoleRecommendation(
                        role=query.role,
                        raw=raw,
                        soft=soft,
                        hard=hard,
                        extrapolated_soft=extrapolated_soft,
                        extrapolated_hard=extrapolated_hard,
                        whitelisted_soft=whitelisted_soft,
                        whitelisted_hard=whitelisted_hard,
                        whitelisted_extrapolated_soft=whitelisted_extrapolated_soft,
                        whitelisted_extrapolated_hard=whitelisted_extrapolated_hard,
                    )
                )

        return recommendations

    def debug_lines(
        self,
        session: dict[str, Any],
        static_data: StaticData,
        *,
        role_priors: RolePriors | None = None,
    ) -> list[str]:
        queries = build_live_queries(
            session,
            static_data,
            self.model_vocab,
            self.champion_features,
            role_priors=role_priors,
        )
        if not queries:
            return ["Inference debug: no live draft query available"]

        lines = ["Inference debug"]
        for query in queries:
            lines.append(f"  role={ROLE_NAMES.get(query.role, query.role)} query_index={query.query_index}")
            for token in decode_live_tokens(query.feature_ids, self.model_vocab):
                feature_text = ", ".join(f"{name}={value}" for name, value in token.values)
                lines.append(f"    [{token.index:02d}] {token.label}: {feature_text}")
        return lines

    def recommend_lines(
        self,
        session: dict[str, Any],
        static_data: StaticData,
        *,
        role_priors: RolePriors | None = None,
        top_k: int = 3,
    ) -> list[str]:
        recommendations = self.recommend(session, static_data, role_priors=role_priors, top_k=top_k)
        if not recommendations:
            return []

        lines = ["Recommendations", *self.recommendation_legend_lines()]
        lane_lines = self.lane_recommendation_lines()
        if lane_lines:
            lines.extend(lane_lines)
        for recommendation in recommendations:
            lines.append(f"  {recommendation.role_label}")
            lines.append(f"    Raw: {_format_pick_list(recommendation.raw, static_data)}")
            if recommendation.soft is None:
                lines.append("    Soft: unavailable")
            else:
                lines.append(f"    Soft: {_format_pick_list(recommendation.soft, static_data)}")
            if recommendation.hard is None:
                lines.append("    Hard: unavailable")
            else:
                lines.append(f"    Hard: {_format_pick_list(recommendation.hard, static_data)}")
            if recommendation.extrapolated_soft is None:
                lines.append("    Extrapolated Soft: unavailable")
            else:
                lines.append(
                    f"    Extrapolated Soft: {_format_pick_list(recommendation.extrapolated_soft, static_data)}"
                )
            if recommendation.extrapolated_hard is None:
                lines.append("    Extrapolated Hard: unavailable")
            else:
                lines.append(
                    f"    Extrapolated Hard: {_format_pick_list(recommendation.extrapolated_hard, static_data)}"
                )
            if recommendation.whitelisted_soft is None:
                lines.append("    Whitelisted Soft: unavailable")
            else:
                lines.append(f"    Whitelisted Soft: {_format_pick_list(recommendation.whitelisted_soft, static_data)}")
            if recommendation.whitelisted_hard is None:
                lines.append("    Whitelisted Hard: unavailable")
            else:
                lines.append(f"    Whitelisted Hard: {_format_pick_list(recommendation.whitelisted_hard, static_data)}")
            if recommendation.whitelisted_extrapolated_soft is None:
                lines.append("    Whitelisted Extrapolated Soft: unavailable")
            else:
                lines.append(
                    "    Whitelisted Extrapolated Soft: "
                    f"{_format_pick_list(recommendation.whitelisted_extrapolated_soft, static_data)}"
                )
            if recommendation.whitelisted_extrapolated_hard is None:
                lines.append("    Whitelisted Extrapolated Hard: unavailable")
            else:
                lines.append(
                    "    Whitelisted Extrapolated Hard: "
                    f"{_format_pick_list(recommendation.whitelisted_extrapolated_hard, static_data)}"
                )
        return lines

    def recommendation_legend_lines(self) -> list[str]:
        return _pruning_legend_lines()

    def lane_recommendation_lines(self) -> list[str]:
        player_prune_index = getattr(self, "player_prune_index", None)
        if player_prune_index is None:
            return []

        hard = player_prune_index.hard_lane_recommendations()
        soft = player_prune_index.soft_lane_recommendations()
        return [
            "  Lane",
            f"    Hard: {_format_lane_list(hard)}",
            f"    Soft: {_format_lane_list(soft)}",
        ]

    def prune_status(self) -> str:
        if self.player_prune_index is None:
            player_status = "unavailable"
        else:
            player_status = f"loaded {self.player_prune_index.source}"
        parts: list[str] = [player_status]
        if not self.champion_blacklist.is_empty():
            parts.append(f"blacklist: {self.champion_blacklist.count} entries")
        else:
            parts.append("blacklist: unavailable")
        return " | ".join(parts)


def build_live_queries(
    session: dict[str, Any],
    static_data: StaticData,
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
    *,
    role_priors: RolePriors | None = None,
) -> list[LiveDraftQuery]:
    my_side = infer_my_side(session)
    if not my_side:
        return []

    ally_players, enemy_players = team_players_by_side(session, my_side)
    local_cell_id = _as_int(session.get("localPlayerCellId"))
    ally_roles = selected_role_map(
        ally_players,
        static_data,
        role_priors=role_priors,
        exclude_hover_cell_ids={local_cell_id} if local_cell_id is not None else set(),
    )
    enemy_roles = selected_role_map(enemy_players, static_data, role_priors=role_priors)
    ally_bans, enemy_bans = bans_by_side(session, my_side)

    blocked_champion_ids = {
        champion_id
        for champion_id in (
            *ally_roles.values(),
            *enemy_roles.values(),
            *ally_bans,
            *enemy_bans,
        )
        if champion_id and champion_id > 0
    }

    queries: list[LiveDraftQuery] = []
    for role in POSITION_ORDER:
        if ally_roles.get(role):
            continue
        feature_ids = _live_feature_ids_for_role(
            role,
            my_side=my_side,
            ally_roles=ally_roles,
            enemy_roles=enemy_roles,
            ally_bans=ally_bans,
            enemy_bans=enemy_bans,
            model_vocab=model_vocab,
            champion_features=champion_features,
        )
        queries.append(
            LiveDraftQuery(
                role=role,
                feature_ids=feature_ids,
                query_index=POSITION_ORDER.index(role),
                blocked_champion_ids=blocked_champion_ids,
            )
        )

    return queries


def _live_feature_ids_for_role(
    role: str,
    *,
    my_side: str,
    ally_roles: dict[str, int | None],
    enemy_roles: dict[str, int | None],
    ally_bans: list[int],
    enemy_bans: list[int],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
) -> list[list[int]]:
    tokens: list[dict[str, Any]] = []
    enemy_side = "red" if my_side == "blue" else "blue"

    for side_label, role_map, map_side in (("ally", ally_roles, my_side), ("enemy", enemy_roles, enemy_side)):
        for position in POSITION_ORDER:
            champion_id = role_map.get(position)
            is_query = side_label == "ally" and position == role
            if is_query:
                champion_token = PICK_TOKEN
                champion_id = None
            elif champion_id and champion_id > 0:
                champion_token = str(champion_id)
            else:
                champion_token = NOT_SELECTED_TOKEN
                champion_id = None

            tokens.append(
                {
                    "champion_token": champion_token,
                    "champion_id": champion_id,
                    "role": position,
                    "side": side_label,
                    "token_type": "pick",
                    "map_side": map_side,
                }
            )

    for side_label, bans, map_side in (("ally", ally_bans, my_side), ("enemy", enemy_bans, enemy_side)):
        for index in range(5):
            champion_id = bans[index] if index < len(bans) else None
            champion_token = PAD_TOKEN if champion_id is None or champion_id <= 0 else str(champion_id)
            tokens.append(
                {
                    "champion_token": champion_token,
                    "champion_id": champion_id,
                    "role": "ban",
                    "side": side_label,
                    "token_type": "ban",
                    "map_side": map_side,
                }
            )

    return [
        token_global_feature_ids(token, {}, model_vocab, champion_features)
        for token in tokens
    ]


def decode_live_tokens(feature_ids: list[list[int]], model_vocab: dict[str, Any]) -> list[DecodedToken]:
    inverse_vocabs = {
        feature_name: {value: token for token, value in vocab.items()}
        for feature_name, vocab in model_vocab["feature_vocabs"].items()
    }
    decoded: list[DecodedToken] = []
    for index, token_features in enumerate(feature_ids):
        values: list[tuple[str, str]] = []
        label = "token"
        for feature_name, global_id in zip(model_vocab["token_features"], token_features):
            offset = int(model_vocab["feature_offsets"][feature_name])
            local_id = int(global_id) - offset
            token = inverse_vocabs[feature_name].get(local_id, "?")
            values.append((feature_name, str(token)))
            if feature_name == "champion":
                label = str(token)
        decoded.append(DecodedToken(index=index, label=label, values=values))
    return decoded


def infer_my_side(session: dict[str, Any]) -> str | None:
    local_cell = _as_int(session.get("localPlayerCellId"))
    if local_cell is not None:
        return "blue" if local_cell <= 5 else "red"

    for player in session.get("myTeam", []):
        if not isinstance(player, dict):
            continue
        cell = _as_int(player.get("cellId"))
        if cell is not None:
            return "blue" if cell <= 5 else "red"

    return None


def team_players_by_side(session: dict[str, Any], my_side: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    my_team = session.get("myTeam", [])
    their_team = session.get("theirTeam", [])
    return _dict_players(my_team), _dict_players(their_team)


def selected_role_map(
    players: list[dict[str, Any]],
    static_data: StaticData,
    *,
    role_priors: RolePriors | None = None,
    exclude_hover_cell_ids: set[int] | None = None,
) -> dict[str, int | None]:
    selected_players: list[dict[str, Any]] = []
    for player in players:
        cell_id = _as_int(player.get("cellId"))
        champion_id = live_champion_id(player, use_hover=cell_id not in (exclude_hover_cell_ids or set()))
        if champion_id and champion_id > 0:
            selected_players.append({**player, "championId": champion_id})

    assignments = assign_roles(selected_players, static_data, infer_missing=True, role_priors=role_priors)
    role_map: dict[str, int | None] = {position: None for position in POSITION_ORDER}

    for player in selected_players:
        cell_id = _as_int(player.get("cellId"))
        if cell_id is None:
            continue
        assignment = assignments.get(cell_id)
        if assignment and assignment.position in POSITION_ORDER:
            champion_id = _as_int(player.get("championId"))
            if champion_id and champion_id > 0:
                role_map[assignment.position] = champion_id

    return role_map


def live_champion_id(player: dict[str, Any], *, use_hover: bool = True) -> int | None:
    champion_id = _as_int(player.get("championId"))
    if champion_id and champion_id > 0:
        return champion_id
    if not use_hover:
        return None
    hover_id = _as_int(player.get("championPickIntent"))
    return hover_id if hover_id and hover_id > 0 else None


def bans_by_side(session: dict[str, Any], my_side: str) -> tuple[list[int], list[int]]:
    return bans_by_team(session)


def _dict_players(players: Any) -> list[dict[str, Any]]:
    return [player for player in players if isinstance(player, dict)]


def _token_id_to_champion_id(model_vocab: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for champion_token, token_id in model_vocab["champion_token_to_id"].items():
        if str(champion_token).isdigit():
            result[int(token_id)] = int(champion_token)
    return result


def load_champion_blacklist(
    path: str | Path,
    champion_features: dict[int, dict[str, Any]],
) -> ChampionBlacklist:
    blacklist_path = Path(path)
    if not blacklist_path.is_file():
        return empty_champion_blacklist()

    champion_lookup: dict[str, int] = {}
    for champion_id, features in champion_features.items():
        for key in ("champion_name", "champion_key"):
            value = str(features.get(key, "")).strip().lower()
            if value:
                champion_lookup[value] = champion_id

    blocked_global: set[int] = set()
    blocked_by_role: dict[str, set[int]] = {}
    try:
        lines = blacklist_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return empty_champion_blacklist()

    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        role, champion_text = parse_blacklist_entry(entry)
        champion_id = champion_id_from_blacklist_text(champion_text, champion_lookup)
        if champion_id is None:
            continue
        if role is None:
            blocked_global.add(champion_id)
        else:
            blocked_by_role.setdefault(role, set()).add(champion_id)

    return ChampionBlacklist(global_champion_ids=blocked_global, by_role=blocked_by_role)


def normalize_champion_blacklist(value: ChampionBlacklist | set[int] | None) -> ChampionBlacklist:
    if value is None:
        return empty_champion_blacklist()
    if isinstance(value, ChampionBlacklist):
        return value
    return ChampionBlacklist(global_champion_ids=set(value), by_role={})


def empty_champion_blacklist() -> ChampionBlacklist:
    return ChampionBlacklist(global_champion_ids=set(), by_role={})


def parse_blacklist_entry(entry: str) -> tuple[str | None, str]:
    if ":" not in entry:
        return None, entry

    prefix, champion_text = entry.split(":", 1)
    if prefix.strip().lower() in {"all", "any", "*"}:
        return None, champion_text.strip()
    role = normalize_blacklist_role(prefix)
    if role is None:
        return None, entry
    return role, champion_text.strip()


def normalize_blacklist_role(value: str) -> str | None:
    token = value.strip().lower()
    aliases = {
        "support": "utility",
        "sup": "utility",
        "bot": "bottom",
        "adc": "bottom",
        "mid": "middle",
        "jg": "jungle",
    }
    role = aliases.get(token, token)
    if role in POSITION_ORDER:
        return role
    return None


def champion_id_from_blacklist_text(text: str, champion_lookup: dict[str, int]) -> int | None:
    token = text.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    return champion_lookup.get(token.lower())



def _score_ranked_candidates(
    ranked_candidates: list[tuple[int, float]],
    *,
    top_k: int,
    torch_module,
) -> list[DraftPickRecommendation]:
    if not ranked_candidates:
        return []

    limited = ranked_candidates[: max(1, top_k)]
    logits = torch_module.tensor([score for _champion_id, score in limited], dtype=torch_module.float32)
    probabilities = torch_module.softmax(logits, dim=0).tolist()
    return [
        DraftPickRecommendation(champion_id=champion_id, score=float(probability))
        for (champion_id, _score), probability in zip(limited, probabilities)
    ]


def _candidate_score(ranked_candidates: list[tuple[int, float]], champion_id: int) -> float:
    for candidate_champion_id, candidate_score in ranked_candidates:
        if candidate_champion_id == champion_id:
            return candidate_score
    return float("-inf")


def _format_pick_list(picks: list[DraftPickRecommendation], static_data: StaticData) -> str:
    if not picks:
        return "-"
    return ", ".join(f"{static_data.champion_name(pick.champion_id)} {pick.score:.0%}" for pick in picks)


def _pruning_legend_lines() -> list[str]:
    return [
        "  Legend",
        f"    Champion Soft: {MIN_GAMES}+ games and {MIN_WIN_RATE:.0%}+ WR overall",
        f"    Champion Hard: Soft plus {MIN_GAMES}+ games and {MIN_WIN_RATE:.0%}+ WR in the recommended role",
        f"    Champion Extrapolated: also keeps <{MIN_GAMES} games when losses < {MAX_LOSSES_FOR_LOW_SAMPLE:g}; missing stats count as 0 games",
        "    Whitelisted: same filters after the champion blacklist",
        f"    Lane Hard: {MIN_GAMES}+ games and {MIN_WIN_RATE:.0%}+ WR on that lane",
        f"    Lane Soft: <{MIN_GAMES} games and losses < {MAX_LOSSES_FOR_LOW_SAMPLE:g} on that lane; missing lanes count as 0 games",
    ]


def _format_lane_list(lanes: list[tuple[str, Any]]) -> str:
    if not lanes:
        return "-"
    return ", ".join(
        f"{ROLE_NAMES.get(role, role)} {stats.win_rate:.0%} ({stats.games}g)"
        for role, stats in lanes
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
