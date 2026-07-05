from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..roles import POSITION_ORDER


SPECIAL_CHAMPION_TOKENS = ["<PAD>", "<PICK>", "<NOT_SELECTED>", "<UNK>"]
PAD_TOKEN = "<PAD>"
PICK_TOKEN = "<PICK>"
NOT_SELECTED_TOKEN = "<NOT_SELECTED>"
UNK_TOKEN = "<UNK>"
NONE_TOKEN = "<NONE>"

TOKEN_FEATURES = [
    "champion",
    "role",
    "side",
    "token_type",
    "map_side",
    "primary_tag",
    "secondary_tag",
    "partype",
    "range_type",
]
CHAMPION_CATEGORICAL_FEATURES = ["primary_tag", "secondary_tag", "partype", "range_type"]


@dataclass(frozen=True)
class TrainingExample:
    feature_ids: list[list[int]]
    target: int
    target_coarse: int
    query_index: int
    target_champion_id: int
    target_role: str


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_champion_feature_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_model_vocab(
    draft_rows: list[dict[str, Any]],
    champion_feature_rows: list[dict[str, Any]],
    *,
    numeric_bins: int = 8,
) -> dict[str, Any]:
    champion_ids = sorted(
        {
            int(row["champion_id"])
            for row in champion_feature_rows
            if str(row.get("champion_id", "")).isdigit()
        }
    )
    champion_token_to_id = {token: index for index, token in enumerate(SPECIAL_CHAMPION_TOKENS)}
    for champion_id in champion_ids:
        champion_token_to_id[str(champion_id)] = len(champion_token_to_id)
    champion_id_to_token_id = {str(champion_id): champion_token_to_id[str(champion_id)] for champion_id in champion_ids}

    numeric_feature_names = sorted(
        name
        for name in (champion_feature_rows[0].keys() if champion_feature_rows else [])
        if name.startswith("info_") or name.startswith("stat_")
    )
    numeric_bin_edges = {
        name: quantile_edges([to_float(row.get(name)) for row in champion_feature_rows], numeric_bins)
        for name in numeric_feature_names
    }
    coarse_bucket_values = sorted(
        {
            coarse_bucket_value(row, numeric_bin_edges)
            for row in champion_feature_rows
            if coarse_bucket_value(row, numeric_bin_edges)
        }
    )
    coarse_bucket_to_id = {bucket: index for index, bucket in enumerate(coarse_bucket_values)}
    champion_id_to_coarse_bucket_id = {
        str(int(row["champion_id"])): coarse_bucket_to_id.get(coarse_bucket_value(row, numeric_bin_edges), 0)
        for row in champion_feature_rows
        if str(row.get("champion_id", "")).isdigit()
    }

    feature_vocabs: dict[str, dict[str, int]] = {
        "champion": champion_token_to_id,
        "role": vocab_from_tokens([NONE_TOKEN, "ban", *POSITION_ORDER]),
        "side": vocab_from_tokens([NONE_TOKEN, "ally", "enemy"]),
        "token_type": vocab_from_tokens(["pick", "ban"]),
        "map_side": vocab_from_tokens([NONE_TOKEN, "blue", "red"]),
    }

    for feature in CHAMPION_CATEGORICAL_FEATURES:
        feature_vocabs[feature] = vocab_from_tokens(row.get(feature) for row in champion_feature_rows)

    for feature_name, edges in numeric_bin_edges.items():
        feature_vocabs[f"bin_{feature_name}"] = {UNK_TOKEN: 0}
        for index in range(len(edges) + 1):
            feature_vocabs[f"bin_{feature_name}"][f"bin_{index}"] = index + 1

    offsets: dict[str, int] = {}
    offset = 0
    for feature_name in [*TOKEN_FEATURES, *(f"bin_{name}" for name in numeric_feature_names)]:
        offsets[feature_name] = offset
        offset += len(feature_vocabs[feature_name])

    return {
        "schema_version": 1,
        "special_champion_tokens": SPECIAL_CHAMPION_TOKENS,
        "token_features": [*TOKEN_FEATURES, *(f"bin_{name}" for name in numeric_feature_names)],
        "context_token_count": len(POSITION_ORDER) * 2 + 10,
        "champion_token_to_id": champion_token_to_id,
        "champion_id_to_token_id": champion_id_to_token_id,
        "coarse_bucket_to_id": coarse_bucket_to_id,
        "champion_id_to_coarse_bucket_id": champion_id_to_coarse_bucket_id,
        "feature_vocabs": feature_vocabs,
        "feature_offsets": offsets,
        "shared_vocab_size": offset,
        "champion_vocab_size": len(champion_token_to_id),
        "coarse_bucket_size": len(coarse_bucket_to_id),
        "numeric_feature_names": numeric_feature_names,
        "numeric_bin_edges": numeric_bin_edges,
        "numeric_bins": numeric_bins,
    }


def champion_features_by_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        champion_id = to_int(row.get("champion_id"))
        if champion_id is not None:
            result[champion_id] = row
    return result


def build_training_example(
    row: dict[str, Any],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
    *,
    rng: random.Random,
    mask_probability: float = 0.25,
    unk_probability: float = 0.03,
    target_role: str | None = None,
) -> TrainingExample:
    winning_side = str(row["winning_side"])
    losing_side = "red" if winning_side == "blue" else "blue"
    ally = row[winning_side]
    enemy = row[losing_side]
    target_role = target_role or rng.choice(list(POSITION_ORDER))
    target_champion_id = to_int(ally[target_role])
    if target_champion_id is None:
        raise ValueError(f"Missing target champion for role {target_role}")

    tokens: list[dict[str, Any]] = []
    query_index = -1

    for side_label, draft, map_side in (("ally", ally, winning_side), ("enemy", enemy, losing_side)):
        for role in POSITION_ORDER:
            champion_id = to_int(draft.get(role))
            is_query = side_label == "ally" and role == target_role
            if is_query:
                champion_token = PICK_TOKEN
                query_index = len(tokens)
            else:
                champion_token = sampled_champion_token(champion_id, rng, mask_probability, unk_probability)
            tokens.append(
                {
                    "champion_token": champion_token,
                    "champion_id": champion_id,
                    "role": role,
                    "side": side_label,
                    "token_type": "pick",
                    "map_side": map_side,
                }
            )

    ally_bans = row[f"{winning_side}_bans"]
    enemy_bans = row[f"{losing_side}_bans"]
    for side_label, bans, map_side in (("ally", ally_bans, winning_side), ("enemy", enemy_bans, losing_side)):
        for index in range(5):
            champion_id = to_int(bans[index]) if index < len(bans) else -1
            champion_token = PAD_TOKEN if champion_id is None or champion_id <= 0 else sampled_champion_token(
                champion_id,
                rng,
                mask_probability=0.0,
                unk_probability=unk_probability,
            )
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

    if query_index < 0:
        raise ValueError("No PICK query token was created.")

    feature_ids = [
        token_global_feature_ids(token, row, model_vocab, champion_features)
        for token in tokens
    ]
    return TrainingExample(
        feature_ids=feature_ids,
        target=champion_token_id_for_champion(target_champion_id, model_vocab),
        target_coarse=int(model_vocab["champion_id_to_coarse_bucket_id"][str(target_champion_id)]),
        query_index=query_index,
        target_champion_id=target_champion_id,
        target_role=target_role,
    )


def token_global_feature_ids(
    token: dict[str, Any],
    row: dict[str, Any],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
) -> list[int]:
    champion_id = to_int(token.get("champion_id"))
    champion_token = str(token["champion_token"])
    static_features = (
        champion_features.get(champion_id or -1, {})
        if champion_token not in SPECIAL_CHAMPION_TOKENS
        else {}
    )
    local_values: dict[str, str] = {
        "champion": champion_token_value(champion_token, champion_id, model_vocab),
        "role": token["role"],
        "side": token["side"],
        "token_type": token["token_type"],
        "map_side": token["map_side"],
        "primary_tag": category_value(static_features.get("primary_tag")),
        "secondary_tag": category_value(static_features.get("secondary_tag")),
        "partype": category_value(static_features.get("partype")),
        "range_type": category_value(static_features.get("range_type")),
    }

    for feature_name in model_vocab["numeric_feature_names"]:
        edges = model_vocab["numeric_bin_edges"][feature_name]
        value = to_float(static_features.get(feature_name))
        local_values[f"bin_{feature_name}"] = numeric_bin_token(value, edges)

    return [
        global_feature_id(
            feature_name,
            local_values.get(feature_name, UNK_TOKEN if feature_name.startswith("bin_") else NONE_TOKEN),
            model_vocab,
        )
        for feature_name in model_vocab["token_features"]
    ]


def champion_token_value(champion_token: str, champion_id: int | None, model_vocab: dict[str, Any]) -> str:
    if champion_token in SPECIAL_CHAMPION_TOKENS:
        return champion_token
    if champion_id is None:
        return UNK_TOKEN
    value = str(champion_id)
    return value if value in model_vocab["feature_vocabs"]["champion"] else UNK_TOKEN


def champion_token_id_for_champion(champion_id: int, model_vocab: dict[str, Any]) -> int:
    return int(model_vocab["champion_id_to_token_id"].get(str(champion_id), model_vocab["champion_token_to_id"][UNK_TOKEN]))


def sampled_champion_token(
    champion_id: int | None,
    rng: random.Random,
    mask_probability: float,
    unk_probability: float,
) -> str:
    if champion_id is None or champion_id <= 0:
        return PAD_TOKEN
    sample = rng.random()
    if sample < unk_probability:
        return UNK_TOKEN
    if sample < unk_probability + mask_probability:
        return NOT_SELECTED_TOKEN
    return str(champion_id)


def global_feature_id(feature_name: str, value: str, model_vocab: dict[str, Any]) -> int:
    vocab = model_vocab["feature_vocabs"][feature_name]
    local_id = vocab.get(value, vocab.get(UNK_TOKEN, 0))
    return int(model_vocab["feature_offsets"][feature_name]) + int(local_id)


def vocab_from_tokens(tokens: Any) -> dict[str, int]:
    values = {category_value(token) for token in tokens}
    ordered = [UNK_TOKEN]
    if NONE_TOKEN in values:
        ordered.append(NONE_TOKEN)
    ordered.extend(sorted(value for value in values if value not in {UNK_TOKEN, NONE_TOKEN}))
    return {value: index for index, value in enumerate(ordered)}


def category_value(value: Any) -> str:
    token = str(value or "").strip()
    return token if token else NONE_TOKEN


COARSE_STAT_FEATURES = (
    "info_attack",
    "info_defense",
    "info_magic",
    "stat_armor",
    "stat_armorperlevel",
    "stat_attackdamage",
    "stat_attackdamageperlevel",
    "stat_attackrange",
    "stat_attackspeed",
    "stat_attackspeedperlevel",
    "stat_hp",
    "stat_hpperlevel",
    "stat_hpregen",
    "stat_hpregenperlevel",
    "stat_movespeed",
    "stat_mp",
    "stat_mpperlevel",
    "stat_mpregen",
    "stat_mpregenperlevel",
    "stat_spellblock",
    "stat_spellblockperlevel",
)


def coarse_bucket_value(row: dict[str, Any], numeric_bin_edges: dict[str, list[float]]) -> str:
    parts = [
        category_value(row.get("primary_tag")),
        category_value(row.get("secondary_tag")),
        category_value(row.get("partype")),
        category_value(row.get("range_type")),
    ]
    for feature_name in COARSE_STAT_FEATURES:
        if feature_name not in numeric_bin_edges:
            continue
        value = to_float(row.get(feature_name))
        parts.append(f"{feature_name}={numeric_bin_token(value, numeric_bin_edges[feature_name])}")
    if all(part == NONE_TOKEN for part in parts[:4]):
        return ""
    return "|".join(parts)


def numeric_bin_token(value: float | None, edges: list[float]) -> str:
    if value is None or math.isnan(value):
        return UNK_TOKEN
    bin_index = 0
    for edge in edges:
        if value > edge:
            bin_index += 1
    return f"bin_{bin_index}"


def quantile_edges(values: list[float | None], bins: int) -> list[float]:
    clean = sorted(value for value in values if value is not None and not math.isnan(value))
    if not clean or bins <= 1:
        return []

    edges: list[float] = []
    for index in range(1, bins):
        raw_position = index * (len(clean) - 1) / bins
        lower = math.floor(raw_position)
        upper = math.ceil(raw_position)
        if lower == upper:
            edge = clean[lower]
        else:
            ratio = raw_position - lower
            edge = clean[lower] * (1 - ratio) + clean[upper] * ratio
        if not edges or edge > edges[-1]:
            edges.append(round(edge, 6))
    return edges


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
