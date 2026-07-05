from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .ddragon import StaticData, load_static_data


UNK_TOKEN = "<UNK>"
NONE_TOKEN = "<NONE>"
CATEGORICAL_FEATURE_TO_VOCAB = {
    "primary_tag": "tag",
    "secondary_tag": "tag",
    "partype": "partype",
    "range_type": "range_type",
}
INFO_COLUMNS = ("attack", "defense", "magic", "difficulty")
STAT_COLUMNS = (
    "hp",
    "hpperlevel",
    "mp",
    "mpperlevel",
    "movespeed",
    "armor",
    "armorperlevel",
    "spellblock",
    "spellblockperlevel",
    "attackrange",
    "hpregen",
    "hpregenperlevel",
    "mpregen",
    "mpregenperlevel",
    "attackdamage",
    "attackdamageperlevel",
    "attackspeed",
    "attackspeedperlevel",
)


def main() -> int:
    args = parse_args()
    static_data = load_static_data(args.language)
    rows = build_champion_feature_rows(static_data)
    vocabs = build_feature_vocabs(rows)
    rows = add_categorical_ids(rows, vocabs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_champion_features_csv(output, rows)
    print(f"Wrote {len(rows)} champion feature rows to {output}")

    vocab_output = Path(args.vocab_output)
    vocab_output.parent.mkdir(parents=True, exist_ok=True)
    write_feature_vocab_json(vocab_output, vocabs)
    print(f"Wrote champion feature vocab to {vocab_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build champion feature rows from Data Dragon static data.")
    parser.add_argument(
        "--language",
        default="en_US",
        help="Data Dragon language code. Default: en_US",
    )
    parser.add_argument(
        "--output",
        default="data/processed/champion_features.csv",
        help="Output CSV path. Default: data/processed/champion_features.csv",
    )
    parser.add_argument(
        "--vocab-output",
        default="data/processed/champion_feature_vocab.json",
        help="Output categorical vocab JSON path. Default: data/processed/champion_feature_vocab.json",
    )
    return parser.parse_args()


def build_champion_feature_rows(static_data: StaticData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for champion_id, payload in sorted(
        static_data.champion_payloads.items(),
        key=lambda item: str(item[1].get("name", "")),
    ):
        tags = [str(tag) for tag in payload.get("tags", []) if tag]
        info = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
        stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
        attack_range = _number(stats.get("attackrange"))

        row: dict[str, Any] = {
            "version": static_data.version or "",
            "champion_id": champion_id,
            "champion_key": payload.get("id", static_data.champion_key(champion_id) or ""),
            "champion_name": payload.get("name", static_data.champion_name(champion_id)),
            "title": payload.get("title", ""),
            "partype": _category_token(payload.get("partype")),
            "primary_tag": _category_token(tags[0] if tags else None),
            "secondary_tag": _category_token(tags[1] if len(tags) > 1 else None),
            "range_type": "ranged" if attack_range >= 300 else "melee",
        }

        for column in INFO_COLUMNS:
            row[f"info_{column}"] = _number(info.get(column))

        for column in STAT_COLUMNS:
            row[f"stat_{column}"] = _number(stats.get(column))

        rows.append(row)

    return rows


def build_feature_vocabs(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    tokens_by_vocab: dict[str, set[str]] = {}

    for feature, vocab_name in CATEGORICAL_FEATURE_TO_VOCAB.items():
        tokens_by_vocab.setdefault(vocab_name, set())
        for row in rows:
            tokens_by_vocab[vocab_name].add(_category_token(row.get(feature)))

    vocabs: dict[str, dict[str, int]] = {}
    for vocab_name, tokens in sorted(tokens_by_vocab.items()):
        ordered_tokens = [UNK_TOKEN]
        if NONE_TOKEN in tokens:
            ordered_tokens.append(NONE_TOKEN)
        ordered_tokens.extend(sorted(token for token in tokens if token not in {UNK_TOKEN, NONE_TOKEN}))
        vocabs[vocab_name] = {token: index for index, token in enumerate(ordered_tokens)}

    return vocabs


def add_categorical_ids(
    rows: list[dict[str, Any]],
    vocabs: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    rows_with_ids: list[dict[str, Any]] = []

    for row in rows:
        row_with_ids = dict(row)
        for feature, vocab_name in CATEGORICAL_FEATURE_TO_VOCAB.items():
            vocab = vocabs[vocab_name]
            token = _category_token(row_with_ids.get(feature))
            row_with_ids[f"{feature}_id"] = vocab.get(token, vocab[UNK_TOKEN])
        rows_with_ids.append(row_with_ids)

    return rows_with_ids


def write_champion_features_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "version",
        "champion_id",
        "champion_key",
        "champion_name",
        "title",
        "partype",
        "partype_id",
        "primary_tag",
        "primary_tag_id",
        "secondary_tag",
        "secondary_tag_id",
        "range_type",
        "range_type_id",
        *[f"info_{column}" for column in INFO_COLUMNS],
        *[f"stat_{column}" for column in STAT_COLUMNS],
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_feature_vocab_json(path: Path, vocabs: dict[str, dict[str, int]]) -> None:
    payload = {
        "unknown_token": UNK_TOKEN,
        "none_token": NONE_TOKEN,
        "feature_to_vocab": CATEGORICAL_FEATURE_TO_VOCAB,
        "vocabs": vocabs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _category_token(value: Any) -> str:
    token = str(value or "").strip()
    return token if token else NONE_TOKEN


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
