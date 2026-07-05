from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


DDRAGON_BASE = "https://ddragon.leagueoflegends.com"


@dataclass(frozen=True)
class StaticData:
    version: str | None
    champions: dict[int, str]
    summoner_spells: dict[int, str]
    champion_keys: dict[int, str] = field(default_factory=dict)
    champion_payloads: dict[int, dict[str, Any]] = field(default_factory=dict)

    def champion_name(self, champion_id: int | None) -> str:
        if not champion_id or champion_id <= 0:
            return "-"
        return self.champions.get(champion_id, f"#{champion_id}")

    def spell_name(self, spell_id: int | None) -> str:
        if not spell_id or spell_id <= 0:
            return "-"
        return self.summoner_spells.get(spell_id, f"#{spell_id}")

    def champion_key(self, champion_id: int | None) -> str | None:
        if not champion_id or champion_id <= 0:
            return None
        return self.champion_keys.get(champion_id)


def load_static_data(language: str = "en_US") -> StaticData:
    version = _latest_version()
    if version is None:
        return _load_latest_cached_data(language) or StaticData(
            version=None,
            champions={},
            summoner_spells={},
            champion_keys={},
            champion_payloads={},
        )

    champions = _load_data_file(version, language, "champion.json")
    summoner_spells = _load_data_file(version, language, "summoner.json")

    return StaticData(
        version=version,
        champions=_parse_named_id_map(champions),
        summoner_spells=_parse_named_id_map(summoner_spells),
        champion_keys=_parse_champion_key_map(champions),
        champion_payloads=_parse_champion_payload_map(champions),
    )


def _latest_version() -> str | None:
    data = _fetch_or_cache_json("versions.json", f"{DDRAGON_BASE}/api/versions.json")
    if isinstance(data, list) and data:
        return str(data[0])
    return None


def _load_data_file(version: str, language: str, filename: str) -> dict[str, Any]:
    cache_name = f"{version}_{language}_{filename}"
    url = f"{DDRAGON_BASE}/cdn/{version}/data/{language}/{filename}"
    data = _fetch_or_cache_json(cache_name, url)
    return data if isinstance(data, dict) else {}


def _parse_named_id_map(payload: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return result

    for item in data.values():
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        name = item.get("name")
        if key is None or name is None:
            continue
        try:
            result[int(key)] = str(name)
        except ValueError:
            continue

    return result


def _parse_champion_key_map(payload: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return result

    for item in data.values():
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        champion_key = item.get("id")
        if key is None or champion_key is None:
            continue
        try:
            result[int(key)] = str(champion_key)
        except ValueError:
            continue

    return result


def _parse_champion_payload_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return result

    for item in data.values():
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key is None:
            continue
        try:
            result[int(key)] = item
        except ValueError:
            continue

    return result


def _load_latest_cached_data(language: str) -> StaticData | None:
    cache = cache_dir()
    champion_files = sorted(cache.glob(f"*_{language}_champion.json"), reverse=True)
    for champion_file in champion_files:
        version = champion_file.name.removesuffix(f"_{language}_champion.json")
        summoner_file = cache / f"{version}_{language}_summoner.json"
        if not summoner_file.exists():
            continue
        try:
            champions = json.loads(champion_file.read_text(encoding="utf-8"))
            summoner_spells = json.loads(summoner_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return StaticData(
            version=version,
            champions=_parse_named_id_map(champions),
            summoner_spells=_parse_named_id_map(summoner_spells),
            champion_keys=_parse_champion_key_map(champions),
            champion_payloads=_parse_champion_payload_map(champions),
        )
    return None


def _fetch_or_cache_json(cache_name: str, url: str) -> Any:
    path = cache_dir() / cache_name
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "lol-champ-select-recommender/0.1",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=4.0) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    path = Path(root).expanduser() if root else Path.home() / ".cache"
    path = path / "lol-champ-select-recommender"
    path.mkdir(parents=True, exist_ok=True)
    return path
