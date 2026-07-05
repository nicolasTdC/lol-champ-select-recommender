from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


VALID_REGIONS = {"americas", "asia", "europe", "sea"}
VALID_PLATFORMS = {
    "br1",
    "eun1",
    "euw1",
    "jp1",
    "kr",
    "la1",
    "la2",
    "me1",
    "na1",
    "oc1",
    "ru",
    "sg2",
    "tr1",
    "tw2",
    "vn2",
}
PLATFORM_TO_REGION = {
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "na1": "americas",
    "oc1": "sea",
    "eun1": "europe",
    "euw1": "europe",
    "me1": "europe",
    "ru": "europe",
    "tr1": "europe",
    "jp1": "asia",
    "kr": "asia",
    "sg2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


class RiotApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class RiotApiClient:
    api_key: str
    timeout: float = 10.0

    def account_by_riot_id(self, game_name: str, tag_line: str, region: str = "americas") -> dict[str, Any]:
        path = (
            "/riot/account/v1/accounts/by-riot-id/"
            f"{urllib.parse.quote(game_name, safe='')}/{urllib.parse.quote(tag_line, safe='')}"
        )
        return self._request_json(region, path)

    def match_ids_by_puuid(
        self,
        puuid: str,
        region: str = "americas",
        *,
        start: int = 0,
        count: int = 20,
        queue: int | None = None,
        match_type: str | None = None,
    ) -> list[str]:
        params: dict[str, Any] = {
            "start": start,
            "count": count,
        }
        if queue is not None:
            params["queue"] = queue
        if match_type:
            params["type"] = match_type

        path = f"/lol/match/v5/matches/by-puuid/{urllib.parse.quote(puuid, safe='')}/ids"
        data = self._request_json(region, path, params=params)
        if not isinstance(data, list):
            raise RiotApiError("Riot returned an unexpected match-id response.")
        return [str(match_id) for match_id in data]

    def match_by_id(self, match_id: str, region: str = "americas") -> dict[str, Any]:
        path = f"/lol/match/v5/matches/{urllib.parse.quote(match_id, safe='')}"
        data = self._request_json(region, path)
        if not isinstance(data, dict):
            raise RiotApiError(f"Riot returned an unexpected match response for {match_id}.")
        return data

    def league_entries(
        self,
        platform: str,
        *,
        queue: str,
        tier: str,
        division: str,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        path = (
            "/lol/league/v4/entries/"
            f"{urllib.parse.quote(queue, safe='')}/"
            f"{urllib.parse.quote(tier.upper(), safe='')}/"
            f"{urllib.parse.quote(division.upper(), safe='')}"
        )
        data = self._request_platform_json(platform, path, params={"page": page})
        if not isinstance(data, list):
            raise RiotApiError("Riot returned an unexpected league entries response.")
        return [entry for entry in data if isinstance(entry, dict)]

    def apex_league(self, platform: str, *, tier: str, queue: str) -> dict[str, Any]:
        tier = tier.upper()
        if tier == "MASTER":
            league_name = "masterleagues"
        elif tier == "GRANDMASTER":
            league_name = "grandmasterleagues"
        elif tier == "CHALLENGER":
            league_name = "challengerleagues"
        else:
            raise RiotApiError(f"{tier} is not an apex tier.")

        path = f"/lol/league/v4/{league_name}/by-queue/{urllib.parse.quote(queue, safe='')}"
        data = self._request_platform_json(platform, path)
        if not isinstance(data, dict):
            raise RiotApiError(f"Riot returned an unexpected {tier} league response.")
        return data

    def apex_league_entries(self, platform: str, *, tier: str, queue: str) -> list[dict[str, Any]]:
        league = self.apex_league(platform, tier=tier, queue=queue)
        entries = league.get("entries", [])
        if not isinstance(entries, list):
            raise RiotApiError(f"Riot returned an unexpected {tier} entries response.")
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            enriched = dict(entry)
            enriched.setdefault("tier", tier.upper())
            enriched.setdefault("queueType", queue)
            result.append(enriched)
        return result

    def summoner_by_id(self, platform: str, encrypted_summoner_id: str) -> dict[str, Any]:
        path = f"/lol/summoner/v4/summoners/{urllib.parse.quote(encrypted_summoner_id, safe='')}"
        data = self._request_platform_json(platform, path)
        if not isinstance(data, dict):
            raise RiotApiError("Riot returned an unexpected summoner response.")
        return data

    def _request_platform_json(
        self,
        platform: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        retry_429_once: bool = True,
    ) -> Any:
        platform = normalize_platform(platform)
        return self._request_host_json(
            f"{platform}.api.riotgames.com",
            path,
            params,
            retry_429_once=retry_429_once,
        )

    def _request_json(
        self,
        region: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        retry_429_once: bool = True,
    ) -> Any:
        region = region.lower()
        if region not in VALID_REGIONS:
            valid = ", ".join(sorted(VALID_REGIONS))
            raise RiotApiError(f"Invalid Riot regional routing value '{region}'. Use one of: {valid}.")

        return self._request_host_json(
            f"{region}.api.riotgames.com",
            path,
            params,
            retry_429_once=retry_429_once,
        )

    def _request_host_json(
        self,
        host: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        retry_429_once: bool = True,
    ) -> Any:
        query = urllib.parse.urlencode(params or {})
        url = f"https://{host}{path}"
        if query:
            url = f"{url}?{query}"

        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "lol-champ-select-recommender/0.1",
                "X-Riot-Token": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and retry_429_once:
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
                time.sleep(retry_after)
                return self._request_host_json(host, path, params, retry_429_once=False)
            raise RiotApiError(_format_http_error(exc.code, url, body)) from exc
        except OSError as exc:
            raise RiotApiError(f"Could not reach Riot API at {url}: {exc}") from exc

        if not body:
            return None

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RiotApiError(f"Riot API returned non-JSON data from {url}.") from exc


def normalize_platform(value: str) -> str:
    platform = value.lower()
    if platform not in VALID_PLATFORMS:
        valid = ", ".join(sorted(VALID_PLATFORMS))
        raise RiotApiError(f"Invalid Riot platform routing value '{value}'. Use one of: {valid}.")
    return platform


def region_for_platform(platform: str) -> str:
    return PLATFORM_TO_REGION[normalize_platform(platform)]


def parse_riot_id(value: str) -> tuple[str, str]:
    if "#" not in value:
        raise ValueError('Riot ID must look like "GameName#TAG".')
    game_name, tag_line = value.rsplit("#", 1)
    game_name = game_name.strip()
    tag_line = tag_line.strip()
    if not game_name or not tag_line:
        raise ValueError('Riot ID must look like "GameName#TAG".')
    return game_name, tag_line


def _retry_after_seconds(value: str | None) -> int:
    if not value:
        return 2
    try:
        return max(1, min(30, int(value)))
    except ValueError:
        return 2


def _format_http_error(status: int, url: str, body: str) -> str:
    details = body.strip()
    if status == 401:
        hint = "Check RIOT_API_KEY."
    elif status == 403:
        hint = "Your Riot API key may be expired or not allowed for this endpoint."
    elif status == 404:
        hint = "The requested Riot resource was not found."
    elif status == 429:
        hint = "Riot API rate limit hit."
    else:
        hint = "Riot API request failed."

    return f"{hint} HTTP {status} for {url}: {details}"
