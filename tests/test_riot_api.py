from __future__ import annotations

import contextlib
import io
import json
import urllib.error
import urllib.request
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lol_champ_select_recommender.collect_ranked_matches import (
    collect_ladder_entries,
    download_match,
    resolve_download_workers,
    select_seed_players,
)
from lol_champ_select_recommender.riot_api import RiotApiClient, RiotApiError, parse_riot_id, region_for_platform


class RiotApiTest(unittest.TestCase):
    def test_parse_riot_id(self) -> None:
        self.assertEqual(parse_riot_id("Some Name#BR1"), ("Some Name", "BR1"))

    def test_parse_riot_id_allows_hash_in_game_name(self) -> None:
        self.assertEqual(parse_riot_id("Name # With Hash#NA1"), ("Name # With Hash", "NA1"))

    def test_parse_riot_id_rejects_missing_tag(self) -> None:
        with self.assertRaises(ValueError):
            parse_riot_id("OnlyName")

    def test_region_for_platform(self) -> None:
        self.assertEqual(region_for_platform("br1"), "americas")
        self.assertEqual(region_for_platform("KR"), "asia")
        self.assertEqual(region_for_platform("euw1"), "europe")

    def test_region_for_platform_rejects_unknown_platform(self) -> None:
        with self.assertRaises(RiotApiError):
            region_for_platform("bad")

    def test_select_seed_players_is_deterministic(self) -> None:
        entries = [{"puuid": str(index)} for index in range(10)]

        self.assertEqual(select_seed_players(entries, 3, seed=7), select_seed_players(entries, 3, seed=7))
        self.assertEqual(len(select_seed_players(entries, 3, seed=7)), 3)

    def test_collect_ladder_entries_uses_apex_endpoint_for_master(self) -> None:
        client = FakeRiotClient()
        args = Namespace(
            platform="br1",
            queue="RANKED_SOLO_5x5",
            tiers=["MASTER"],
            divisions=["I"],
            pages=1,
        )

        entries = collect_ladder_entries(client, args)

        self.assertEqual(entries, [{"puuid": "master-puuid", "tier": "MASTER"}])
        self.assertEqual(client.apex_calls, [("br1", "MASTER", "RANKED_SOLO_5x5")])
        self.assertEqual(client.standard_calls, [])

    def test_collect_ladder_entries_uses_standard_endpoint_for_diamond(self) -> None:
        client = FakeRiotClient()
        args = Namespace(
            platform="br1",
            queue="RANKED_SOLO_5x5",
            tiers=["DIAMOND"],
            divisions=["I"],
            pages=1,
        )

        entries = collect_ladder_entries(client, args)

        self.assertEqual(entries, [{"puuid": "diamond-puuid"}])
        self.assertEqual(client.standard_calls, [("br1", "RANKED_SOLO_5x5", "DIAMOND", "I", 1)])
        self.assertEqual(client.apex_calls, [])

    def test_request_host_json_retries_429_before_succeeding(self) -> None:
        client = RiotApiClient(api_key="test-key", timeout=0.1)
        response = FakeHttpResponse(b'{"ok": true}')
        error = urllib.error.HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{"status":{"status_code":429}}'),
        )

        with patch.object(urllib.request, "urlopen", side_effect=[error, response]) as mocked_urlopen, patch(
            "lol_champ_select_recommender.riot_api.time.sleep"
        ) as mocked_sleep:
            data = client._request_host_json("example.com", "/test")

        self.assertEqual(data, {"ok": True})
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called()

    def test_request_host_json_logs_429_backoff_when_enabled(self) -> None:
        client = RiotApiClient(api_key="test-key", timeout=0.1, log_rate_limits=True)
        response = FakeHttpResponse(b'{"ok": true}')
        error = urllib.error.HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            {"Retry-After": "0"},
            io.BytesIO(b'{"status":{"status_code":429}}'),
        )
        stderr = io.StringIO()

        with patch.object(urllib.request, "urlopen", side_effect=[error, response]), patch(
            "lol_champ_select_recommender.riot_api.time.sleep"
        ), contextlib.redirect_stderr(stderr):
            data = client._request_host_json("example.com", "/test")

        self.assertEqual(data, {"ok": True})
        self.assertIn("rate-limit", stderr.getvalue())
        self.assertIn("backing off", stderr.getvalue())

    def test_download_match_writes_file_and_skips_existing_without_force(self) -> None:
        client = FakeMatchClient()
        with TemporaryDirectory() as tmpdir:
            match_path = Path(tmpdir) / "BR1_1.json"

            status, error = download_match(client, "BR1_1", "americas", match_path, force=False)
            self.assertEqual((status, error), ("downloaded", None))
            self.assertEqual(json.loads(match_path.read_text(encoding="utf-8")), {"info": {"gameVersion": "16.13"}})
            self.assertEqual(client.calls, [("BR1_1", "americas")])

            status, error = download_match(client, "BR1_1", "americas", match_path, force=False)
            self.assertEqual((status, error), ("existing", None))
            self.assertEqual(client.calls, [("BR1_1", "americas")])

    def test_resolve_download_workers_auto_scales_with_job_size(self) -> None:
        self.assertEqual(resolve_download_workers("auto", 0), 1)
        self.assertEqual(resolve_download_workers("auto", 10), 2)
        self.assertEqual(resolve_download_workers("auto", 100), 4)
        self.assertEqual(resolve_download_workers("auto", 500), 8)
        self.assertEqual(resolve_download_workers("3", 500), 3)


class FakeRiotClient:
    def __init__(self) -> None:
        self.apex_calls = []
        self.standard_calls = []

    def apex_league_entries(self, platform: str, *, tier: str, queue: str):
        self.apex_calls.append((platform, tier, queue))
        return [{"puuid": "master-puuid", "tier": tier}]

    def league_entries(self, platform: str, *, queue: str, tier: str, division: str, page: int):
        self.standard_calls.append((platform, queue, tier, division, page))
        return [{"puuid": "diamond-puuid"}]


class FakeMatchClient:
    def __init__(self) -> None:
        self.calls = []

    def match_by_id(self, match_id: str, region: str):
        self.calls.append((match_id, region))
        return {"info": {"gameVersion": "16.13"}}


class FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload


if __name__ == "__main__":
    unittest.main()
