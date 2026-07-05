from __future__ import annotations

import unittest
from argparse import Namespace

from lol_champ_select_recommender.collect_ranked_matches import collect_ladder_entries, select_seed_players
from lol_champ_select_recommender.riot_api import RiotApiError, parse_riot_id, region_for_platform


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


if __name__ == "__main__":
    unittest.main()
