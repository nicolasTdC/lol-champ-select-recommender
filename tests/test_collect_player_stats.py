import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from lol_champ_select_recommender.collect_player_stats import collect_player_stats, _load_match


class CollectPlayerStatsTest(unittest.TestCase):
    def test_concurrent_downloads_are_cached_and_reused(self):
        from concurrent.futures import ThreadPoolExecutor

        client = Mock()
        client.match_by_id.side_effect = lambda match_id, region: {"metadata": {"matchId": match_id}, "info": {"queueId": 420}}
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(_load_match, client, mid, "americas", cache, 0) for mid in ("BR1_1", "BR1_2")]
                results = [future.result() for future in futures]
            self.assertEqual({match["metadata"]["matchId"] for match, _ in results}, {"BR1_1", "BR1_2"})
            self.assertTrue(all(status == "downloaded" for _, status in results))
            match, status = _load_match(client, "BR1_1", "americas", cache, 0)
            self.assertEqual(status, "existing")
            self.assertEqual(match["metadata"]["matchId"], "BR1_1")
            self.assertEqual(client.match_by_id.call_count, 2)

    def test_paginates_each_queue_and_stops_at_short_page(self):
        client = Mock()
        client.account_by_riot_id.return_value = {"puuid": "player"}
        client.match_ids_by_puuid.side_effect = [
            [f"solo{i}" for i in range(100)],
            [f"solo{i}" for i in range(100, 200)],
            ["solo200"],
            [],
        ]
        client.match_by_id.return_value = {"info": {"queueId": 420, "participants": []}}
        collect_player_stats(client, Mock(), riot_ids=["Alice#BR1"], region="americas", queue=[420, 440], match_type="ranked", matches_per_player=250, sleep_seconds=0)
        self.assertEqual(
            [(call.kwargs["queue"], call.kwargs["start"], call.kwargs["count"]) for call in client.match_ids_by_puuid.call_args_list],
            [(420, 0, 100), (420, 100, 100), (420, 200, 50), (440, 0, 100)],
        )
        self.assertEqual(client.match_by_id.call_count, 201)

    def test_collects_both_queues_without_merging_queue_rows(self):
        client = Mock()
        client.account_by_riot_id.return_value = {"puuid": "player"}
        client.match_ids_by_puuid.side_effect = [["solo"], ["flex"]]
        participant = {"puuid": "player", "championId": 1, "participantId": 1, "win": True}
        client.match_by_id.side_effect = [
            {"info": {"queueId": queue, "participants": [participant]}}
            for queue in (420, 440)
        ]
        static = Mock()
        static.champion_name.return_value = "Annie"
        with patch("lol_champ_select_recommender.collect_player_stats._participant_roles", return_value={1: "middle"}):
            rows = collect_player_stats(client, static, riot_ids=["Alice#BR1"], region="americas", queue=[420, 440], match_type="ranked", matches_per_player=100, sleep_seconds=0)
        self.assertEqual([call.kwargs["queue"] for call in client.match_ids_by_puuid.call_args_list], [420, 440])
        self.assertEqual([(row["queue_id"], row["games"]) for row in rows], [(420, 1), (440, 1)])
