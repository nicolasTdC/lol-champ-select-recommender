import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from lol_champ_select_recommender.collect_player_stats import collect_player_stats, _load_match, accumulate_opponents, preserve_other_stats


class CollectPlayerStatsTest(unittest.TestCase):
    def test_aram_records_without_inferred_lanes(self):
        from lol_champ_select_recommender.collect_player_stats import _accumulate_player_match

        match = {"info": {"queueId": 450, "participants": [
            {"puuid": "me", "participantId": 1, "championId": 1, "teamPosition": "", "win": True}
        ]}}
        aggregates = {}
        with patch("lol_champ_select_recommender.collect_player_stats._participant_roles") as infer:
            _accumulate_player_match(aggregates, match, "me", "alice", Mock())
        infer.assert_not_called()
        self.assertEqual(list(aggregates), [("alice", 1, "aram", 450)])
        self.assertEqual(next(iter(aggregates.values()))["wins"], 1)

    def test_incremental_preserves_other_profiles_and_queues(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            path.write_text("player,queue_id,games\nalice,420,10\nalice,440,20\nbob,420,30\n")
            rows = preserve_other_stats(path, [{"player": "alice", "queue_id": 420, "games": 11}], ["alice"], [420])
        self.assertEqual([(r["player"], int(r["queue_id"]), int(r["games"])) for r in rows],
                         [("alice", 420, 11), ("alice", 440, 20), ("bob", 420, 30)])

    def test_incremental_stops_at_known_match_and_rebuilds_without_duplicates(self):
        import json

        client = Mock()
        client.account_by_riot_id.return_value = {"puuid": "player"}
        client.match_ids_by_puuid.return_value = ["BR1_new", "BR1_old"]
        participant = {"puuid": "player", "championId": 1, "participantId": 1, "teamId": 100, "win": True}
        match = {"info": {"queueId": 420, "participants": [participant]}}
        client.match_by_id.return_value = match
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "BR1_old.json").write_text(json.dumps(match))
            (cache / "BR1_older.json").write_text(json.dumps(match))
            with patch("lol_champ_select_recommender.collect_player_stats._participant_roles", return_value={1: "middle"}):
                rows = collect_player_stats(client, Mock(), riot_ids=["Alice#BR1"], region="americas", queue=[420], match_type="ranked", matches_per_player=1000, sleep_seconds=0, incremental=True, matches_dir=cache)
        self.assertEqual(client.match_ids_by_puuid.call_count, 1)
        client.match_by_id.assert_called_once_with("BR1_new", "americas")
        self.assertEqual(rows[0]["games"], 3)

    def test_enemy_stats_record_personal_outcome_and_exclude_allies(self):
        player = {"puuid": "me", "teamId": 100, "participantId": 1, "championId": 1, "win": False}
        ally = {"puuid": "ally", "teamId": 100, "participantId": 2, "championId": 2, "win": False}
        enemy = {"puuid": "enemy", "teamId": 200, "participantId": 6, "championId": 3, "win": True}
        match = {"info": {"queueId": 440, "participants": [player, ally, enemy]}}
        aggregates = {}
        with patch("lol_champ_select_recommender.collect_player_stats._participant_roles", return_value={1: "top", 2: "middle", 6: "utility"}):
            accumulate_opponents(aggregates, match, "me", "Alice#BR1", Mock())
        self.assertEqual(list(aggregates), [("Alice#BR1", 3, "utility", 440)])
        row = next(iter(aggregates.values()))
        self.assertEqual((row["games"], row["wins"], row["losses"]), (1, 0, 1))
        self.assertTrue(enemy["win"])

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
