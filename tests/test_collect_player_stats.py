import unittest
from unittest.mock import Mock, patch

from lol_champ_select_recommender.collect_player_stats import collect_player_stats


class CollectPlayerStatsTest(unittest.TestCase):
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
