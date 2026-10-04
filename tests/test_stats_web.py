import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lol_champ_select_recommender.stats_web import stats_payload


class StatsWebTest(unittest.TestCase):
    def test_scope_filters_complete_lists_and_queue(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            path.write_text("player,champion_id,champion_name,role,queue_id,games,wins,losses\n"
                            "alice,1,Annie,middle,420,20,12,8\n"
                            "alice,2,Olaf,top,440,30,5,25\n")
            args = SimpleNamespace(player_stats=path, opponent_stats=path, champion_features=Path(directory)/"missing.csv", champion_blacklist=Path(directory)/"blacklist.txt", profiles=None, ranked_queue="all")
            data = stats_payload(args, {"queue": ["soloduo"]})
        overall = data["scopes"]["all"]["picks"][0]
        mid = data["scopes"]["middle"]["picks"][0]
        top = data["scopes"]["top"]["picks"][0]
        self.assertEqual([r["id"] for r in overall["recommended"]], [1])
        self.assertEqual([r["id"] for r in mid["recommended"]], [1])
        self.assertEqual(top["recommended"], [])
        self.assertEqual(len(top["rejected"]), 2)
        self.assertTrue(next(r for r in data["lanes"] if r["role"] == "middle")["soft"])
