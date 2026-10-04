import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lol_champ_select_recommender.stats_recommendations import ban_recommendation_lines


class BanRecommendationsTest(unittest.TestCase):
    def test_performance_thresholds_queue_filter_and_no_unseen_bans(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "enemy.csv"
            path.write_text(
                "player,champion_id,role,queue_id,games,wins,losses\n"
                "alice,1,top,420,20,10,10\n"
                "alice,2,top,420,19,9,10\n"
                "alice,3,top,420,19,10,9\n"
                "alice,4,top,440,50,0,50\n"
            )
            args = SimpleNamespace(opponent_stats=path, profiles=["alice"], ranked_queue="soloduo", recommendation_count=100)
            features = {cid: {"champion_name": f"Champ{cid}"} for cid in range(1, 6)}
            lines = "\n".join(ban_recommendation_lines(args, features))
        soft = lines.split("    Soft:\n", 1)[1].split("    Hard:", 1)[0]
        extrapolated = lines.split("    Extrapolated Soft:\n", 1)[1].split("    Extrapolated Hard:", 1)[0]
        self.assertIn("Champ1:", soft)
        self.assertNotIn("Champ2:", soft)
        self.assertIn("Champ2:", extrapolated)
        for name in ("Champ3:", "Champ4:", "Champ5:"):
            self.assertNotIn(name, lines)
