from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lol_champ_select_recommender.modeling.player_pruning import (
    load_player_prune_index,
    prune_candidates,
    extrapolated_hard_prune_candidates,
    extrapolated_soft_prune_candidates,
)


class PlayerPruningTest(unittest.TestCase):
    def test_prune_rules_use_soft_and_hard_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player_stats.csv"
            path.write_text(
                "\n".join(
                    [
                        "player,champion_id,champion_name,role,role_name,games,wins,losses,win_rate",
                        "alice,1,Annie,top,Top,12,7,5,0.5833",
                        "bob,1,Annie,top,Top,8,4,4,0.5000",
                        "alice,1,Annie,jungle,Jungle,3,1,2,0.3333",
                        "bob,2,Olaf,top,Top,21,10,11,0.4762",
                        "bob,3,Galio,top,Top,19,8,11,0.4211",
                        "bob,4,Shen,top,Top,19,10,9,0.5263",
                        "bob,5,Sett,top,Top,19,8,11,0.4211",
                        "bob,1,Annie,utility,Support,1,1,0,1.0000",
                    ]
                ),
                encoding="utf-8",
            )

            index = load_player_prune_index(path)

        self.assertIsNotNone(index)
        assert index is not None
        self.assertTrue(index.passes_soft(1))
        self.assertTrue(index.passes_hard(1, "top"))
        self.assertFalse(index.passes_hard(1, "jungle"))
        self.assertFalse(index.passes_soft(2))
        self.assertFalse(index.passes_hard(2, "top"))
        self.assertFalse(index.passes_soft(3))
        self.assertFalse(index.passes_hard(3, "top"))
        self.assertFalse(index.passes_soft(4))
        self.assertFalse(index.passes_hard(4, "top"))
        self.assertFalse(index.passes_soft(5))
        self.assertFalse(index.passes_hard(5, "top"))
        self.assertFalse(index.passes_soft(6))
        self.assertFalse(index.passes_hard(6, "top"))
        self.assertTrue(index.passes_soft_extrapolated(4))
        self.assertTrue(index.passes_hard_extrapolated(4, "top"))
        self.assertFalse(index.passes_soft_extrapolated(5))
        self.assertFalse(index.passes_hard_extrapolated(5, "top"))
        self.assertTrue(index.passes_soft_extrapolated(6))
        self.assertTrue(index.passes_hard_extrapolated(6, "top"))
        self.assertEqual(prune_candidates([1, 2, 3], role="top", prune_index=index), [1])
        self.assertEqual(extrapolated_soft_prune_candidates([1, 2, 3, 4, 5, 6], prune_index=index), [1, 4, 6])
        self.assertEqual(extrapolated_hard_prune_candidates([1, 2, 3, 4, 5, 6], role="top", prune_index=index), [1, 4, 6])

    def test_lane_recommendations_use_role_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player_stats.csv"
            path.write_text(
                "\n".join(
                    [
                        "player,champion_id,champion_name,role,role_name,games,wins,losses,win_rate",
                        "alice,1,Annie,utility,Support,12,8,4,0.6667",
                        "alice,2,Olaf,utility,Support,10,5,5,0.5000",
                        "alice,3,Galio,middle,Mid,8,4,4,0.5000",
                        "alice,4,Shen,jungle,Jungle,10,7,3,0.7000",
                        "alice,5,Sett,top,Top,18,8,10,0.4444",
                    ]
                ),
                encoding="utf-8",
            )

            index = load_player_prune_index(path)

        self.assertIsNotNone(index)
        assert index is not None
        self.assertEqual([role for role, _stats in index.hard_lane_recommendations()], ["utility"])
        self.assertEqual([role for role, _stats in index.soft_lane_recommendations()], ["jungle", "middle"])


if __name__ == "__main__":
    unittest.main()
