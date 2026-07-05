from __future__ import annotations

import unittest

from lol_champ_select_recommender.ddragon import StaticData
from lol_champ_select_recommender.modeling.draft_data import build_model_vocab, champion_features_by_id
from lol_champ_select_recommender.modeling.draft_inference import (
    DraftPickRecommendation,
    DraftRecommender,
    DraftRoleRecommendation,
    build_live_queries,
    infer_my_side,
)
from lol_champ_select_recommender.modeling.player_pruning import PlayerPruneIndex, PruneStats


class DraftInferenceTest(unittest.TestCase):
    def test_infer_my_side_uses_local_cell_id(self) -> None:
        self.assertEqual(infer_my_side({"localPlayerCellId": 2}), "blue")
        self.assertEqual(infer_my_side({"localPlayerCellId": 8}), "red")

    def test_build_live_queries_returns_open_ally_roles(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 3: "Galio", 4: "Twisted Fate", 5: "Xin Zhao", 99: "Lux"},
            summoner_spells={},
            champion_keys={1: "Annie", 2: "Olaf", 3: "Galio", 4: "TwistedFate", 5: "XinZhao", 99: "Lux"},
        )
        draft_rows = [
            {
                "patch": "16.13",
                "queue_id": 420,
                "winning_side": "blue",
                "blue": {"top": 1, "jungle": None, "middle": None, "bottom": None, "utility": None},
                "red": {"top": 2, "jungle": None, "middle": None, "bottom": None, "utility": None},
                "blue_bans": [3, -1, -1, -1, -1],
                "red_bans": [99, -1, -1, -1, -1],
            }
        ]
        feature_rows = [
            {
                "champion_id": "1",
                "champion_key": "Annie",
                "champion_name": "Annie",
                "primary_tag": "Mage",
                "secondary_tag": "<NONE>",
                "partype": "Mana",
                "range_type": "ranged",
                "info_attack": "2",
                "info_defense": "3",
                "info_magic": "10",
                "info_difficulty": "7",
                "stat_attackrange": "625",
                "stat_hp": "1200",
            },
            {
                "champion_id": "2",
                "champion_key": "Olaf",
                "champion_name": "Olaf",
                "primary_tag": "Fighter",
                "secondary_tag": "<NONE>",
                "partype": "Mana",
                "range_type": "melee",
                "info_attack": "8",
                "info_defense": "5",
                "info_magic": "2",
                "info_difficulty": "4",
                "stat_attackrange": "175",
                "stat_hp": "1200",
            },
            {
                "champion_id": "3",
                "champion_key": "Galio",
                "champion_name": "Galio",
                "primary_tag": "Tank",
                "secondary_tag": "Mage",
                "partype": "Mana",
                "range_type": "melee",
                "info_attack": "3",
                "info_defense": "8",
                "info_magic": "7",
                "info_difficulty": "5",
                "stat_attackrange": "150",
                "stat_hp": "1200",
            },
            {
                "champion_id": "4",
                "champion_key": "TwistedFate",
                "champion_name": "Twisted Fate",
                "primary_tag": "Mage",
                "secondary_tag": "<NONE>",
                "partype": "Mana",
                "range_type": "ranged",
                "info_attack": "4",
                "info_defense": "2",
                "info_magic": "9",
                "info_difficulty": "6",
                "stat_attackrange": "525",
                "stat_hp": "1200",
            },
            {
                "champion_id": "5",
                "champion_key": "XinZhao",
                "champion_name": "Xin Zhao",
                "primary_tag": "Fighter",
                "secondary_tag": "Assassin",
                "partype": "Mana",
                "range_type": "melee",
                "info_attack": "8",
                "info_defense": "6",
                "info_magic": "2",
                "info_difficulty": "4",
                "stat_attackrange": "175",
                "stat_hp": "1200",
            },
            {
                "champion_id": "99",
                "champion_key": "Lux",
                "champion_name": "Lux",
                "primary_tag": "Mage",
                "secondary_tag": "Support",
                "partype": "Mana",
                "range_type": "ranged",
                "info_attack": "2",
                "info_defense": "4",
                "info_magic": "8",
                "info_difficulty": "5",
                "stat_attackrange": "550",
                "stat_hp": "1200",
            },
        ]
        model_vocab = build_model_vocab(draft_rows, feature_rows, numeric_bins=4)
        champion_features = champion_features_by_id(feature_rows)
        session = {
            "localPlayerCellId": 1,
            "myTeam": [
                {"cellId": 1, "championId": 1, "assignedPosition": "top"},
                {"cellId": 2, "championId": 0},
                {"cellId": 3, "championId": 0},
                {"cellId": 4, "championId": 0},
                {"cellId": 5, "championId": 0},
            ],
            "theirTeam": [
                {"cellId": 6, "championId": 2, "assignedPosition": "jungle"},
                {"cellId": 7, "championId": 0},
                {"cellId": 8, "championId": 0},
                {"cellId": 9, "championId": 0},
                {"cellId": 10, "championId": 0},
            ],
            "bans": {
                "myTeamBans": [3],
                "theirTeamBans": [99],
            },
        }

        queries = build_live_queries(session, static_data, model_vocab, champion_features)

        self.assertEqual([query.role for query in queries], ["jungle", "middle", "bottom", "utility"])
        self.assertTrue(all(query.query_index in range(5) for query in queries))
        self.assertIn(1, queries[0].blocked_champion_ids)
        self.assertIn(2, queries[0].blocked_champion_ids)
        self.assertIn(3, queries[0].blocked_champion_ids)
        self.assertIn(99, queries[0].blocked_champion_ids)

    def test_debug_lines_include_token_dump(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 3: "Galio", 99: "Lux"},
            summoner_spells={},
            champion_keys={1: "Annie", 2: "Olaf", 3: "Galio", 99: "Lux"},
        )
        draft_rows = [
            {
                "patch": "16.13",
                "queue_id": 420,
                "winning_side": "blue",
                "blue": {"top": 1, "jungle": None, "middle": None, "bottom": None, "utility": None},
                "red": {"top": 2, "jungle": None, "middle": None, "bottom": None, "utility": None},
                "blue_bans": [3, -1, -1, -1, -1],
                "red_bans": [99, -1, -1, -1, -1],
            }
        ]
        feature_rows = [
            {
                "champion_id": "1",
                "champion_key": "Annie",
                "champion_name": "Annie",
                "primary_tag": "Mage",
                "secondary_tag": "<NONE>",
                "partype": "Mana",
                "range_type": "ranged",
                "info_attack": "2",
                "info_defense": "3",
                "info_magic": "10",
                "info_difficulty": "7",
                "stat_attackrange": "625",
                "stat_hp": "1200",
            },
            {
                "champion_id": "2",
                "champion_key": "Olaf",
                "champion_name": "Olaf",
                "primary_tag": "Fighter",
                "secondary_tag": "<NONE>",
                "partype": "Mana",
                "range_type": "melee",
                "info_attack": "8",
                "info_defense": "5",
                "info_magic": "2",
                "info_difficulty": "4",
                "stat_attackrange": "175",
                "stat_hp": "1200",
            },
            {
                "champion_id": "3",
                "champion_key": "Galio",
                "champion_name": "Galio",
                "primary_tag": "Tank",
                "secondary_tag": "Mage",
                "partype": "Mana",
                "range_type": "melee",
                "info_attack": "3",
                "info_defense": "8",
                "info_magic": "7",
                "info_difficulty": "5",
                "stat_attackrange": "150",
                "stat_hp": "1200",
            },
            {
                "champion_id": "99",
                "champion_key": "Lux",
                "champion_name": "Lux",
                "primary_tag": "Mage",
                "secondary_tag": "Support",
                "partype": "Mana",
                "range_type": "ranged",
                "info_attack": "2",
                "info_defense": "4",
                "info_magic": "8",
                "info_difficulty": "5",
                "stat_attackrange": "550",
                "stat_hp": "1200",
            },
        ]
        model_vocab = build_model_vocab(draft_rows, feature_rows, numeric_bins=4)
        champion_features = champion_features_by_id(feature_rows)
        session = {
            "localPlayerCellId": 1,
            "myTeam": [
                {"cellId": 1, "championId": 1, "assignedPosition": "top"},
                {"cellId": 2, "championId": 0},
                {"cellId": 3, "championId": 0},
                {"cellId": 4, "championId": 0},
                {"cellId": 5, "championId": 0},
            ],
            "theirTeam": [
                {"cellId": 6, "championId": 2, "assignedPosition": "jungle"},
                {"cellId": 7, "championId": 0},
                {"cellId": 8, "championId": 0},
                {"cellId": 9, "championId": 0},
                {"cellId": 10, "championId": 0},
            ],
            "bans": {
                "myTeamBans": [3],
                "theirTeamBans": [99],
            },
        }

        queries = build_live_queries(session, static_data, model_vocab, champion_features)
        self.assertTrue(queries)

    def test_recommend_lines_show_raw_soft_and_hard_lists(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 3: "Galio"},
            summoner_spells={},
            champion_keys={1: "Annie", 2: "Olaf", 3: "Galio"},
        )
        recommender = object.__new__(DraftRecommender)
        recommender.recommend = lambda *args, **kwargs: [  # type: ignore[assignment]
            DraftRoleRecommendation(
                role="top",
                raw=[
                    DraftPickRecommendation(champion_id=1, score=0.6),
                    DraftPickRecommendation(champion_id=2, score=0.3),
                ],
                soft=[DraftPickRecommendation(champion_id=1, score=0.7)],
                hard=[DraftPickRecommendation(champion_id=2, score=0.8)],
                extrapolated_soft=[DraftPickRecommendation(champion_id=1, score=0.9)],
                extrapolated_hard=[DraftPickRecommendation(champion_id=2, score=1.0)],
                whitelisted_soft=None,
                whitelisted_hard=None,
                whitelisted_extrapolated_soft=None,
                whitelisted_extrapolated_hard=None,
            )
        ]

        lines = recommender.recommend_lines({}, static_data)

        self.assertIn("Recommendations", lines)
        self.assertIn("  Top", lines)
        self.assertIn("    Raw: Annie 60%, Olaf 30%", lines)
        self.assertIn("    Soft: Annie 70%", lines)
        self.assertIn("    Hard: Olaf 80%", lines)
        self.assertIn("    Extrapolated Soft: Annie 90%", lines)
        self.assertIn("    Extrapolated Hard: Olaf 100%", lines)
        self.assertIn("    Whitelisted Soft: unavailable", lines)
        self.assertIn("    Whitelisted Hard: unavailable", lines)

    def test_recommend_lines_show_lane_recommendations_from_player_stats(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie"},
            summoner_spells={},
            champion_keys={1: "Annie"},
        )
        recommender = object.__new__(DraftRecommender)
        recommender.player_prune_index = PlayerPruneIndex(
            source="test.csv",
            overall_by_champion={},
            by_role_by_champion={},
            by_role={
                "utility": PruneStats(games=22, wins=12, losses=10),
                "jungle": PruneStats(games=10, wins=7, losses=3),
                "top": PruneStats(games=18, wins=8, losses=10),
            },
        )
        recommender.recommend = lambda *args, **kwargs: [  # type: ignore[assignment]
            DraftRoleRecommendation(
                role="top",
                raw=[DraftPickRecommendation(champion_id=1, score=1.0)],
                soft=None,
                hard=None,
                extrapolated_soft=None,
                extrapolated_hard=None,
                whitelisted_soft=None,
                whitelisted_hard=None,
                whitelisted_extrapolated_soft=None,
                whitelisted_extrapolated_hard=None,
            )
        ]

        lines = recommender.recommend_lines({}, static_data)

        self.assertIn("  Lane", lines)
        self.assertIn("    Hard: Support 55% (22g)", lines)
        self.assertIn("    Soft: Jungle 70% (10g)", lines)

    def test_recommend_lines_show_whitelisted_views(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 3: "Galio"},
            summoner_spells={},
            champion_keys={1: "Annie", 2: "Olaf", 3: "Galio"},
        )
        recommender = object.__new__(DraftRecommender)
        recommender.champion_blacklist = {2}
        recommender.recommend = lambda *args, **kwargs: [  # type: ignore[assignment]
            DraftRoleRecommendation(
                role="top",
                raw=[
                    DraftPickRecommendation(champion_id=1, score=0.6),
                    DraftPickRecommendation(champion_id=2, score=0.3),
                ],
                soft=[DraftPickRecommendation(champion_id=1, score=0.7), DraftPickRecommendation(champion_id=2, score=0.2)],
                hard=[DraftPickRecommendation(champion_id=2, score=0.8)],
                extrapolated_soft=[DraftPickRecommendation(champion_id=1, score=0.9)],
                extrapolated_hard=[DraftPickRecommendation(champion_id=2, score=1.0)],
                whitelisted_soft=[DraftPickRecommendation(champion_id=1, score=1.0)],
                whitelisted_hard=[],
                whitelisted_extrapolated_soft=[DraftPickRecommendation(champion_id=1, score=1.0)],
                whitelisted_extrapolated_hard=[],
            )
        ]

        lines = recommender.recommend_lines({}, static_data)

        self.assertIn("    Whitelisted Soft: Annie 100%", lines)
        self.assertIn("    Whitelisted Hard: ", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
