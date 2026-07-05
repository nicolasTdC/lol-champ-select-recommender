from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lol_champ_select_recommender.aggregate_matches import aggregate_match_files, build_role_priors
from lol_champ_select_recommender.ddragon import StaticData


class AggregateMatchesTest(unittest.TestCase):
    def test_aggregate_match_files_counts_champion_role_wins(self) -> None:
        static_data = StaticData(
            version="test",
            champions={82: "Mordekaiser", 267: "Nami"},
            summoner_spells={},
            champion_keys={82: "Mordekaiser", 267: "Nami"},
        )
        match = {
            "info": {
                "gameVersion": "16.13.123.456",
                "queueId": 420,
                "participants": [
                    {
                        "participantId": 1,
                        "teamId": 100,
                        "championId": 82,
                        "teamPosition": "TOP",
                        "summoner1Id": 4,
                        "summoner2Id": 12,
                        "win": True,
                    },
                    {
                        "participantId": 2,
                        "teamId": 100,
                        "championId": 267,
                        "teamPosition": "UTILITY",
                        "summoner1Id": 4,
                        "summoner2Id": 14,
                        "win": False,
                    },
                ],
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BR1_1.json"
            path.write_text(json.dumps(match), encoding="utf-8")

            rows = aggregate_match_files(Path(directory), static_data)

        morde = next(row for row in rows if row["champion_id"] == 82)
        nami = next(row for row in rows if row["champion_id"] == 267)

        self.assertEqual(morde["patch"], "16.13")
        self.assertEqual(morde["role"], "top")
        self.assertEqual(morde["games"], 1)
        self.assertEqual(morde["wins"], 1)
        self.assertEqual(morde["win_rate"], 1.0)
        self.assertEqual(nami["role"], "utility")
        self.assertEqual(nami["win_rate"], 0.0)

    def test_build_role_priors_computes_role_share(self) -> None:
        stats = [
            {
                "patch": "16.13",
                "queue_id": 420,
                "role": "top",
                "role_name": "Top",
                "champion_id": 57,
                "champion_name": "Maokai",
                "games": 3,
                "wins": 1,
                "win_rate": 0.3333,
            },
            {
                "patch": "16.13",
                "queue_id": 420,
                "role": "jungle",
                "role_name": "Jungle",
                "champion_id": 57,
                "champion_name": "Maokai",
                "games": 7,
                "wins": 4,
                "win_rate": 0.5714,
            },
        ]

        priors = build_role_priors(stats)

        jungle = next(row for row in priors if row["role"] == "jungle")
        top = next(row for row in priors if row["role"] == "top")
        self.assertEqual(jungle["total_games"], 10)
        self.assertEqual(jungle["role_share"], 0.7)
        self.assertEqual(top["role_share"], 0.3)


if __name__ == "__main__":
    unittest.main()
