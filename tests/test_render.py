from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from lol_champ_select_recommender.ddragon import StaticData
from lol_champ_select_recommender.lcu import LcuConnection, connection_hosts
from lol_champ_select_recommender.render import render_session
from lol_champ_select_recommender.roles import assign_roles, champion_role_weights, load_role_priors


class RenderSessionTest(unittest.TestCase):
    def test_render_champ_select_snapshot(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 3: "Galio", 99: "Lux"},
            summoner_spells={4: "Flash", 14: "Ignite", 7: "Heal"},
        )
        session = {
            "localPlayerCellId": 1,
            "timer": {
                "phase": "PLANNING",
                "adjustedTimeLeftInPhase": 27000,
                "totalTimeInPhase": 30000,
            },
            "actions": [
                [
                    {
                        "actorCellId": 1,
                        "championId": 1,
                        "isAllyAction": True,
                        "isInProgress": True,
                        "type": "pick",
                    }
                ]
            ],
            "myTeam": [
                {
                    "cellId": 1,
                    "assignedPosition": "middle",
                    "championId": 1,
                    "championPickIntent": 99,
                    "spell1Id": 4,
                    "spell2Id": 14,
                }
            ],
            "theirTeam": [
                {
                    "cellId": 6,
                    "assignedPosition": "top",
                    "championId": 2,
                    "spell1Id": 4,
                    "spell2Id": 7,
                }
            ],
            "bans": {
                "myTeamBans": [3],
                "theirTeamBans": [99],
            },
        }

        output = render_session(
            phase="ChampSelect",
            session=session,
            static_data=static_data,
            lockfile_label="test-lockfile",
        )

        self.assertIn("Gameflow: ChampSelect", output)
        self.assertIn("Annie", output)
        self.assertIn("Lux", output)
        self.assertIn("Flash", output)
        self.assertIn("Ignite", output)
        self.assertIn("Ally bans:  Galio", output)
        self.assertIn("Enemy bans: Lux", output)

    def test_render_infers_enemy_roles(self) -> None:
        static_data = StaticData(
            version="test",
            champions={1: "Annie", 2: "Olaf", 22: "Ashe", 86: "Garen", 99: "Lux"},
            summoner_spells={4: "Flash", 7: "Heal", 11: "Smite", 14: "Ignite"},
            champion_keys={1: "Annie", 2: "Olaf", 22: "Ashe", 86: "Garen", 99: "Lux"},
        )
        session = {
            "localPlayerCellId": 1,
            "timer": {},
            "actions": [],
            "myTeam": [],
            "theirTeam": [
                {"cellId": 6, "championId": 86, "spell1Id": 4, "spell2Id": 14},
                {"cellId": 7, "championId": 2, "spell1Id": 4, "spell2Id": 11},
                {"cellId": 8, "championId": 1, "spell1Id": 4, "spell2Id": 14},
                {"cellId": 9, "championId": 22, "spell1Id": 4, "spell2Id": 7},
                {"cellId": 10, "championId": 99, "spell1Id": 4, "spell2Id": 14},
            ],
            "bans": {},
        }

        output = render_session(
            phase="ChampSelect",
            session=session,
            static_data=static_data,
            lockfile_label="test-lockfile",
        )

        self.assertIn("Top?", output)
        self.assertIn("Jungle?", output)
        self.assertIn("Mid?", output)
        self.assertIn("Bot?", output)
        self.assertIn("Support?", output)
        self.assertIn("? = inferred from champion/spells", output)


class RoleAssignmentTest(unittest.TestCase):
    def test_smite_forces_enemy_jungle_when_available(self) -> None:
        static_data = StaticData(
            version="test",
            champions={2: "Olaf"},
            summoner_spells={11: "Smite"},
            champion_keys={2: "Olaf"},
        )
        assignments = assign_roles(
            [{"cellId": 7, "championId": 2, "spell1Id": 4, "spell2Id": 11}],
            static_data,
            infer_missing=True,
        )

        self.assertEqual(assignments[7].position, "jungle")
        self.assertTrue(assignments[7].inferred)

    def test_global_assignment_does_not_orphan_mordekaiser(self) -> None:
        static_data = StaticData(
            version="test",
            champions={57: "Maokai", 82: "Mordekaiser", 85: "Kennen", 145: "Kai'Sa", 267: "Nami"},
            summoner_spells={},
            champion_keys={
                57: "Maokai",
                82: "Mordekaiser",
                85: "Kennen",
                145: "Kaisa",
                267: "Nami",
            },
        )
        assignments = assign_roles(
            [
                {"cellId": 5, "championId": 267},
                {"cellId": 6, "championId": 57},
                {"cellId": 7, "championId": 82},
                {"cellId": 8, "championId": 85},
                {"cellId": 9, "championId": 145},
            ],
            static_data,
            infer_missing=True,
        )

        self.assertEqual(assignments[5].position, "utility")
        self.assertEqual(assignments[6].position, "jungle")
        self.assertEqual(assignments[7].position, "top")
        self.assertEqual(assignments[8].position, "middle")
        self.assertEqual(assignments[9].position, "bottom")

    def test_load_role_priors_filters_latest_patch_queue_and_min_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priors.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "patch",
                        "queue_id",
                        "champion_id",
                        "champion_name",
                        "total_games",
                        "role",
                        "role_name",
                        "games",
                        "wins",
                        "win_rate",
                        "role_share",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patch": "16.12",
                        "queue_id": 420,
                        "champion_id": 57,
                        "champion_name": "Maokai",
                        "total_games": 20,
                        "role": "top",
                        "role_name": "Top",
                        "games": 20,
                        "wins": 10,
                        "win_rate": 0.5,
                        "role_share": 1.0,
                    }
                )
                writer.writerow(
                    {
                        "patch": "16.13",
                        "queue_id": 420,
                        "champion_id": 57,
                        "champion_name": "Maokai",
                        "total_games": 10,
                        "role": "jungle",
                        "role_name": "Jungle",
                        "games": 7,
                        "wins": 4,
                        "win_rate": 0.5714,
                        "role_share": 0.7,
                    }
                )
                writer.writerow(
                    {
                        "patch": "16.13",
                        "queue_id": 420,
                        "champion_id": 57,
                        "champion_name": "Maokai",
                        "total_games": 10,
                        "role": "utility",
                        "role_name": "Support",
                        "games": 3,
                        "wins": 1,
                        "win_rate": 0.3333,
                        "role_share": 0.3,
                    }
                )

            priors = load_role_priors(path, queue_id=420, min_total_games=5)

        self.assertIsNotNone(priors)
        assert priors is not None
        self.assertEqual(priors.patch, "16.13")
        self.assertEqual(priors.role_weights(57), {"jungle": 700, "utility": 300})

    def test_champion_role_weights_prefers_loaded_priors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priors.csv"
            path.write_text(
                "\n".join(
                    [
                        "patch,queue_id,champion_id,champion_name,total_games,role,role_name,games,wins,win_rate,role_share",
                        "16.13,420,57,Maokai,10,jungle,Jungle,10,5,0.5,1.0",
                    ]
                ),
                encoding="utf-8",
            )
            priors = load_role_priors(path, queue_id=420, min_total_games=5)

        self.assertIsNotNone(priors)
        self.assertEqual(
            champion_role_weights(champion_id=57, champion_key="Maokai", role_priors=priors),
            {"jungle": 1000},
        )


class LcuHostTest(unittest.TestCase):
    def test_explicit_host_takes_precedence(self) -> None:
        self.assertEqual(connection_hosts("192.0.2.10"), ["192.0.2.10"])

    def test_windows_curl_connection_uses_windows_loopback(self) -> None:
        connection = LcuConnection(port=1234, password="secret", host="10.0.0.1").with_windows_curl()

        self.assertEqual(connection.host, "127.0.0.1")
        self.assertEqual(connection.transport, "windows-curl")


if __name__ == "__main__":
    unittest.main()
