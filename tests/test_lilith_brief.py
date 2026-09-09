from __future__ import annotations

import unittest

from lol_champ_select_recommender.ddragon import StaticData
from lol_champ_select_recommender.lilith_brief import format_lilith_brief


class LilithBriefTest(unittest.TestCase):
    def test_not_in_champ_select(self) -> None:
        text = format_lilith_brief(
            phase="Lobby",
            session=None,
            static_data=StaticData(version="1", champions={}, summoner_spells={}),
        )
        self.assertIn("Not in champion select", text)

    def test_lists_roles_and_picks(self) -> None:
        static = StaticData(
            version="1",
            champions={412: "Thresh", 222: "Jinx", 103: "Ahri"},
            summoner_spells={},
        )
        session = {
            "localPlayerCellId": 1,
            "myTeam": [
                {
                    "cellId": 1,
                    "assignedPosition": "utility",
                    "championId": 412,
                    "championPickIntent": 0,
                },
                {
                    "cellId": 2,
                    "assignedPosition": "bottom",
                    "championId": 0,
                    "championPickIntent": 222,
                },
            ],
            "theirTeam": [
                {
                    "cellId": 6,
                    "assignedPosition": "middle",
                    "championId": 103,
                    "championPickIntent": 0,
                }
            ],
            "bans": {"myTeamBans": [], "theirTeamBans": []},
            "actions": [],
        }
        text = format_lilith_brief(phase="ChampSelect", session=session, static_data=static)
        self.assertIn("Your lane: Support", text)
        self.assertIn("pick Thresh", text)
        self.assertIn("cell 1 YOU: Support pick=Thresh", text)
        self.assertIn("cell 2: Bot pick=- hover=Jinx", text)
        self.assertIn("cell 6: Mid pick=Ahri", text)


if __name__ == "__main__":
    unittest.main()
