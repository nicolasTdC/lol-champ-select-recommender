from __future__ import annotations

import unittest

from lol_champ_select_recommender.build_champion_features import (
    NONE_TOKEN,
    add_categorical_ids,
    build_champion_feature_rows,
    build_feature_vocabs,
)
from lol_champ_select_recommender.ddragon import StaticData


class ChampionFeaturesTest(unittest.TestCase):
    def test_build_champion_feature_rows_from_datadragon_payload(self) -> None:
        static_data = StaticData(
            version="test",
            champions={103: "Ahri"},
            summoner_spells={},
            champion_keys={103: "Ahri"},
            champion_payloads={
                103: {
                    "id": "Ahri",
                    "name": "Ahri",
                    "title": "the Nine-Tailed Fox",
                    "partype": "Mana",
                    "tags": ["Mage", "Assassin"],
                    "info": {"attack": 3, "defense": 4, "magic": 8, "difficulty": 5},
                    "stats": {
                        "hp": 590,
                        "armor": 21,
                        "spellblock": 30,
                        "attackrange": 550,
                        "movespeed": 330,
                    },
                }
            },
        )

        rows = build_champion_feature_rows(static_data)
        vocabs = build_feature_vocabs(rows)
        rows = add_categorical_ids(rows, vocabs)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["champion_id"], 103)
        self.assertEqual(row["champion_key"], "Ahri")
        self.assertEqual(row["primary_tag"], "Mage")
        self.assertEqual(row["secondary_tag"], "Assassin")
        self.assertEqual(row["range_type"], "ranged")
        self.assertEqual(row["primary_tag_id"], vocabs["tag"]["Mage"])
        self.assertEqual(row["secondary_tag_id"], vocabs["tag"]["Assassin"])
        self.assertEqual(row["range_type_id"], vocabs["range_type"]["ranged"])
        self.assertNotIn("tag_mage", row)
        self.assertNotIn("is_ranged", row)
        self.assertNotIn("is_melee", row)
        self.assertEqual(row["info_magic"], 8.0)
        self.assertEqual(row["info_difficulty"], 5.0)
        self.assertEqual(row["stat_attackrange"], 550.0)

    def test_secondary_tag_uses_none_token_and_shared_tag_vocab(self) -> None:
        static_data = StaticData(
            version="test",
            champions={266: "Aatrox", 103: "Ahri"},
            summoner_spells={},
            champion_keys={266: "Aatrox", 103: "Ahri"},
            champion_payloads={
                266: {
                    "id": "Aatrox",
                    "name": "Aatrox",
                    "tags": ["Fighter"],
                    "info": {},
                    "stats": {"attackrange": 175},
                },
                103: {
                    "id": "Ahri",
                    "name": "Ahri",
                    "partype": "Mana",
                    "tags": ["Mage", "Assassin"],
                    "info": {},
                    "stats": {"attackrange": 550},
                },
            },
        )

        rows = build_champion_feature_rows(static_data)
        vocabs = build_feature_vocabs(rows)
        rows = add_categorical_ids(rows, vocabs)
        aatrox = next(row for row in rows if row["champion_key"] == "Aatrox")

        self.assertEqual(aatrox["secondary_tag"], NONE_TOKEN)
        self.assertEqual(aatrox["secondary_tag_id"], vocabs["tag"][NONE_TOKEN])
        self.assertEqual(aatrox["range_type"], "melee")
        self.assertIn("Mage", vocabs["tag"])
        self.assertIn("Assassin", vocabs["tag"])
        self.assertIn("Fighter", vocabs["tag"])


if __name__ == "__main__":
    unittest.main()
