from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import torch

from lol_champ_select_recommender.train_draft_model import DraftDataset
from lol_champ_select_recommender.train_draft_model import build_champion_loss_weights
from lol_champ_select_recommender.train_draft_model import build_lr_scheduler
from lol_champ_select_recommender.modeling.draft_model import build_model_class
from lol_champ_select_recommender.modeling.draft_data import (
    NONE_TOKEN,
    PICK_TOKEN,
    build_model_vocab,
    build_training_example,
    champion_features_by_id,
    global_feature_id,
    numeric_bin_token,
    quantile_edges,
)


class DraftModelDataTest(unittest.TestCase):
    def test_quantile_edges_and_bins(self) -> None:
        edges = quantile_edges([1, 2, 3, 4, 5], bins=3)

        self.assertEqual(len(edges), 2)
        self.assertEqual(numeric_bin_token(1, edges), "bin_0")
        self.assertEqual(numeric_bin_token(5, edges), "bin_2")

    def test_build_training_example_uses_pick_token_without_target_static_feature_leak(self) -> None:
        draft_rows = [sample_draft_row()]
        feature_rows = sample_feature_rows()
        vocab = build_model_vocab(draft_rows, feature_rows, numeric_bins=4)
        features = champion_features_by_id(feature_rows)

        example = build_training_example(
            draft_rows[0],
            vocab,
            features,
            rng=random.Random(1),
            mask_probability=0.0,
            unk_probability=0.0,
            target_role="middle",
        )

        champion_pick_id = global_feature_id("champion", PICK_TOKEN, vocab)
        query_features = example.feature_ids[example.query_index]
        self.assertEqual(query_features[0], champion_pick_id)
        self.assertEqual(example.target, vocab["champion_id_to_token_id"]["103"])

        primary_tag_vocab = vocab["feature_vocabs"]["primary_tag"]
        expected_static_id = vocab["feature_offsets"]["primary_tag"] + primary_tag_vocab.get(
            NONE_TOKEN,
            primary_tag_vocab["<UNK>"],
        )
        primary_tag_feature_index = vocab["token_features"].index("primary_tag")
        self.assertEqual(query_features[primary_tag_feature_index], expected_static_id)

    def test_draft_dataset_can_upsample_rows_with_distinct_masks(self) -> None:
        draft_rows = [sample_draft_row()]
        feature_rows = sample_feature_rows()
        vocab = build_model_vocab(draft_rows, feature_rows, numeric_bins=4)
        features = champion_features_by_id(feature_rows)
        dataset = DraftDataset(
            draft_rows,
            vocab,
            features,
            mask_probability=1.0,
            unk_probability=0.0,
            seed=123,
            examples_per_row=3,
        )

        self.assertEqual(len(dataset), 3)
        dataset.set_epoch(1)
        with patch("lol_champ_select_recommender.train_draft_model.build_training_example") as mock_build:
            mock_build.side_effect = lambda *args, **kwargs: kwargs["rng"].random()
            sample_a = dataset[0]
            sample_b = dataset[1]

        self.assertNotEqual(sample_a, sample_b)
        self.assertEqual(mock_build.call_count, 2)

    def test_champion_loss_weights_upweight_rare_champions(self) -> None:
        draft_rows = [
            sample_draft_row(),
            {
                **sample_draft_row(),
                "blue": {
                    "top": 82,
                    "jungle": 82,
                    "middle": 82,
                    "bottom": 82,
                    "utility": 103,
                },
            },
        ]
        feature_rows = sample_feature_rows()
        vocab = build_model_vocab(draft_rows, feature_rows, numeric_bins=4)

        weights = build_champion_loss_weights(draft_rows, vocab, power=0.5, torch_module=torch)

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertGreater(weights[vocab["champion_id_to_token_id"]["103"]].item(), weights[vocab["champion_id_to_token_id"]["82"]].item())
        self.assertEqual(weights[vocab["champion_token_to_id"]["<PAD>"]].item(), 0.0)

    def test_lr_scheduler_can_be_disabled(self) -> None:
        import torch.optim

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = type("Args", (), {"lr_scheduler": "none", "lr_scheduler_factor": 0.5, "lr_scheduler_patience": 2, "lr_scheduler_min_lr": 1e-6})()

        scheduler = build_lr_scheduler(torch, optimizer, args)

        self.assertIsNone(scheduler)

    def test_model_can_use_role_specific_heads(self) -> None:
        SharedFeatureDraftTransformer = build_model_class()
        model = SharedFeatureDraftTransformer(
            shared_vocab_size=16,
            champion_vocab_size=6,
            d_model=8,
            num_heads=1,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            use_role_heads=True,
        )

        feature_ids = torch.randint(0, 16, (2, 5, 4))
        query_index = torch.tensor([0, 4])
        logits = model(feature_ids, query_index)

        self.assertEqual(tuple(logits.shape), (2, 6))


def sample_draft_row():
    return {
        "match_id": "BR1_1",
        "patch": "16.13",
        "queue_id": 420,
        "winning_side": "blue",
        "rank_bucket": "MASTER",
        "blue": {
            "top": 82,
            "jungle": 64,
            "middle": 103,
            "bottom": 145,
            "utility": 267,
        },
        "red": {
            "top": 266,
            "jungle": 141,
            "middle": 99,
            "bottom": 22,
            "utility": 412,
        },
        "blue_bans": [157, 350, 238, -1, 555],
        "red_bans": [887, 7, 517, 55, -1],
    }


def sample_feature_rows():
    return [
        feature_row(82, "Mordekaiser", "Fighter", "Mage", 175, 4, 6, 7, 4),
        feature_row(64, "LeeSin", "Fighter", "Assassin", 125, 8, 5, 3, 6),
        feature_row(103, "Ahri", "Mage", "Assassin", 550, 3, 4, 8, 5),
        feature_row(145, "Kaisa", "Marksman", "Mage", 525, 8, 5, 3, 6),
        feature_row(267, "Nami", "Support", "Mage", 550, 4, 3, 7, 5),
        feature_row(266, "Aatrox", "Fighter", "", 175, 8, 4, 3, 4),
        feature_row(141, "Kayn", "Fighter", "Assassin", 175, 10, 6, 1, 8),
        feature_row(99, "Lux", "Mage", "Support", 550, 2, 4, 9, 5),
        feature_row(22, "Ashe", "Marksman", "Support", 600, 7, 3, 2, 4),
        feature_row(412, "Thresh", "Support", "Tank", 450, 5, 6, 6, 7),
        feature_row(157, "Yasuo", "Fighter", "Assassin", 175, 8, 4, 4, 10),
        feature_row(350, "Yuumi", "Support", "Mage", 425, 5, 1, 8, 2),
        feature_row(238, "Zed", "Assassin", "", 125, 9, 2, 1, 7),
        feature_row(555, "Pyke", "Support", "Assassin", 150, 9, 3, 1, 7),
        feature_row(887, "Gwen", "Fighter", "Assassin", 150, 7, 4, 5, 5),
        feature_row(7, "Leblanc", "Assassin", "Mage", 525, 1, 4, 10, 9),
        feature_row(517, "Sylas", "Mage", "Assassin", 175, 3, 4, 8, 5),
        feature_row(55, "Katarina", "Assassin", "Mage", 125, 4, 3, 9, 8),
    ]


def feature_row(champion_id, key, primary, secondary, attack_range, attack, defense, magic, difficulty):
    return {
        "champion_id": str(champion_id),
        "champion_key": key,
        "champion_name": key,
        "primary_tag": primary,
        "secondary_tag": secondary or NONE_TOKEN,
        "partype": "Mana",
        "range_type": "ranged" if attack_range >= 300 else "melee",
        "info_attack": str(attack),
        "info_defense": str(defense),
        "info_magic": str(magic),
        "info_difficulty": str(difficulty),
        "stat_attackrange": str(attack_range),
        "stat_hp": "600",
    }


if __name__ == "__main__":
    unittest.main()
