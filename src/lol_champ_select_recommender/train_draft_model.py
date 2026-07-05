from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from .modeling.draft_data import (
    SPECIAL_CHAMPION_TOKENS,
    build_model_vocab,
    build_training_example,
    champion_features_by_id,
    load_champion_feature_rows,
    load_jsonl,
    numeric_bin_token,
    to_float,
    to_int,
)
from .modeling.draft_model import MissingTorchError, build_model_class, require_torch
from .roles import POSITION_ORDER

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else range(0)


class DraftDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_vocab: dict[str, Any],
        champion_features: dict[int, dict[str, Any]],
        *,
        mask_probability: float,
        unk_probability: float,
        seed: int,
        examples_per_row: int = 1,
    ) -> None:
        self.rows = rows
        self.model_vocab = model_vocab
        self.champion_features = champion_features
        self.mask_probability = mask_probability
        self.unk_probability = unk_probability
        self.seed = seed
        self.examples_per_row = max(1, examples_per_row)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows) * self.examples_per_row

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        row_index = index // self.examples_per_row
        repeat_index = index % self.examples_per_row
        rng = random.Random(self.seed + self.epoch * 1_000_003 + row_index * 10_000 + repeat_index)
        return build_training_example(
            self.rows[row_index],
            self.model_vocab,
            self.champion_features,
            rng=rng,
            mask_probability=self.mask_probability,
            unk_probability=self.unk_probability,
        )


def main() -> int:
    args = parse_args()

    try:
        torch, _nn = require_torch()
        SharedFeatureDraftTransformer = build_model_class()
    except MissingTorchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    draft_rows = load_jsonl(args.dataset)
    feature_rows = load_champion_feature_rows(args.champion_features)
    if not draft_rows:
        print(f"Error: no draft rows found in {args.dataset}", file=sys.stderr)
        return 1
    if not feature_rows:
        print(f"Error: no champion features found in {args.champion_features}", file=sys.stderr)
        return 1

    champion_features = champion_features_by_id(feature_rows)
    if args.finetune_patch:
        draft_rows, finetune_summary = build_finetune_rows(
            draft_rows,
            finetune_patch=args.finetune_patch,
            historical_ratio=args.finetune_historical_ratio,
            seed=args.seed,
        )
        print(
            "Finetune mix: "
            f"latest_patch={args.finetune_patch} "
            f"latest_rows={finetune_summary['latest_rows']} "
            f"historical_sampled={finetune_summary['historical_sampled']} "
            f"historical_pool={finetune_summary['historical_rows']}"
        )
    else:
        rng = random.Random(args.seed)
        rng.shuffle(draft_rows)

    split_index = max(1, int(len(draft_rows) * (1 - args.val_split)))
    train_rows = draft_rows[:split_index]
    val_rows = draft_rows[split_index:] or draft_rows[: min(len(draft_rows), args.batch_size)]

    model_vocab = build_model_vocab(train_rows, feature_rows, numeric_bins=args.numeric_bins)
    checkpoint = None
    checkpoint_model_config = None
    if args.init_checkpoint:
        checkpoint = torch.load(Path(args.init_checkpoint), map_location="cpu")
        checkpoint_model_config = checkpoint["model_config"]
        draft_rows = filter_rows_for_finetuning(draft_rows, model_vocab)
        split_index = max(1, int(len(draft_rows) * (1 - args.val_split)))
        train_rows = draft_rows[:split_index]
        val_rows = draft_rows[split_index:] or draft_rows[: min(len(draft_rows), args.batch_size)]
        model_vocab = build_model_vocab(train_rows, feature_rows, numeric_bins=args.numeric_bins)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_vocab.json").write_text(json.dumps(model_vocab, indent=2, sort_keys=True), encoding="utf-8")

    train_dataset = DraftDataset(
        train_rows,
        model_vocab,
        champion_features,
        mask_probability=args.mask_probability,
        unk_probability=args.unk_probability,
        seed=args.seed,
        examples_per_row=args.train_examples_per_row,
    )
    val_dataset = DraftDataset(
        val_rows,
        model_vocab,
        champion_features,
        mask_probability=args.mask_probability,
        unk_probability=args.unk_probability,
        seed=args.seed + 17,
        examples_per_row=1,
    )
    champion_loss_weights = build_champion_loss_weights(
        train_rows,
        model_vocab,
        power=args.champion_loss_weight_power,
        torch_module=torch,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using {device=}")
    if checkpoint_model_config is None:
        model_config = {
            "shared_vocab_size": model_vocab["shared_vocab_size"],
            "champion_vocab_size": model_vocab["champion_vocab_size"],
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "use_role_heads": True,
            "use_hierarchy": args.use_hierarchy,
            "coarse_bucket_size": model_vocab["coarse_bucket_size"],
        }
    else:
        model_config = {
            "shared_vocab_size": model_vocab["shared_vocab_size"],
            "champion_vocab_size": model_vocab["champion_vocab_size"],
            "coarse_bucket_size": model_vocab["coarse_bucket_size"],
            "d_model": checkpoint_model_config["d_model"],
            "num_heads": checkpoint_model_config["num_heads"],
            "num_layers": checkpoint_model_config["num_layers"],
            "dim_feedforward": checkpoint_model_config["dim_feedforward"],
            "dropout": checkpoint_model_config["dropout"],
            "use_role_heads": bool(checkpoint_model_config.get("use_role_heads", False)),
            "use_hierarchy": bool(checkpoint_model_config.get("use_hierarchy", False)),
        }
    model = SharedFeatureDraftTransformer(
        shared_vocab_size=model_config["shared_vocab_size"],
        champion_vocab_size=model_config["champion_vocab_size"],
        coarse_bucket_size=model_config.get("coarse_bucket_size", 0),
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
        use_role_heads=bool(model_config.get("use_role_heads", False)),
        use_hierarchy=bool(model_config.get("use_hierarchy", False)),
    ).to(device)
    if checkpoint is not None:
        load_finetune_checkpoint(
            model,
            checkpoint["model_state_dict"],
            checkpoint["model_vocab"],
            model_vocab,
            champion_features,
        )
        print(f"Loaded checkpoint from {args.init_checkpoint} with vocab expansion")
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(
        weight=champion_loss_weights.to(device) if champion_loss_weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    coarse_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = build_lr_scheduler(torch, optimizer, args)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        train_loss, train_acc, train_top5, train_top10, train_coarse_acc, train_mrr, train_macro_f1 = run_epoch(
            torch,
            model,
            train_dataset,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            coarse_loss_weight=args.coarse_loss_weight,
            use_teacher_forcing_coarse=True,
            use_hierarchy=args.use_hierarchy,
            optimizer=optimizer,
            train=True,
            batch_fraction=args.train_fraction,
            progress_label="train",
            eval_every_batches=args.eval_every_train_batches,
            eval_callback=build_mid_epoch_eval_callback(
                torch,
                model,
                val_dataset,
                batch_size=args.batch_size,
                device=device,
                criterion=criterion,
                coarse_criterion=coarse_criterion,
                coarse_loss_weight=args.coarse_loss_weight,
                use_hierarchy=args.use_hierarchy,
                eval_fraction=args.mid_epoch_eval_fraction,
                epoch=epoch,
            )
            if args.eval_every_train_batches > 0
            else None,
        )
        val_loss, val_acc, val_top5, val_top10, val_coarse_acc, val_mrr, val_macro_f1 = run_epoch(
            torch,
            model,
            val_dataset,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            coarse_loss_weight=args.coarse_loss_weight,
            use_teacher_forcing_coarse=False,
            use_hierarchy=args.use_hierarchy,
            optimizer=None,
            train=False,
            batch_fraction=args.eval_fraction,
            progress_label="eval",
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_top5={train_top5:.4f} train_top10={train_top10:.4f} "
            f"train_coarse_acc={train_coarse_acc:.4f} train_mrr={train_mrr:.4f} train_macro_f1={train_macro_f1:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_top5={val_top5:.4f} val_top10={val_top10:.4f} "
            f"val_coarse_acc={val_coarse_acc:.4f} val_mrr={val_mrr:.4f} val_macro_f1={val_macro_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "shared_vocab_size": model_vocab["shared_vocab_size"],
                "champion_vocab_size": model_vocab["champion_vocab_size"],
                "d_model": model_config["d_model"],
                "num_heads": model_config["num_heads"],
                "num_layers": model_config["num_layers"],
                "dim_feedforward": model_config["dim_feedforward"],
                "dropout": model_config["dropout"],
                "use_role_heads": bool(model_config.get("use_role_heads", False)),
                "use_hierarchy": bool(model_config.get("use_hierarchy", False)),
                "coarse_bucket_size": model_vocab["coarse_bucket_size"],
            },
            "model_vocab": model_vocab,
            "epoch": epoch,
            "val_loss": val_loss,
            "special_champion_tokens": SPECIAL_CHAMPION_TOKENS,
            "train_config": {
                "champion_loss_weight_power": args.champion_loss_weight_power,
                "label_smoothing": args.label_smoothing,
                "coarse_loss_weight": args.coarse_loss_weight,
                "init_checkpoint": args.init_checkpoint,
                "finetune_patch": args.finetune_patch,
                "finetune_historical_ratio": args.finetune_historical_ratio,
                "train_fraction": args.train_fraction,
                "eval_fraction": args.eval_fraction,
                "eval_every_train_batches": args.eval_every_train_batches,
                "mid_epoch_eval_fraction": args.mid_epoch_eval_fraction,
                "lr_scheduler": args.lr_scheduler,
                "lr_scheduler_factor": args.lr_scheduler_factor,
                "lr_scheduler_patience": args.lr_scheduler_patience,
                "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
            },
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"Saved checkpoints to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked winning-draft champion model.")
    parser.add_argument("--dataset", default="data/processed/draft_dataset.jsonl")
    parser.add_argument("--champion-features", default="data/processed/champion_features.csv")
    parser.add_argument("--output-dir", default="data/models/draft_transformer")
    parser.add_argument(
        "--init-checkpoint",
        help="Checkpoint to initialize weights and vocab from before finetuning.",
    )
    parser.add_argument(
        "--finetune-patch",
        help="If set, finetune on this patch and mix in historical replay rows.",
    )
    parser.add_argument(
        "--finetune-historical-ratio",
        type=float,
        default=0.2,
        help="Historical rows to mix in relative to the latest patch rows. Default: 0.2",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--use-hierarchy",
        dest="use_hierarchy",
        action="store_true",
        help="Enable the auxiliary coarse hierarchy head. Default: on",
    )
    parser.add_argument(
        "--no-hierarchy",
        dest="use_hierarchy",
        action="store_false",
        help="Disable the auxiliary coarse hierarchy head.",
    )
    parser.set_defaults(use_hierarchy=True)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing. Default: 0.05",
    )
    parser.add_argument(
        "--coarse-loss-weight",
        type=float,
        default=0.5,
        help="Auxiliary loss weight for the coarse hierarchy head. Default: 0.5",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="plateau",
        help="Learning-rate scheduler to use. Default: plateau",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Factor applied by ReduceLROnPlateau. Default: 0.5",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=2,
        help="Epochs without val-loss improvement before reducing LR. Default: 2",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for the scheduler. Default: 1e-6",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.25)
    parser.add_argument("--unk-probability", type=float, default=0.03)
    parser.add_argument("--numeric-bins", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--champion-loss-weight-power",
        type=float,
        default=0.5,
        help="Loss reweighting power for rare champions. 0 disables weighting. Default: 0.5",
    )
    parser.add_argument(
        "--train-examples-per-row",
        type=int,
        default=1,
        help="How many masked training examples to draw from each match row per epoch. Default: 1",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of train batches to process each epoch. Useful for faster sweeps. Default: 1.0",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation batches to evaluate each epoch. Useful for faster sweeps. Default: 1.0",
    )
    parser.add_argument(
        "--eval-every-train-batches",
        type=int,
        default=0,
        help="Run an extra validation pass every N train batches within each epoch. 0 disables it. Default: 0",
    )
    parser.add_argument(
        "--mid-epoch-eval-fraction",
        type=float,
        default=0.25,
        help="Fraction of validation batches for mid-epoch evals. Default: 0.25",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", help="Torch device override, e.g. cpu, cuda")
    return parser.parse_args()


def filter_rows_for_finetuning(
    rows: list[dict[str, Any]],
    model_vocab: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if model_vocab is None:
        return list(rows)

    known_champion_ids = {int(champion_id) for champion_id in model_vocab["champion_id_to_token_id"].keys()}
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        winning_side = str(row.get("winning_side") or "")
        side = row.get(winning_side)
        if not isinstance(side, dict):
            continue
        if all(to_int(side.get(role)) in known_champion_ids for role in POSITION_ORDER):
            filtered_rows.append(row)
    return filtered_rows


def load_finetune_checkpoint(
    model,
    checkpoint_state_dict: dict[str, Any],
    checkpoint_vocab: dict[str, Any],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
) -> None:
    current_state = model.state_dict()
    updated_state = dict(current_state)

    for key, value in checkpoint_state_dict.items():
        if key not in current_state:
            continue
        target = current_state[key]
        if key == "embedding.weight":
            updated_state[key] = remap_embedding_weights(value, checkpoint_vocab, model_vocab, target)
            continue

        if key in {"output.weight", "output.bias"} or key.startswith("role_outputs.") or key.startswith("role_coarse_outputs."):
            updated_state[key] = remap_head_weights(key, value, checkpoint_vocab, model_vocab, target)
            continue

        if target.shape == value.shape:
            updated_state[key] = value

    initialize_new_champion_rows(
        updated_state,
        checkpoint_vocab=checkpoint_vocab,
        model_vocab=model_vocab,
        champion_features=champion_features,
    )
    model.load_state_dict(updated_state, strict=True)


def initialize_new_champion_rows(
    state_dict: dict[str, Any],
    *,
    checkpoint_vocab: dict[str, Any],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
) -> None:
    old_champion_ids = {int(champion_id) for champion_id in checkpoint_vocab["champion_id_to_token_id"].keys()}
    current_champion_ids = {int(champion_id) for champion_id in model_vocab["champion_id_to_token_id"].keys()}
    new_champion_ids = sorted(current_champion_ids - old_champion_ids)
    if not new_champion_ids:
        return

    champion_offset = int(model_vocab["feature_offsets"]["champion"])
    current_champion_tokens = model_vocab["feature_vocabs"]["champion"]
    current_output_rows = model_vocab["champion_id_to_token_id"]

    for champion_id in new_champion_ids:
        source_ids = similar_champion_ids(champion_id, checkpoint_vocab, model_vocab, champion_features)
        if not source_ids:
            source_ids = sorted(old_champion_ids)
        if not source_ids:
            continue

        source_embedding_indices = [
            champion_offset + int(current_champion_tokens[str(source_id)])
            for source_id in source_ids
            if str(source_id) in current_champion_tokens
        ]
        if source_embedding_indices:
            state_dict["embedding.weight"][champion_offset + int(current_champion_tokens[str(champion_id)])] = weighted_mean_rows(
                state_dict["embedding.weight"],
                source_embedding_indices,
            )

        target_output_index = int(current_output_rows[str(champion_id)])
        source_output_indices = [
            int(current_output_rows[str(source_id)])
            for source_id in source_ids
            if str(source_id) in current_output_rows
        ]
        if source_output_indices:
            state_dict["output.weight"][target_output_index] = weighted_mean_rows(
                state_dict["output.weight"],
                source_output_indices,
            )
            state_dict["output.bias"][target_output_index] = weighted_mean_rows(
                state_dict["output.bias"].unsqueeze(1),
                source_output_indices,
            ).squeeze(0)
            for role in POSITION_ORDER:
                role_weight_key = f"role_outputs.{role}.weight"
                role_bias_key = f"role_outputs.{role}.bias"
                state_dict[role_weight_key][target_output_index] = weighted_mean_rows(
                    state_dict[role_weight_key],
                    source_output_indices,
                )
                state_dict[role_bias_key][target_output_index] = weighted_mean_rows(
                    state_dict[role_bias_key].unsqueeze(1),
                    source_output_indices,
                ).squeeze(0)
                coarse_weight_key = f"role_coarse_outputs.{role}.weight"
                coarse_bias_key = f"role_coarse_outputs.{role}.bias"
                if coarse_weight_key in state_dict and coarse_bias_key in state_dict:
                    state_dict[coarse_weight_key][target_output_index] = weighted_mean_rows(
                        state_dict[coarse_weight_key],
                        source_output_indices,
                    )
                    state_dict[coarse_bias_key][target_output_index] = weighted_mean_rows(
                        state_dict[coarse_bias_key].unsqueeze(1),
                        source_output_indices,
                    ).squeeze(0)


def similar_champion_ids(
    champion_id: int,
    checkpoint_vocab: dict[str, Any],
    model_vocab: dict[str, Any],
    champion_features: dict[int, dict[str, Any]],
) -> list[int]:
    target_row = champion_features.get(champion_id)
    if not target_row:
        return []

    current_ids = [int(champion_id_str) for champion_id_str in checkpoint_vocab["champion_id_to_token_id"].keys()]
    scored: list[tuple[float, int]] = []
    for candidate_id in current_ids:
        candidate_row = champion_features.get(candidate_id)
        if not candidate_row:
            continue
        score = champion_similarity_score(target_row, candidate_row, model_vocab)
        scored.append((score, candidate_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    positive = [champion_id for score, champion_id in scored if score > 0]
    if positive:
        return positive[: min(5, len(positive))]
    return [champion_id for _score, champion_id in scored[: min(5, len(scored))]]


def champion_similarity_score(
    target_row: dict[str, Any],
    candidate_row: dict[str, Any],
    model_vocab: dict[str, Any],
) -> float:
    score = 0.0
    categorical_weights = {
        "primary_tag": 4.0,
        "secondary_tag": 3.0,
        "partype": 2.0,
        "range_type": 4.0,
    }
    for feature_name, weight in categorical_weights.items():
        if normalized_feature_value(target_row.get(feature_name)) == normalized_feature_value(candidate_row.get(feature_name)):
            score += weight

    for feature_name in model_vocab["numeric_feature_names"]:
        edges = model_vocab["numeric_bin_edges"].get(feature_name, [])
        target_bin = numeric_bin_token(to_float(target_row.get(feature_name)), edges)
        candidate_bin = numeric_bin_token(to_float(candidate_row.get(feature_name)), edges)
        if target_bin == candidate_bin:
            score += 1.0

    return score


def normalized_feature_value(value: Any) -> str:
    token = str(value or "").strip()
    return token if token else "<NONE>"


def weighted_mean_rows(matrix, row_indices: list[int]):
    if not row_indices:
        raise ValueError("weighted_mean_rows requires at least one row index")
    return matrix[row_indices].mean(dim=0)


def remap_embedding_weights(
    checkpoint_weights,
    checkpoint_vocab: dict[str, Any],
    model_vocab: dict[str, Any],
    target_tensor,
):
    remapped = target_tensor.clone()
    checkpoint_offsets = checkpoint_vocab["feature_offsets"]
    model_offsets = model_vocab["feature_offsets"]
    for feature_name in checkpoint_vocab["token_features"]:
        old_vocab = checkpoint_vocab["feature_vocabs"][feature_name]
        new_vocab = model_vocab["feature_vocabs"][feature_name]
        old_offset = int(checkpoint_offsets[feature_name])
        new_offset = int(model_offsets[feature_name])
        for token, old_local_id in old_vocab.items():
            if token not in new_vocab:
                continue
            new_local_id = int(new_vocab[token])
            remapped[new_offset + new_local_id] = checkpoint_weights[old_offset + int(old_local_id)]
    return remapped


def remap_head_weights(
    key: str,
    checkpoint_tensor,
    checkpoint_vocab: dict[str, Any],
    model_vocab: dict[str, Any],
    target_tensor,
):
    remapped = target_tensor.clone()
    if key.startswith("role_coarse_outputs.") and "coarse_bucket_to_id" in checkpoint_vocab:
        old_vocab = checkpoint_vocab["coarse_bucket_to_id"]
        new_vocab = model_vocab["coarse_bucket_to_id"]
    else:
        old_vocab = checkpoint_vocab["champion_token_to_id"]
        new_vocab = model_vocab["champion_token_to_id"]

    if checkpoint_tensor.dim() == 2:
        for token, old_index in old_vocab.items():
            if token not in new_vocab:
                continue
            new_index = int(new_vocab[token])
            remapped[new_index] = checkpoint_tensor[int(old_index)]
    elif checkpoint_tensor.dim() == 1:
        for token, old_index in old_vocab.items():
            if token not in new_vocab:
                continue
            new_index = int(new_vocab[token])
            remapped[new_index] = checkpoint_tensor[int(old_index)]
    return remapped


def build_finetune_rows(
    rows: list[dict[str, Any]],
    *,
    finetune_patch: str,
    historical_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    latest_rows = [row for row in rows if str(row.get("patch")) == str(finetune_patch)]
    historical_rows = [row for row in rows if str(row.get("patch")) != str(finetune_patch)]
    if not latest_rows:
        raise ValueError(f"No rows found for finetune patch {finetune_patch}")

    rng = random.Random(seed)
    rng.shuffle(historical_rows)
    historical_sample_count = min(
        len(historical_rows),
        max(0, int(round(len(latest_rows) * max(0.0, historical_ratio)))),
    )
    selected_rows = latest_rows + historical_rows[:historical_sample_count]
    rng.shuffle(selected_rows)
    return selected_rows, {
        "latest_rows": len(latest_rows),
        "historical_rows": len(historical_rows),
        "historical_sampled": historical_sample_count,
        "selected_rows": len(selected_rows),
    }


def run_epoch(
    torch,
    model,
    dataset: DraftDataset,
    *,
    batch_size: int,
    device,
    criterion,
    coarse_criterion,
    coarse_loss_weight: float,
    use_teacher_forcing_coarse: bool,
    use_hierarchy: bool,
    optimizer,
    train: bool,
    batch_fraction: float = 1.0,
    progress_label: str = "train",
    eval_every_batches: int = 0,
    eval_callback=None,
) -> tuple[float, float, float, float, float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_top10 = 0
    total_mrr = 0.0
    total_coarse_correct = 0
    total_examples = 0
    label_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}
    true_positive_counts: dict[int, int] = {}

    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)

    batch_starts = list(range(0, len(indices), batch_size))
    if batch_fraction < 1.0 and batch_starts:
        limit = max(1, int(math.ceil(len(batch_starts) * max(0.0, batch_fraction))))
        batch_starts = batch_starts[:limit]

    for batch_number, start in enumerate(tqdm(batch_starts, desc=progress_label, leave=False, dynamic_ncols=True), start=1):
        examples = [dataset[index] for index in indices[start : start + batch_size]]
        feature_ids = torch.tensor([example.feature_ids for example in examples], dtype=torch.long, device=device)
        query_index = torch.tensor([example.query_index for example in examples], dtype=torch.long, device=device)
        role_index = torch.tensor([POSITION_ORDER.index(example.target_role) for example in examples], dtype=torch.long, device=device)
        target = torch.tensor([example.target for example in examples], dtype=torch.long, device=device)
        target_coarse = torch.tensor([example.target_coarse for example in examples], dtype=torch.long, device=device)
        coarse_context_target = target_coarse if use_teacher_forcing_coarse else None

        with torch.set_grad_enabled(train):
            if use_hierarchy:
                outputs = model(
                    feature_ids,
                    query_index,
                    role_index=role_index,
                    target_coarse_index=coarse_context_target,
                )
            else:
                outputs = model(feature_ids, query_index, role_index=role_index)
            if isinstance(outputs, tuple):
                logits, coarse_logits = outputs
            else:
                logits, coarse_logits = outputs, None
            logits[:, : len(SPECIAL_CHAMPION_TOKENS)] = -1e9
            loss = criterion(logits, target)
            if coarse_logits is not None:
                loss = loss + coarse_criterion(coarse_logits, target_coarse) * coarse_loss_weight
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == target).sum().item())
        total_top5 += _topk_hits(logits, target, k=5)
        total_top10 += _topk_hits(logits, target, k=10)
        total_mrr += _mean_reciprocal_rank(logits, target)
        if coarse_logits is not None:
            coarse_predictions = coarse_logits.argmax(dim=1)
            total_coarse_correct += int((coarse_predictions == target_coarse).sum().item())
        for target_id, pred_id in zip(target.tolist(), predictions.tolist()):
            label_counts[target_id] = label_counts.get(target_id, 0) + 1
            pred_counts[pred_id] = pred_counts.get(pred_id, 0) + 1
            if target_id == pred_id:
                true_positive_counts[target_id] = true_positive_counts.get(target_id, 0) + 1
        total_loss += float(loss.item()) * len(examples)
        total_examples += len(examples)
        if train and eval_callback is not None and eval_every_batches > 0 and batch_number % eval_every_batches == 0:
            eval_callback(batch_number, len(batch_starts))
            model.train(True)

    macro_f1 = _macro_f1(label_counts, pred_counts, true_positive_counts)
    if total_examples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples, total_top5 / total_examples, total_top10 / total_examples, total_coarse_correct / total_examples, total_mrr / total_examples, macro_f1


def build_mid_epoch_eval_callback(
    torch,
    model,
    val_dataset: DraftDataset,
    *,
    batch_size: int,
    device,
    criterion,
    coarse_criterion,
    coarse_loss_weight: float,
    use_hierarchy: bool,
    eval_fraction: float,
    epoch: int,
):
    def callback(batch_number: int, total_batches: int) -> None:
        val_loss, val_acc, val_top5, val_top10, val_coarse_acc, val_mrr, val_macro_f1 = run_epoch(
            torch,
            model,
            val_dataset,
            batch_size=batch_size,
            device=device,
            criterion=criterion,
            coarse_criterion=coarse_criterion,
            coarse_loss_weight=coarse_loss_weight,
            use_teacher_forcing_coarse=False,
            use_hierarchy=use_hierarchy,
            optimizer=None,
            train=False,
            batch_fraction=eval_fraction,
            progress_label=f"mid-eval e{epoch:03d} b{batch_number}",
        )
        print(
            f"mid_epoch epoch={epoch:03d} batch={batch_number}/{total_batches} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_top5={val_top5:.4f} val_top10={val_top10:.4f} "
            f"val_coarse_acc={val_coarse_acc:.4f} val_mrr={val_mrr:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    return callback


def _topk_hits(logits, target, *, k: int) -> int:
    if logits.numel() == 0:
        return 0
    k = min(k, logits.size(1))
    _, topk = logits.topk(k, dim=1)
    return int((topk == target.unsqueeze(1)).any(dim=1).sum().item())


def _mean_reciprocal_rank(logits, target) -> float:
    if logits.numel() == 0:
        return 0.0
    rankings = logits.argsort(dim=1, descending=True)
    hits = rankings.eq(target.unsqueeze(1))
    ranks = hits.float().argmax(dim=1) + 1
    reciprocal_ranks = 1.0 / ranks.float()
    return float(reciprocal_ranks.sum().item())


def _macro_f1(
    label_counts: dict[int, int],
    pred_counts: dict[int, int],
    true_positive_counts: dict[int, int],
) -> float:
    labels = set(label_counts) | set(pred_counts)
    if not labels:
        return 0.0

    scores: list[float] = []
    for label in labels:
        tp = true_positive_counts.get(label, 0)
        fp = pred_counts.get(label, 0) - tp
        fn = label_counts.get(label, 0) - tp
        denom = (2 * tp) + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return sum(scores) / len(scores)


def build_champion_loss_weights(
    train_rows: list[dict[str, Any]],
    model_vocab: dict[str, Any],
    *,
    power: float,
    torch_module,
):
    if power <= 0:
        return None

    champion_counts: dict[int, int] = {}
    for row in train_rows:
        winning_side = str(row["winning_side"])
        ally = row[winning_side]
        for role in ("top", "jungle", "middle", "bottom", "utility"):
            champion_id = to_int(ally.get(role))
            if champion_id is None or champion_id <= 0:
                continue
            champion_counts[champion_id] = champion_counts.get(champion_id, 0) + 1

    weights = torch_module.ones(model_vocab["champion_vocab_size"], dtype=torch_module.float32)
    for special_token in SPECIAL_CHAMPION_TOKENS:
        weights[int(model_vocab["champion_token_to_id"][special_token])] = 0.0

    if not champion_counts:
        return weights

    max_count = max(champion_counts.values())
    champion_weights: list[float] = []
    for champion_id, count in champion_counts.items():
        token_id = int(model_vocab["champion_id_to_token_id"][str(champion_id)])
        value = (max_count / count) ** power
        weights[token_id] = float(value)
        champion_weights.append(float(value))

    mean_weight = sum(champion_weights) / len(champion_weights)
    if mean_weight > 0:
        for champion_id, count in champion_counts.items():
            token_id = int(model_vocab["champion_id_to_token_id"][str(champion_id)])
            weights[token_id] = float(weights[token_id] / mean_weight)

    return weights


def build_lr_scheduler(torch, optimizer, args):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")
