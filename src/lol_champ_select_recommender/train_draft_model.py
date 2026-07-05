from __future__ import annotations

import argparse
import json
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
)
from .modeling.draft_model import MissingTorchError, build_model_class, require_torch


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
    ) -> None:
        self.rows = rows
        self.model_vocab = model_vocab
        self.champion_features = champion_features
        self.mask_probability = mask_probability
        self.unk_probability = unk_probability
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        return build_training_example(
            self.rows[index],
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

    rng = random.Random(args.seed)
    rng.shuffle(draft_rows)
    split_index = max(1, int(len(draft_rows) * (1 - args.val_split)))
    train_rows = draft_rows[:split_index]
    val_rows = draft_rows[split_index:] or draft_rows[: min(len(draft_rows), args.batch_size)]

    model_vocab = build_model_vocab(train_rows, feature_rows, numeric_bins=args.numeric_bins)
    champion_features = champion_features_by_id(feature_rows)
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
    )
    val_dataset = DraftDataset(
        val_rows,
        model_vocab,
        champion_features,
        mask_probability=args.mask_probability,
        unk_probability=args.unk_probability,
        seed=args.seed + 17,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using {device=}")
    model = SharedFeatureDraftTransformer(
        shared_vocab_size=model_vocab["shared_vocab_size"],
        champion_vocab_size=model_vocab["champion_vocab_size"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss, train_acc = run_epoch(
            torch,
            model,
            train_dataset,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            train=True,
        )
        val_dataset.set_epoch(epoch)
        val_loss, val_acc = run_epoch(
            torch,
            model,
            val_dataset,
            batch_size=args.batch_size,
            device=device,
            criterion=criterion,
            optimizer=None,
            train=False,
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "shared_vocab_size": model_vocab["shared_vocab_size"],
                "champion_vocab_size": model_vocab["champion_vocab_size"],
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
            "model_vocab": model_vocab,
            "epoch": epoch,
            "val_loss": val_loss,
            "special_champion_tokens": SPECIAL_CHAMPION_TOKENS,
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.25)
    parser.add_argument("--unk-probability", type=float, default=0.03)
    parser.add_argument("--numeric-bins", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", help="Torch device override, e.g. cpu, cuda")
    return parser.parse_args()


def run_epoch(
    torch,
    model,
    dataset: DraftDataset,
    *,
    batch_size: int,
    device,
    criterion,
    optimizer,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        examples = [dataset[index] for index in indices[start : start + batch_size]]
        feature_ids = torch.tensor([example.feature_ids for example in examples], dtype=torch.long, device=device)
        query_index = torch.tensor([example.query_index for example in examples], dtype=torch.long, device=device)
        target = torch.tensor([example.target for example in examples], dtype=torch.long, device=device)

        with torch.set_grad_enabled(train):
            logits = model(feature_ids, query_index)
            logits[:, : len(SPECIAL_CHAMPION_TOKENS)] = -1e9
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == target).sum().item())
        total_loss += float(loss.item()) * len(examples)
        total_examples += len(examples)

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples
