# Experiment Log

## 2026-07-05 - Hierarchy with coarse conditioning

Config:

```text
--d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4
```

Observed validation peaks:

```text
val_top10 ~ 0.3853
val_top5  ~ 0.2102
val_acc   ~ 0.0630
```

Diagnosis:

- champ-oracle remained high
- champ-pred collapsed hard
- coarse prediction was the bottleneck

## 2026-07-05 - Mixed coarse teacher forcing

Config:

```text
same as above, with mixed coarse conditioning in training
```

Observed validation peaks:

```text
val_top10 ~ 0.3853
val_top5  ~ 0.2242
val_acc   ~ 0.0595
```

Diagnosis:

- reduced exposure bias slightly
- did not fix the coarse handoff

## 2026-07-05 - Auxiliary-only hierarchy

Config:

```text
same as above, with hierarchy kept as auxiliary only and no coarse conditioning in the champion head
```

Observed validation peaks:

```text
val_top10 ~ 0.4326
val_top5  ~ 0.2434
val_acc   ~ 0.0771
```

Checkpoint ablation:

```text
coarse_only   acc=0.2399 top5=0.6357 top10=0.7758
champ_pred    acc=0.0665 top5=0.2557 top10=0.3975
champ_oracle  acc=0.0665 top5=0.2557 top10=0.3975
```

Diagnosis:

- removed the oracle/pred gap
- hierarchy no longer affects champion inference
- performance improved a bit, but not enough to justify a hard dependency

## 2026-07-05 - Flat role-head baseline

Config:

```text
--no-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --champion-loss-weight-power 0.35 --train-examples-per-row 4
```

Observed validation peaks:

```text
val_top10 ~ 0.4273
val_top5  ~ 0.2452
val_acc   ~ 0.0806
```

Checkpoint ablation:

```text
champ_pred acc=0.0648 top5=0.2452 top10=0.4273
```

Diagnosis:

- flat role heads are at least as good as the auxiliary hierarchy
- removing the coarse dependency did not hurt inference
- this is the current clean baseline to beat

## 2026-07-05 - 6-epoch hierarchy baseline

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 16 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4151
val_top5  ~ 0.2732
val_mrr   ~ 0.1870
```

## 2026-07-05 - 6-epoch small model

Config:

```text
--use-hierarchy --d-model 64 --num-heads 1 --num-layers 2 --dim-feedforward 256 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 16 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4361
val_top5  ~ 0.2592
val_mrr   ~ 0.1800
```

## 2026-07-05 - 6-epoch large model

Config:

```text
--use-hierarchy --d-model 256 --num-heads 1 --num-layers 4 --dim-feedforward 1024 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 16 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4291
val_top5  ~ 0.2382
val_mrr   ~ 0.1667
```

## 2026-07-05 - 6-epoch low LR

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 16 --lr 1e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4361
val_top5  ~ 0.2697
val_mrr   ~ 0.1744
```

## 2026-07-05 - 6-epoch high LR

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 16 --lr 5e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4203
val_top5  ~ 0.2434
val_mrr   ~ 0.1716
```

## 2026-07-05 - 6-epoch batch 8

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 8 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4133
val_top5  ~ 0.2592
val_mrr   ~ 0.1765
```

## 2026-07-05 - 6-epoch batch 32

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --batch-size 32 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4308
val_top5  ~ 0.2347
val_mrr   ~ 0.1848
```

## 2026-07-05 - 6-epoch no smoothing / no champion weighting

Config:

```text
--use-hierarchy --d-model 128 --num-heads 1 --num-layers 4 --dim-feedforward 512 --numeric-bins 10
--label-smoothing 0.0 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.0
--train-examples-per-row 4 --batch-size 16 --lr 3e-4
```

Observed validation peaks:

```text
val_top10 ~ 0.4343
val_top5  ~ 0.2504
val_mrr   ~ 0.1795
```

## Sweep conclusion

- `d_model=64`, `num_layers=2`, `dim_feedforward=256` was the best small-model point in this short sweep.
- `d_model=256`, `dim_feedforward=1024` was worse than the baseline.
- `lr=5e-4` was worse; `lr=1e-4` and `lr=3e-4` were the competitive points.
- `batch_size=8` was worse; `batch_size=32` was competitive but not a clear win.
- Label smoothing and champion-loss reweighting were not decisive. The no-smoothing/no-weight run was competitive enough that this is not a strong lever yet.
- Net: the model is still data-limited more than architecture-limited. The current best short-run point is the small model or low-LR variant, but the gains are small enough that I would not treat them as a hard baseline change yet.

## 2026-07-05 - Bigger corpus model size sweep

Dataset:

```text
data/processed/draft_dataset.jsonl rows: 16033
```

Shared config:

```text
--use-hierarchy --epochs 6 --batch-size 16 --lr 3e-4 --weight-decay 0.01
--label-smoothing 0.03 --coarse-loss-weight 0.3 --champion-loss-weight-power 0.35
--train-examples-per-row 4 --numeric-bins 10 --mask-probability 0.25 --unk-probability 0.03
--train-fraction 0.35 --eval-fraction 1.0 --val-split 0.15 --seed 1
```

Raw generated logs:

```text
data/experiments/bigger_corpus_size_sweep_2026-07-05_123555.log
data/experiments/bigger_corpus_size_sweep_2026-07-05_123555.summary.tsv
```

Peak validation metrics, selected by best `val_top10` epoch per run:

| Config | Best epoch | Top-10 | Top-5 | Acc | MRR | Macro F1 | Best loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d64_l2_ff256` | 6 | 0.4146 | 0.2432 | 0.0657 | 0.1722 | 0.0056 | 4.8831 |
| `d128_l4_ff512` | 4 | 0.4158 | 0.2437 | 0.0628 | 0.1715 | 0.0098 | 4.8903 |
| `d192_l4_ff768` | 6 | 0.4233 | 0.2620 | 0.0599 | 0.1690 | 0.0079 | 4.8754 |
| `d256_l4_ff1024` | 6 | 0.4195 | 0.2457 | 0.0553 | 0.1672 | 0.0135 | 4.8890 |
| `d384_l4_ff1536` | 6 | 0.4262 | 0.2482 | 0.0628 | 0.1690 | 0.0100 | 4.8932 |

Conclusion:

- The only configs that materially matter right now are model capacity, learning rate, batch size, label smoothing, champion frequency weighting, and the hierarchy/coarse-loss controls. The other flags should stay fixed for comparability until these are settled.
- On the bigger corpus, extra width helps top-10 a little, but not cleanly: `d384` wins top-10, while `d192` wins top-5 and validation loss.
- `d256` is dominated by `d192` and `d384`, so it is not a good main pretrain target.
- For the main pretrain, prefer `d192_l4_ff768` as the balanced config if runtime/VRAM matter. Use `d384_l4_ff1536` only if optimizing specifically for top-10 candidate recall.
- Keep `lr=3e-4`, `batch_size=16`, `label_smoothing=0.03`, `champion_loss_weight_power=0.35`, and `coarse_loss_weight=0.3` for the next main pretrain unless a dedicated sweep contradicts them on the full corpus.
