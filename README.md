# LoL Champ Select Recommender

Terminal MVP for reading live League of Legends champion select state from the local League Client API.

This first version focuses on detection, not recommendation quality:

- watches whether the League client is in champ select
- prints ally/enemy visible picks
- prints champion hovers / pick intents
- prints ally/enemy bans
- prints summoner spell IDs as names when Data Dragon is reachable
- prints phase, timer, and current in-progress actions

The app is read-only. It does not pick, ban, change summoner spells, or automate the League client.

## Run

From this directory:

```bash
python watch.py
```

Or install the local command:

```bash
python -m pip install -e .
lol-champ-select
```

If the lockfile is not found automatically:

```bash
python watch.py --lockfile "C:\Riot Games\League of Legends\lockfile"
```

From WSL, use the Windows-mounted path:

```bash
python watch.py --lockfile "/mnt/c/Riot Games/League of Legends/lockfile"
```

When running inside WSL, the watcher automatically tries `127.0.0.1` and then the Windows host IP from WSL's network config.
If those fail and `curl.exe` is available, it automatically falls back to Windows `curl.exe` so the request runs from the Windows network context.

If WSL can read the lockfile but still cannot connect to the local client, pass the Windows host IP explicitly:

```bash
export LCU_HOST="$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)"
python watch.py --lockfile "/mnt/c/Riot Games/League of Legends/lockfile"
```

Or pass it directly:

```bash
python watch.py --host 172.x.x.x --lockfile "/mnt/c/Riot Games/League of Legends/lockfile"
```

If every WSL host fails with connection refused, run the same project with Windows Python. Some Windows/WSL networking setups cannot reach services that bind only to Windows loopback.

Useful options:

```bash
python watch.py --once
python watch.py --interval 0.5
python watch.py --language pt_BR
python watch.py --no-clear
python watch.py --host 127.0.0.1
python watch.py --role-priors-min-games 10
python watch.py --role-priors-patch 16.13
```

## Fetch Match-V5 Data

Set your Riot API key for the current shell:

```bash
export RIOT_API_KEY="RGAPI-your-key-here"
```

Or create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder value. `.env` is ignored by git.

Fetch recent matches for a Riot ID:

```bash
python fetch_matches.py --riot-id "GameName#TAG" --region americas --count 20
```

Useful queue filters:

```bash
python fetch_matches.py --riot-id "GameName#TAG" --region americas --queue 420 --count 20
python fetch_matches.py --riot-id "GameName#TAG" --region americas --queue 440 --count 20
```

Queue `420` is ranked solo/duo. Queue `440` is ranked flex.

Downloaded matches are saved to:

```text
data/raw/matches/
```

Build a first champion-role aggregate from downloaded matches:

```bash
python aggregate_matches.py
```

The aggregate CSV is written to:

```text
data/processed/champion_role_stats.csv
```

The same command also writes role priors for live enemy-role inference:

```text
data/processed/champion_role_priors.csv
```

Build static champion features from Data Dragon:

```bash
python build_champion_features.py
```

The feature CSV and categorical vocab are written to:

```text
data/processed/champion_features.csv
data/processed/champion_feature_vocab.json
```

It includes categorical tokens and IDs for tags, resource type, and melee/ranged range type, plus Data Dragon `info` ratings and base stats. The categorical IDs are meant for model embedding layers.

Collect per-player champion-role stats for inference pruning:

```bash
python collect_player_stats.py --riot-id "GameName#TAG" --riot-id "Other#TAG"
```

The watcher can then prune model rankings with:

```bash
python watch.py --player-stats data/processed/player_champion_role_stats.csv
```

## Collect Server-Level Ranked Data

The single-account fetcher is biased toward one player's games. For broader server data, seed from League-V4 ranked ladder entries and download recent Match-V5 games from those players:

```bash
python collect_ranked_matches.py --platform br1 --tiers EMERALD --divisions I --pages 1 --max-players 5 --matches-per-player 3
```

Larger sample:

```bash
python collect_ranked_matches.py --platform br1 --tiers EMERALD PLATINUM --divisions I II III IV --pages 1 --max-players 50 --matches-per-player 5
```

Master, Grandmaster, and Challenger use different Riot ladder endpoints, so divisions/pages are ignored for those tiers:

```bash
python collect_ranked_matches.py --platform br1 --tiers MASTER --max-players 25 --matches-per-player 5
python collect_ranked_matches.py --platform br1 --tiers GRANDMASTER CHALLENGER --max-players 25 --matches-per-player 5
```

Then rebuild aggregates:

```bash
python aggregate_matches.py
```

Routing notes:

- `--platform br1` is the server/platform route used by League-V4 ladder endpoints.
- Match-V5 uses a regional route. The collector derives it automatically, e.g. `br1 -> americas`.
- The default queue is ranked solo/duo: League-V4 `RANKED_SOLO_5x5`, Match-V5 queue `420`.

## How It Works

The League client writes a `lockfile` while running. It contains the local API port and auth token:

```text
LeagueClient:<pid>:<port>:<password>:<protocol>
```

The CLI reads that file and calls local endpoints:

```text
/lol-gameflow/v1/gameflow-phase
/lol-champ-select/v1/session
```

Champion and summoner spell names are resolved from Data Dragon and cached under:

```text
~/.cache/lol-champ-select-recommender
```

## Notes

The League Client API is local and can change with League updates. Treat this as a prototype surface and keep the app read-only unless Riot's current policies and API behavior clearly allow a change.

## QUICK START
These commands use the current recommended configs from the repo. Replace the Riot IDs, lockfile path, and patch values as needed.

Personal match samples:

```bash
export RIOT_API_KEY="RGAPI-your-key-here"
python fetch_matches.py \
  --riot-id "GameName#TAG" \
  --region americas \
  --count 20 \
  --queue 420 \
  --match-type ranked \
  --output-dir data/raw \
  --force
```

You can also pass `--game-name` plus `--tag-line` instead of `--riot-id`.

Player pruning stats:

```bash
python collect_player_stats.py \
  --riot-id "GameName#TAG" \
  --riot-id "Other#TAG" \
  --region americas \
  --queue 420 \
  --match-type ranked \
  --matches-per-player 100 \
  --sleep 0.05 \
  --language en_US \
  --output data/processed/player_champion_role_stats.csv
```

Server-level ranked corpus:

```bash
python collect_ranked_matches.py \
  --platform br1 \
  --region americas \
  --queue RANKED_SOLO_5x5 \
  --match-queue 420 \
  --match-type ranked \
  --tiers DIAMOND MASTER GRANDMASTER CHALLENGER \
  --divisions I \
  --pages 30 \
  --page-mode random \
  --patch-mode latest \
  --language en_US \
  --max-players 500 \
  --matches-per-player 20 \
  --download-workers auto \
  --request-rate-limit 5 \
  --request-rate-burst 2 \
  --no-log-rate-limits \
  --seed 1 \
  --output-dir data/raw \
  --force
```

Aggregate server stats:

```bash
python aggregate_matches.py \
  --input-dir data/raw/matches \
  --output data/processed/champion_role_stats.csv \
  --priors-output data/processed/champion_role_priors.csv \
  --language en_US \
  --include-non-sr
```

Build champion features:

```bash
python build_champion_features.py \
  --language en_US \
  --output data/processed/champion_features.csv \
  --vocab-output data/processed/champion_feature_vocab.json
```

Build the raw draft dataset:

```bash
python build_draft_dataset.py \
  --input-dir data/raw/matches \
  --match-sources data/raw/match_sources.jsonl \
  --output data/processed/draft_dataset.jsonl \
  --language en_US \
  --include-non-sr \
  --allow-incomplete
```

Train the baseline model:

```bash
python train_draft_model.py \
  --dataset data/processed/draft_dataset.jsonl \
  --champion-features data/processed/champion_features.csv \
  --output-dir data/models/draft_transformer \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --use-hierarchy \
  --label-smoothing 0.03 \
  --coarse-loss-weight 0.3 \
  --lr-scheduler plateau \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 2 \
  --lr-scheduler-min-lr 1e-6 \
  --d-model 128 \
  --num-heads 1 \
  --num-layers 4 \
  --dim-feedforward 512 \
  --dropout 0.1 \
  --mask-probability 0.25 \
  --unk-probability 0.03 \
  --numeric-bins 10 \
  --val-split 0.15 \
  --champion-loss-weight-power 0.35 \
  --train-examples-per-row 4 \
  --seed 1 \
  --device cuda
```

Fine-tune on the latest patch while keeping some historical replay:

```bash
python train_draft_model.py \
  --dataset data/processed/draft_dataset.jsonl \
  --champion-features data/processed/champion_features.csv \
  --output-dir data/models/draft_transformer_finetune \
  --init-checkpoint data/models/draft_transformer/best.pt \
  --finetune-patch PATCH_VERSION \
  --finetune-historical-ratio 0.2 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --use-hierarchy \
  --label-smoothing 0.03 \
  --coarse-loss-weight 0.3 \
  --lr-scheduler plateau \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 2 \
  --lr-scheduler-min-lr 1e-6 \
  --d-model 128 \
  --num-heads 1 \
  --num-layers 4 \
  --dim-feedforward 512 \
  --dropout 0.1 \
  --mask-probability 0.25 \
  --unk-probability 0.03 \
  --numeric-bins 10 \
  --val-split 0.15 \
  --champion-loss-weight-power 0.35 \
  --train-examples-per-row 4 \
  --seed 1 \
  --device cuda
```

Live terminal inference:

```bash
python watch.py \
  --lockfile "/mnt/c/Riot Games/League of Legends/lockfile" \
  --host 127.0.0.1 \
  --interval 0.5 \
  --language en_US \
  --once \
  --no-clear \
  --role-priors data/processed/champion_role_priors.csv \
  --role-priors-queue 420 \
  --role-priors-patch PATCH_VERSION \
  --role-priors-min-games 1 \
  --model-checkpoint data/models/draft_transformer/best.pt \
  --champion-features data/processed/champion_features.csv \
  --recommendation-count 10 \
  --player-stats data/processed/player_champion_role_stats.csv \
  --champion-blacklist data/processed/champion_blacklist.txt \
  --debug-inference
```
