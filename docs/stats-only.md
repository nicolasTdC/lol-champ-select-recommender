# Offline Player Recommendations

## Local Web UI

```bash
python3 watch.py --web --port 8765
```

Open `http://localhost:8765`. Select profiles, ranked queue, and lane; the page shows All Lanes and the selected lane together. Recommended, Not Recommended, and Bans views include complete rankings with Show 20 more / Show all controls and champion search. Refresh reloads the CSVs. Champion portraits use Data Dragon when available. No League client, model, or API key is needed to view saved stats. A busy port automatically advances to the next free port.

Run without a League client connection, API key, or model checkpoint:

```bash
python3 watch.py --stats-only
```

Use selected profiles from the existing stats CSV:

```bash
python3 watch.py --stats-only --profiles "NICKNINJA#BR1" --player-stats data/processed/player_champion_role_stats.csv --champion-features data/processed/champion_features.csv --champion-blacklist data/processed/champion_blacklist.txt --recommendation-count 10
```

Multiple values after `--profiles` combine those accounts' games, wins, and losses. Omitting it combines all accounts in the CSV. This mode reads saved stats; refresh them with the existing player stats pipeline when needed.

Refresh both ranked queues (100 recent matches per account per queue):

```bash
python3 collect_player_stats.py --riot-id "NICKNINJA#BR1" --riot-id "kakashi2003#BR1" --riot-id "XXNAGATO1234#BR1" --riot-id "XXXMADARA123#BR1" --queue 420 440 --matches-per-player 100
```

Filter recommendations, including lane stats:

```bash
python3 watch.py --stats-only --ranked-queue all
python3 watch.py --stats-only --ranked-queue soloduo
python3 watch.py --stats-only --ranked-queue flex
```

The collector defaults to both queues. Use `--queue 420` or `--queue 440` to collect only one. New CSVs preserve `queue_id` per row; legacy CSVs without that column are treated as Solo/Duo. Collection replaces the stats CSV with the requested sample, so collect both queues to retain both views.

Add `--incremental` to preserve historical games and fetch new IDs until a cached game is reached:

```bash
python3 collect_player_stats.py --incremental --riot-id "NICKNINJA#BR1" --riot-id "kakashi2003#BR1" --riot-id "XXNAGATO1234#BR1" --riot-id "XXXMADARA123#BR1" --queue 420 440 --matches-per-player 1000
```

Incremental mode indexes `data/raw/matches` by account PUUID, unions new IDs with existing personal games, and rebuilds totals so overlap is not double-counted. It preserves CSV rows for unrequested profiles and queues. Keep the raw cache: historical CSV aggregates alone cannot identify duplicate games. All matching cached games (including corpus matches) enter the selected profiles' totals. If no known game is reached, pagination continues up to the requested limit. Without `--incremental`, the collector still replaces the selected dataset with the requested recent sample.

`--matches-per-player` accepts any positive count, including 1000. Match IDs are fetched in pages of at most 100 for each queue; collection stops early if fewer games are available.

The sample uses recent available matches without a patch or season filter. Match downloads use auto workers (up to 8), a shared rate limiter, and the corpus cache at `data/raw/matches`. Cached matches avoid new detail requests, including matches shared by multiple profiles. Progress shows downloaded, cached, and failed matches; rate-limit retries are logged automatically.

Optional collection controls:

```bash
python3 collect_player_stats.py --riot-id "NICKNINJA#BR1" --queue 420 440 --matches-per-player 1000 --download-workers auto --request-rate-limit 5 --matches-dir data/raw/matches --sleep 0
```

Use `--download-workers 1` for sequential downloads or another positive integer for explicit concurrency. `--request-rate-limit 0` disables local pacing; Riot's rate-limit retries still apply.

Soft requires 20+ games and 52%+ win rate overall. Hard also requires those thresholds in the listed role. Extrapolated variants allow fewer than 20 games if losses are below 9.6, including zero games. Whitelisted variants apply the global and role-specific blacklist.

Recommended and Not Recommended each print Soft variants once under All Lanes, and Hard variants under each lane. All Lanes whitelisted lists apply only global blacklist entries; per-lane whitelisted Hard lists also apply that lane's blacklist.

The Lane section also shows Not Recommended Hard and Soft lists, each the exact complement of its lane recommendation rule. Hard rejects fewer than 20 games or WR below 52%. Lane Soft accepts Hard-qualified lanes OR fewer than 20 games with losses below 9.6. Not Recommended Soft therefore includes only lanes with 20+ games below 52% WR, or fewer than 20 games with at least 10 losses. Rejected lanes sort by games descending, then WR ascending.

Lists rank by game count descending, then historical win rate; zero-game entries come last. Displayed percentages are historical win rates. Champion features supply the champion catalog and names; if unavailable, the catalog falls back to champions present in the stats CSV.

The Not Recommended section lists the exact complement of each recommendation filter using the same profiles, queue, sorting, and list limit. Strict complements include champions with fewer than 20 games or win rate below 52%. Extrapolated complements include champions with 20+ games below 52% WR, or fewer than 20 games with at least 10 losses. Hard complements reject if either the overall or role condition fails. Whitelisted complements also include champions excluded by the blacklist. These are complements of the rules, not of the displayed top-N lists; insufficient data in a strict complement does not mean a champion is proven bad.

## Offline Bans

The collector also writes `data/processed/player_enemy_champion_role_stats.csv`. Each encounter records your profile's win/loss against an enemy champion, grouped by the enemy's role and queue. These are your outcomes against champions, not enemy account mastery or win rates. A match contributes one encounter for each enemy; multiple selected profiles in the same match contribute separate profile encounters.

Offline mode automatically shows Ban Recommendations with the same profile and queue filters. All Roles shows Soft lists: personal WR below 52% over 20+ encounters. Each enemy lane shows Hard lists using only encounters against that champion in that lane, independently of overall results. Extrapolated variants additionally flag fewer than 20 encounters with at least 10 losses. Unseen champions and insufficient samples alone do not qualify. Bans sort by encounters descending, then personal WR ascending, and ignore the pick blacklist.

Rebuild enemy stats from all matching cached ranked matches without API requests:

```bash
python3 collect_player_stats.py --cached-only --riot-id "NICKNINJA#BR1" --riot-id "kakashi2003#BR1" --riot-id "XXNAGATO1234#BR1" --riot-id "XXXMADARA123#BR1" --queue 420 440
python3 watch.py --stats-only --ranked-queue all
```

Cached-only rebuilding matches participant Riot IDs stored in the raw files, so games under previous account names may not be included. It uses all matching cache files, rather than the latest-N limit, and leaves your pick stats CSV intact. Override `--opponent-output` during collection and `--opponent-stats` during offline display to use another path.

## ARAM

ARAM is a separate queue filter in the web UI and terminal. It uses the same overall champion thresholds and expandable rankings, without lane recommendations or ban lists. All Ranked continues to include only Solo/Duo and Flex.

```bash
python3 collect_player_stats.py --incremental --riot-id "NICKNINJA#BR1" --riot-id "kakashi2003#BR1" --riot-id "XXNAGATO1234#BR1" --riot-id "XXXMADARA123#BR1" --queue 450 --matches-per-player 1000
python3 watch.py --stats-only --ranked-queue aram
python3 watch.py --web
```

Do not pass `--match-type ranked` for ARAM. The collector now defaults to no type filter because queue IDs determine the mode. Incremental ARAM collection preserves existing ranked stats. Global blacklist entries apply to ARAM; ranked lane-specific entries do not.
