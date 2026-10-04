# Offline Player Recommendations

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

Soft requires 20+ games and 52%+ win rate overall. Hard also requires those thresholds in the listed role. Extrapolated variants allow fewer than 20 games if losses are below 9.6, including zero games. Whitelisted variants apply the global and role-specific blacklist.

Lists rank by game count descending, then historical win rate; zero-game entries come last. Displayed percentages are historical win rates. Champion features supply the champion catalog and names; if unavailable, the catalog falls back to champions present in the stats CSV.
