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

Soft requires 20+ games and 52%+ win rate overall. Hard also requires those thresholds in the listed role. Extrapolated variants allow fewer than 20 games if losses are below 9.6, including zero games. Whitelisted variants apply the global and role-specific blacklist.

Lists rank by historical win rate, then game count; zero-game entries come last. Displayed percentages are historical win rates. Champion features supply the champion catalog and names; if unavailable, the catalog falls back to champions present in the stats CSV.
