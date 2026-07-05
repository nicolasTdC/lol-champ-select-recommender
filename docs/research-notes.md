# Research Notes

This project is a personal research tool, not a public product. That makes local experimentation easier, but the data pipeline should still be reproducible and avoid depending on fragile page scraping when an official source or documented API is available.

## Data Direction

Preferred long-term source: Riot API match data.

Useful Riot sources:

- LCU: live champ-select picks, bans, visible spells, hovers, and phase.
- Data Dragon: static champion, spell, item, rune, and asset metadata. Champion `tags`, `info`, and base stats should feed model metadata embeddings and help with new-champion cold starts.
- Match-V5: historical match participants, champion IDs, role fields, teams, outcomes, items, summoner spells, and timelines.
- League-V4: seed high-ranked players by queue/tier/division.
- Account-V1 / Summoner-V4: resolve Riot IDs, PUUIDs, and summoner IDs.
- Champion-Mastery-V4: player comfort signals.

Stats sites such as OP.GG, U.GG, LeagueOfGraphs, and LoLTheory are useful references for product shape and metrics, but the core dataset should eventually be our own derived aggregate from Riot match data.

## LoLTheory Reference

LoLTheory appears to combine a desktop app/overlay with web analyzers:

- real-time item recommendations that adapt to enemy builds and the player's current state
- team-based champion recommendations for champion + role combinations
- team comp analysis using ally synergy, enemy counters, and draft weaknesses
- role-filtered counters and synergies
- rank filters such as Platinum+
- app-based automatic import of ally/enemy champions

Reference URLs:

- https://loltheory.gg/
- https://loltheory.gg/lol/team-comp-analyzer/solo-queue
- https://loltheory.gg/lol/synergies
- https://loltheory.gg/lol/counters
- https://www.reddit.com/r/leagueoflegends/comments/1gndb6x/loltheory_realtime_item_recommendations_that/

Useful design takeaways for this project:

- Treat champion recommendations as champion-role recommendations, not just champion recommendations.
- Split the score into interpretable components: meta strength, player comfort, ally synergy, enemy matchup/counter value, team needs, and confidence/sample size.
- Keep role inference probabilistic where possible. Example: Maokai can be top/jungle/support, so the resolver should use team fit and pick-rate priors.
- Show multiple recommendations with reasons instead of one directive answer.
- Keep the live watcher as the ingestion layer, then make recommendation scoring independent from the terminal renderer.

## Immediate Modeling Target

Replace the temporary curated champion-role map with a generated table:

```text
patch
region
rank_bucket
queue_id
champion_id
role
games
wins
pick_rate
win_rate
ban_rate
```

Then use that table for:

- enemy role inference
- champion-role meta strength
- ally synergy
- enemy counters
- blind-pick safety
- recommendation confidence

## Server-Level Collection

Single-account match history is useful for testing but too biased for recommendations. Broader server stats should be collected by:

```text
League-V4 ranked entries for a platform server
-> seed PUUIDs from ranked players
-> Match-V5 recent ranked match IDs for each PUUID
-> raw Match-V5 match JSON cache
-> aggregate champion-role, matchup, and synergy tables
```

Use bounded crawls while on a development API key. Example: 50 seed players * 5 matches each before dedupe is already hundreds of API requests.
