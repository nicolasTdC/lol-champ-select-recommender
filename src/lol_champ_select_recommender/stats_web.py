from __future__ import annotations

import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .modeling.draft_inference import load_champion_blacklist
from .modeling.player_pruning import load_player_prune_index, PruneStats, passes_extrapolated_threshold
from .roles import POSITION_ORDER, ROLE_NAMES


def read_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def stats_payload(args, query):
    queue = query.get("queue", [args.ranked_queue])[0]
    if queue not in ("all", "soloduo", "flex"):
        raise ValueError("Invalid ranked queue")
    profiles = query.get("profile", args.profiles)
    rows = read_rows(args.player_stats)
    available = sorted({row.get("riot_id") or row.get("player") for row in rows} - {None, ""})
    features = {int(row["champion_id"]): row for row in read_rows(args.champion_features)}
    for row in rows + read_rows(args.opponent_stats):
        features.setdefault(int(row["champion_id"]), {"champion_name": row.get("champion_name", row["champion_id"])})
    blacklist = load_champion_blacklist(args.champion_blacklist, features)
    index = load_player_prune_index(args.player_stats, profiles=profiles, ranked_queue=queue)
    enemy = load_player_prune_index(args.opponent_stats, profiles=profiles, ranked_queue=queue)
    zero = PruneStats(0, 0, 0)

    def entry(cid, stats):
        feature = features.get(cid, {})
        key, version = feature.get("champion_key"), feature.get("version")
        return {"id": cid, "name": feature.get("champion_name", str(cid)), "games": stats.games,
                "wins": stats.wins, "losses": stats.losses, "wr": stats.win_rate if stats.games else None,
                "image": f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{key}.png" if key and version else None}

    result = {"profiles": available, "selected_profiles": profiles or available, "scopes": {}, "lanes": [],
              "has_stats": bool(index and index.overall_by_champion), "has_bans": bool(enemy and enemy.overall_by_champion)}
    for role in (None, *POSITION_ORDER):
        scope = {"name": ROLE_NAMES[role] if role else "All Lanes", "picks": [], "bans": []}
        if index:
            for extrapolated in (False, True):
                for whitelisted in (False, True):
                    overall_pass = index.passes_soft_extrapolated if extrapolated else index.passes_soft
                    role_pass = index.passes_hard_extrapolated if extrapolated else index.passes_hard
                    accepted, rejected = [], []
                    for cid in features:
                        stats = (index.role_stats(cid, role) if role else index.overall_stats(cid)) or zero
                        passes = overall_pass(cid) and (role is None or role_pass(cid, role))
                        passes = passes and (not whitelisted or not blacklist.blocks(cid, role))
                        (accepted if passes else rejected).append(entry(cid, stats))
                    for items in (accepted, rejected):
                        items.sort(key=lambda row: (-row["games"], -(row["wr"] or 0), row["name"]))
                    label = ("Whitelisted " if whitelisted else "") + ("Extrapolated " if extrapolated else "") + ("Hard" if role else "Soft")
                    scope["picks"].append({"label": label, "recommended": accepted, "rejected": rejected})
        if enemy:
            bucket = enemy.by_role_by_champion.get(role, {}) if role else enemy.overall_by_champion
            for extrapolated in (False, True):
                banned = [entry(cid, stats) for cid, stats in bucket.items() if
                          (stats.games >= 20 and stats.win_rate < .52) or
                          (extrapolated and stats.games < 20 and stats.losses >= 20 * (1 - .52))]
                banned.sort(key=lambda row: (-row["games"], row["wr"], row["name"]))
                scope["bans"].append({"label": ("Extrapolated " if extrapolated else "") + ("Hard" if role else "Soft"), "rows": banned})
        result["scopes"][role or "all"] = scope
    if index:
        for role in POSITION_ORDER:
            stats = index.by_role.get(role, zero)
            result["lanes"].append({"role": role, "name": ROLE_NAMES[role], "games": stats.games,
                                    "wr": stats.win_rate if stats.games else None,
                                    "hard": stats.games >= 20 and stats.win_rate >= .52,
                                    "soft": passes_extrapolated_threshold(stats)})
    return result


def serve(args):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    body = Path(__file__).with_name("stats_ui.html").read_bytes()
                    content_type = "text/html; charset=utf-8"
                elif request.path == "/api/stats":
                    body = json.dumps(stats_payload(args, parse_qs(request.query))).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(400, str(exc))

    port = args.port
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as exc:
            if exc.errno != 98 or port >= args.port + 10:
                raise
            port += 1
    print(f"Offline stats UI: http://localhost:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
