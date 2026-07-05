from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .ddragon import load_static_data
from .lcu import LcuError, connect
from .modeling.draft_inference import DraftRecommender
from .render import render_session
from .roles import load_role_priors


def main() -> int:
    args = parse_args()

    try:
        connection, lockfile = connect(args.lockfile, args.host)
    except LcuError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    static_data = load_static_data(args.language)
    role_priors = load_role_priors(
        args.role_priors,
        queue_id=args.role_priors_queue,
        patch=args.role_priors_patch,
        min_total_games=args.role_priors_min_games,
    )
    recommender, model_status = load_recommender(args)
    model_lines = str(recommender.model).splitlines() if recommender else None
    lockfile_label = (
        f"{lockfile} -> {connection.base_url} ({connection.transport})"
        if lockfile
        else f"process args -> {connection.base_url} ({connection.transport})"
    )

    while True:
        try:
            phase = connection.gameflow_phase()
            session = connection.champ_select_session() if phase == "ChampSelect" else None
            recommendation_lines = (
                ["Recommendations", *recommender.lane_recommendation_lines()]
                if recommender and phase != "ChampSelect"
                else None
            )
            debug_lines = None
            if session and recommender:
                recommendation_lines = recommender.recommend_lines(
                    session,
                    static_data,
                    role_priors=role_priors,
                    top_k=args.recommendation_count,
                )
                if args.debug_inference:
                    write_debug_inference_log(
                        args.debug_inference_log,
                        phase=phase,
                        lines=recommender.debug_lines(
                            session,
                            static_data,
                            role_priors=role_priors,
                        ),
                    )
            output = render_session(
                phase=phase,
                session=session,
                static_data=static_data,
                lockfile_label=lockfile_label,
                role_priors=role_priors,
                model_status=model_status,
                model_lines=model_lines,
                recommendation_lines=recommendation_lines,
                debug_lines=debug_lines,
            )
        except LcuError as exc:
            output = f"League Champ Select Watcher\nError: {exc}"

        if not args.no_clear:
            _clear_screen()
        print(output, flush=True)

        if args.once:
            return 0

        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch live League champion select in the terminal.")
    parser.add_argument(
        "--lockfile",
        help="Path to League's lockfile. Overrides automatic detection and LOL_LOCKFILE.",
    )
    parser.add_argument(
        "--host",
        help="LCU host/IP. Default: 127.0.0.1, or LCU_HOST if set. Useful from WSL.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--language",
        default="en_US",
        help="Data Dragon language code. Example: en_US, pt_BR. Default: en_US",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal between updates.",
    )
    parser.add_argument(
        "--role-priors",
        default="data/processed/champion_role_priors.csv",
        help="Champion role-priors CSV path. Default: data/processed/champion_role_priors.csv",
    )
    parser.add_argument(
        "--role-priors-queue",
        type=int,
        default=420,
        help="Queue ID to load from role priors. Default: 420",
    )
    parser.add_argument(
        "--role-priors-patch",
        help="Patch to load from role priors. Default: latest patch in the CSV.",
    )
    parser.add_argument(
        "--role-priors-min-games",
        type=int,
        default=1,
        help="Minimum champion sample size before trusting role priors. Default: 1",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="data/models/draft_transformer/best.pt",
        help="Draft model checkpoint path. Default: data/models/draft_transformer/best.pt",
    )
    parser.add_argument(
        "--champion-features",
        default="data/processed/champion_features.csv",
        help="Champion feature CSV path for model inference. Default: data/processed/champion_features.csv",
    )
    parser.add_argument(
        "--recommendation-count",
        type=int,
        default=10,
        help="How many champion recommendations to show per open role. Default: 10",
    )
    parser.add_argument(
        "--player-stats",
        default="data/processed/player_champion_role_stats.csv",
        help="Per-player champion-role stats CSV for heuristic pruning. Default: data/processed/player_champion_role_stats.csv",
    )
    parser.add_argument(
        "--champion-blacklist",
        default="data/processed/champion_blacklist.txt",
        help="Text file with champion IDs or names to exclude from whitelisted pruning views. Default: data/processed/champion_blacklist.txt",
    )
    parser.add_argument(
        "--debug-inference",
        action="store_true",
        help="Write the live inference token sequence and decoded feature values to --debug-inference-log.",
    )
    parser.add_argument(
        "--debug-inference-log",
        default="data/logs/inference_debug.log",
        help="Path for --debug-inference output. Default: data/logs/inference_debug.log",
    )
    return parser.parse_args()


def _clear_screen() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def write_debug_inference_log(path: str | Path, *, phase: str, lines: list[str]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] phase={phase}\n")
        for line in lines:
            file.write(f"{line}\n")
        file.write("\n")


def load_recommender(args: argparse.Namespace) -> tuple[DraftRecommender | None, str]:
    candidates = [
        Path(args.model_checkpoint),
        Path("data/models/draft_transformer/best.pt"),
        Path("data/models/smoke-test/best.pt"),
    ]
    seen: set[Path] = set()

    for checkpoint in candidates:
        checkpoint = checkpoint.expanduser()
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        if not checkpoint.is_file():
            continue

        try:
            recommender = DraftRecommender.load(
                checkpoint,
                champion_features_path=args.champion_features,
                player_stats_path=args.player_stats,
                champion_blacklist_path=args.champion_blacklist,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"unavailable ({checkpoint}: {exc})"

        prune_status = recommender.prune_status()
        return recommender, f"loaded {checkpoint} | player-prune: {prune_status}"

    return None, "unavailable (no checkpoint found)"


if __name__ == "__main__":
    raise SystemExit(main())
