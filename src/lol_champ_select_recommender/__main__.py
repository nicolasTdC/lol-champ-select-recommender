from __future__ import annotations

import argparse
import os
import sys
import time

from .ddragon import load_static_data
from .lcu import LcuError, connect
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
    lockfile_label = (
        f"{lockfile} -> {connection.base_url} ({connection.transport})"
        if lockfile
        else f"process args -> {connection.base_url} ({connection.transport})"
    )

    while True:
        try:
            phase = connection.gameflow_phase()
            session = connection.champ_select_session() if phase == "ChampSelect" else None
            output = render_session(
                phase=phase,
                session=session,
                static_data=static_data,
                lockfile_label=lockfile_label,
                role_priors=role_priors,
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
        default=5,
        help="Minimum champion sample size before trusting role priors. Default: 5",
    )
    return parser.parse_args()


def _clear_screen() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    raise SystemExit(main())
