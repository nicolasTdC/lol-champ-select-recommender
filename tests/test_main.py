from __future__ import annotations

import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from lol_champ_select_recommender.__main__ import main, reset_debug_inference_log, write_debug_inference_log


class MainTest(unittest.TestCase):
    def test_stats_only_filters_profiles_and_skips_client_and_model(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            path.write_text(
                "player,riot_id,champion_id,champion_name,role,games,wins,losses\n"
                "Alice,Alice#BR1,1,Annie,middle,20,12,8\n"
                "Alice,Alice#BR1,2,Olaf,top,30,16,14\n"
                "Bob,Bob#BR1,1,Annie,middle,80,0,80\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("sys.argv", ["watch.py", "--stats-only", "--profiles", "Alice#BR1", "--player-stats", str(path), "--champion-features", str(Path(directory) / "missing.csv")]), patch("lol_champ_select_recommender.__main__.connect") as connect, patch("lol_champ_select_recommender.__main__.load_recommender") as model, redirect_stdout(output):
                self.assertEqual(main(), 0)
            connect.assert_not_called()
            model.assert_not_called()
            self.assertIn("Annie: 20 games, 12W/8L, WR 60.0%", output.getvalue())
            self.assertNotIn("100 games", output.getvalue())
            soft_list = output.getvalue().split("  Soft:\n", 1)[1].split("  Hard:", 1)[0]
            self.assertLess(soft_list.index("Olaf:"), soft_list.index("Annie:"))

    def test_write_debug_inference_log_appends_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "debug.log"

            write_debug_inference_log(path, phase="ChampSelect", lines=["Inference debug", "  token"])
            write_debug_inference_log(path, phase="Lobby", lines=["Inference debug: no live draft query available"])

            text = path.read_text(encoding="utf-8")

        self.assertIn("phase=ChampSelect", text)
        self.assertIn("Inference debug\n  token", text)
        self.assertIn("phase=Lobby", text)
        self.assertIn("Inference debug: no live draft query available", text)

    def test_reset_debug_inference_log_clears_previous_content(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "debug.log"
            write_debug_inference_log(path, phase="ChampSelect", lines=["old"])

            reset_debug_inference_log(path)
            write_debug_inference_log(path, phase="ChampSelect", lines=["new"])

            text = path.read_text(encoding="utf-8")

        self.assertNotIn("old", text)
        self.assertIn("new", text)


if __name__ == "__main__":
    unittest.main()
