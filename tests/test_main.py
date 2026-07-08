from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lol_champ_select_recommender.__main__ import reset_debug_inference_log, write_debug_inference_log


class MainTest(unittest.TestCase):
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
