from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from robokeeper.run_folders import labelled_path, parse_measurement, session_folder

DAY = datetime(2026, 11, 8, 16, 49)


class RunFolderTests(unittest.TestCase):
    def test_tests_are_numbered_per_day_and_reused(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(session_folder(root, now=DAY), Path(root) / "2026-11-08")
            first = session_folder(root, "chips left", now=DAY)
            self.assertEqual(first.name, "1_chips_left")
            first.mkdir(parents=True)
            second = session_folder(root, "fast ground", now=DAY)
            self.assertEqual(second.name, "2_fast_ground")
            second.mkdir()
            self.assertEqual(session_folder(root, "chips_left", now=DAY), first)
            self.assertEqual(session_folder(root, "chips", now=DAY).name, "3_chips")

    def test_measurement_labels(self):
        self.assertEqual(parse_measurement("left1 x-4.5 h2.4"),
                         ("left1_x-4.5ft_height2.4ft", {"x_ft": -4.5, "height_ft": 2.4, "name": "left1"}))
        self.assertEqual(parse_measurement("x4 height=3.25ft")[0], "x+4.0ft_height3.2ft")
        self.assertEqual(parse_measurement("missed chip")[0], "missed_chip")
        with self.assertRaises(ValueError):
            parse_measurement("  ")

    def test_labelled_path_keeps_the_original_name(self):
        path = Path("runs/2026-11-08/kick_20261108_164940.json")
        once = labelled_path(path, "left1_x-4.5ft")
        self.assertEqual(once.name, "left1_x-4.5ft__kick_20261108_164940.json")
        self.assertEqual(labelled_path(once, "left1_x-4.6ft").name, "left1_x-4.6ft__kick_20261108_164940.json")


if __name__ == "__main__":
    unittest.main()
