import unittest

from satquery.router import route


class RouterTests(unittest.TestCase):
    def test_grounding(self):
        selected = route("Highlight the water body", "single_optical")
        self.assertEqual(selected.task, "grounding")
        self.assertEqual(selected.target, "water")

    def test_change_mode_wins(self):
        self.assertEqual(route("Analyse this pair", "temporal_pair").task, "change_detection")

    def test_fusion(self):
        selected = route("Use optical and SAR to identify built-up areas", "optical_sar_pair")
        self.assertEqual(selected.task, "optical_sar_fusion")
        self.assertEqual(selected.target, "built_up")

    def test_unsupported(self):
        self.assertEqual(route("calculate orbital velocity", "single_optical").task, "unsupported")

    def test_rgb_colour_routing(self):
        for colour in ("red", "green", "blue"):
            selected = route(f"Highlight the {colour} pixels", "single_optical")
            self.assertEqual(selected.task, "grounding")
            self.assertEqual(selected.target, f"{colour}_region")


if __name__ == "__main__":
    unittest.main()
