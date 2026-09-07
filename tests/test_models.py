import unittest

from PIL import Image, ImageDraw

from satquery.models import analyze_change, analyze_fusion, analyze_grounding


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.optical = Image.new("RGB", (100, 100), (95, 145, 75))
        ImageDraw.Draw(self.optical).rectangle((0, 55, 100, 100), fill=(35, 95, 150))

    def test_water_grounding_returns_evidence(self):
        result = analyze_grounding(self.optical, "water")
        self.assertIn("evidence_image", result)
        self.assertGreater(result["metrics"]["estimated_coverage_percent"], 20)

    def test_rgb_regions_are_grounded(self):
        image = Image.new("RGB", (90, 60), (25, 25, 25))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 29, 59), fill=(230, 30, 30))
        draw.rectangle((30, 0, 59, 59), fill=(30, 230, 30))
        draw.rectangle((60, 0, 89, 59), fill=(30, 30, 230))
        for target in ("red_region", "green_region", "blue_region"):
            result = analyze_grounding(image, target)
            self.assertGreater(result["metrics"]["estimated_coverage_percent"], 25)
            self.assertIn("evidence_image", result)

    def test_change_detects_modified_region(self):
        after = self.optical.copy()
        ImageDraw.Draw(after).rectangle((20, 10, 50, 40), fill=(230, 230, 230))
        result = analyze_change(self.optical, after)
        self.assertGreater(result["metrics"]["changed_area_percent"], 1)

    def test_fusion_contract(self):
        sar = self.optical.convert("L").convert("RGB")
        result = analyze_fusion(self.optical, sar, "water")
        self.assertEqual(result["metrics"]["target"], "water")
        self.assertIn("fusion_weights", result["metrics"])


if __name__ == "__main__":
    unittest.main()
