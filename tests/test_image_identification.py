import unittest

import numpy as np
from PIL import Image

from satquery.imaging import identify_image_type, spectral_masks


class ImageIdentificationTests(unittest.TestCase):
    def test_primary_colour_composite_is_identified(self):
        data = np.zeros((90, 90, 3), dtype=np.uint8)
        data[:30, :] = (240, 20, 20)
        data[30:60, :] = (20, 240, 20)
        data[60:, :] = (20, 20, 240)
        info = identify_image_type(Image.fromarray(data))
        self.assertEqual(info["type"], "rgb_composite")

    def test_natural_image_is_identified(self):
        data = np.full((90, 90, 3), (125, 142, 119), dtype=np.uint8)
        data[:, :40] = (65, 102, 130)
        info = identify_image_type(Image.fromarray(data))
        self.assertEqual(info["type"], "normal_optical")

    def test_dark_and_blue_water_are_detected(self):
        data = np.full((100, 100, 3), (145, 132, 104), dtype=np.uint8)
        data[15:55, 10:45] = (32, 64, 96)
        data[60:90, 55:90] = (28, 31, 34)
        image = Image.fromarray(data)
        mask = spectral_masks(image, identify_image_type(image)["type"])["water"]
        self.assertGreater(mask[20:50, 15:40].mean(), 0.8)
        self.assertGreater(mask[65:85, 60:85].mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
