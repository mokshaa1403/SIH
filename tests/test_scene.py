import unittest

from PIL import Image

from satquery.models.scene import extract_features


class SceneFeatureTests(unittest.TestCase):
    def test_feature_shape_is_stable(self):
        feature = extract_features(Image.new("RGB", (64, 64), "blue"))
        self.assertEqual(feature.shape, (250,))


if __name__ == "__main__":
    unittest.main()
