"""Input-normalization semantics; no PyTorch/GPU required."""
import unittest

import numpy as np

from mcbench.baseline import normalize


class NormalizationTests(unittest.TestCase):
    def test_positive_percentiles_exclude_zero_background(self):
        # Positive values1..201 have linear p1=3 and p99.5=200.
        image = np.concatenate((np.zeros(1000), np.arange(1, 202))).reshape(1, -1)
        result = normalize(image)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result[0, 1002], 0)
        self.assertAlmostEqual(float(result[0, 1099]), 97/197, places=6)
        self.assertEqual(result[0, 1199], 1)
        self.assertEqual(result[0, 1200], 1)
        self.assertTrue((result[0, :1000] == 0).all())

    def test_empty_and_constant_ranges_are_zero(self):
        for image in (np.zeros((3, 4)), np.full((3, 4), 7), np.full((3, 4), -1)):
            self.assertFalse(normalize(image).any())

    def test_positive_linear_rescaling_preserves_representation(self):
        image = np.arange(120, dtype=np.float32).reshape(10, 12)
        np.testing.assert_allclose(normalize(image), normalize(image*100), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
