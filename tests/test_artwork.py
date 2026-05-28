import unittest

import numpy as np

from metetl.images.models import Artwork


class TestArtwork(unittest.TestCase):
    def test_rgb_to_grayscale_shape(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 20
        image[:, :, 2] = 30

        artwork = Artwork(image)
        gray = artwork.to_grayscale()

        self.assertEqual(gray.shape, (4, 5))
        self.assertTrue(np.issubdtype(gray.dtype, np.floating))

    def test_convolution_keeps_shape(self):
        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        kernel = Artwork.gaussian_kernel(size=3, sigma=1.0)

        result = Artwork.convolve2d_manual(image, kernel)

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
