import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from metetl.images.processing import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    def test_process_bytes_returns_png(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:, :, 1] = 255
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        processor = ImageProcessor(max_side=20)
        result = processor.process_bytes(encoded.tobytes())

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_integration_processor_with_mock_artwork(self):
        mock_artwork = Mock()
        mock_artwork.image = np.zeros((10, 10, 3), dtype=np.uint8)

        ok, encoded = cv2.imencode(".jpg", mock_artwork.image)
        self.assertTrue(ok)

        processor = ImageProcessor(max_side=10)
        result = processor.process_bytes(encoded.tobytes(), index=1, object_id="mock")

        self.assertTrue(result.startswith(b"\x89PNG"))
        mock_artwork.image.shape = (10, 10, 3)


if __name__ == "__main__":
    unittest.main()
