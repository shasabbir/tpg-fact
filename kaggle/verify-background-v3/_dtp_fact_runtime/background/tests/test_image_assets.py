import unittest
from io import BytesIO

from PIL import Image

from data.image_assets import inspect_image_bytes, perceptual_hash


class ImageAssetsTest(unittest.TestCase):
    def _image(self, color):
        buffer = BytesIO()
        Image.new("RGB", (32, 32), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_hashes_are_deterministic(self):
        data = self._image((255, 0, 0))
        details = inspect_image_bytes(data)
        self.assertEqual(details["image_phash"], perceptual_hash(data))
        self.assertEqual(len(details["image_sha256"]), 64)
        self.assertEqual(len(details["image_phash"]), 16)
        self.assertEqual(details["width"], 32)


if __name__ == "__main__":
    unittest.main()
