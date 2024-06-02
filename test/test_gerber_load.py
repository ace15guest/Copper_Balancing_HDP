import unittest
import os
from PIL import Image
from loading.gerber_load import gerber_to_svg, svg_to_tiff

class TestGerberLoad(unittest.TestCase):
    def setUp(self):
        # TODO: Get Example Gerber File
        self.gerber_path = 'path/to/your/test.gerber'
        self.svg_path = 'test_output.svg'
        self.tiff_path = 'test_output.tiff'

    def tearDown(self):
        if os.path.exists(self.svg_path):
            os.remove(self.svg_path)
        if os.path.exists(self.tiff_path):
            os.remove(self.tiff_path)

    def test_gerber_to_svg(self):
        gerber_to_svg(self.gerber_path, self.svg_path)
        self.assertTrue(os.path.exists(self.svg_path))

    def test_svg_to_tiff(self):
        # Assuming that gerber_to_svg works correctly
        gerber_to_svg(self.gerber_path, self.svg_path)
        svg_to_tiff(self.svg_path, self.tiff_path)
        self.assertTrue(os.path.exists(self.tiff_path))

        # Load the image to check if it was saved correctly
        img = Image.open(self.tiff_path)
        self.assertEqual(img.format, 'TIFF')

if __name__ == '__main__':
    unittest.main()