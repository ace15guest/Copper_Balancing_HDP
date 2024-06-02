import unittest
import numpy as np
import os
from PIL import Image
from loading.img2array import array_to_bitmap

class TestBitmap(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        self.output_path = 'test.bmp'

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_array_to_bitmap(self):
        array_to_bitmap(self.array, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

        # Load the image to check if it was saved correctly
        img = Image.open(self.output_path)
        loaded_array = np.array(img)

        # Check if the saved image matches the original array
        np.testing.assert_array_equal(self.array, loaded_array)

if __name__ == '__main__':
    unittest.main()