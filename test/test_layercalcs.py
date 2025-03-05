import unittest
import numpy as np
from calculations.layer_calcs import blur_tiff_manual, met_ave

class TestLayerCalcs(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[1, 2, 3,4], [4, 5, 6, 6], [7, 8, 9,3]])
        self.blur_x = 1
        self.blur_y = 1

    def test_blur_tiff_manual(self):
        actual_output = blur_tiff_manual(self.array, self.blur_x, self.blur_y)
        expected_output = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 9.0]])

        np.testing.assert_array_equal(actual_output, expected_output)

    def test_met_ave(self):
        actual_output = met_ave(self.array, 2)

if __name__ == "__main__":
    unittest.main()
