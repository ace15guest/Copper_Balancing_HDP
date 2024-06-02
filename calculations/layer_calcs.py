import numpy as np

def blur_tiff_manual(array, blur_x = 2, blur_y = 2):

    """
    This function blurs the image starting from the top left corner of the image using a Gaussian filter.

    """
    # TODO: Add thieving option
    array_for_blur = np.zeros((array.shape[0] + 2*blur_x, array.shape[1] + 2*blur_y))
    array_for_blur[blur_x:blur_x + array.shape[0], blur_y:blur_y + array.shape[1]] = array
    new_array = np.zeros(array_for_blur.shape)

    for row in range(array_for_blur.shape[0]):
        for col in range(array_for_blur.shape[1]):
            if row < blur_x or col < blur_y or row >= array.shape[0]+blur_x or col >= array.shape[1]+blur_y:
                continue
            else:
                new_array[row, col] = 1/(2*blur_x+1)**2 *np.mean(array_for_blur[row-blur_x:row+blur_x+1, col-blur_y:col+blur_y+1])

    return new_array

from scipy.ndimage import gaussian_filter

def blur_tiff_gauss(array, sigma = 1):
    """
    This function blurs the image using a Gaussian filter.
    """
    # TODO: Add thieving option
    return gaussian_filter(array, sigma=sigma)
