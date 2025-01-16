import numpy as np


def blur_tiff_manual(array, blur_x=2, blur_y=2):
    """
    This function blurs the image starting from the top left corner of the image using a Gaussian filter.

    """
    # TODO: Add thieving option
    array_for_blur = np.zeros((array.shape[0] + 2 * blur_x, array.shape[1] + 2 * blur_y))
    array_for_blur[blur_x:blur_x + array.shape[0], blur_y:blur_y + array.shape[1]] = array
    new_array = np.zeros(array_for_blur.shape)

    for row in range(array_for_blur.shape[0]):
        for col in range(array_for_blur.shape[1]):
            if row < blur_x or col < blur_y or row >= array.shape[0] + blur_x or col >= array.shape[1] + blur_y:
                continue
            else:
                new_array[row, col] = 1 / (2 * blur_x + 1) ** 2 * np.mean(
                    array_for_blur[row - blur_x:row + blur_x + 1, col - blur_y:col + blur_y + 1])

    return new_array


from scipy.ndimage import gaussian_filter


def blur_tiff_gauss(array: np.array, sigma=1):
    """
    This function blurs the image using a Gaussian filter.
    """
    # TODO: Add thieving option
    return gaussian_filter(array, sigma=sigma)

def box_blur(array, radius=1):
    """Applies a box blur to a 2D numpy array.

    Args:
        array (numpy.ndarray): The 2D array to blur.
        radius (int): The radius of the blur kernel.

    Returns:
        numpy.ndarray: The blurred array.
    """

    height, width = array.shape
    blurred_array = np.zeros_like(array)

    for i in range(height):
        for j in range(width):
            # Calculate the bounds of the kernel
            i_min = max(0, i - radius)
            i_max = min(height, i + radius + 1)
            j_min = max(0, j - radius)
            j_max = min(width, j + radius + 1)

            # Extract the kernel and calculate the average
            kernel = array[i_min:i_max, j_min:j_max]
            blurred_array[i, j] = np.mean(kernel)

    return blurred_array

def median_blur(img, kernel_size_from_center):
    # Check if kernel size is odd
    kernel_size_from_center = kernel_size_from_center * 2 + 1

    # Pad the image to handle borders
    pad_size = kernel_size_from_center // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    # Create output image
    blurred_img = np.zeros_like(img)

    # Apply median filter
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighborhood = padded_img[i:i + kernel_size_from_center, j:j + kernel_size_from_center]
            blurred_img[i, j] = np.median(neighborhood)

    return blurred_img

def blur_algo(img_array: np.array, x_subset: int, y_subset: int):
    """
    This algorithm employs the idea of only changing the last column
    """
    temp_array = None

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            try:
                row_start = i - y_subset if i - y_subset >= 0 else 0
                row_end = i + y_subset + 1
                col_start = j - x_subset if j - x_subset >= 0 else 0
                col_end = j + x_subset + 1

                temp_array = img_array[row_start:row_end, col_start:col_end]

            except:
                pass
