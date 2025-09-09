import numpy as np
import re
import os

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

def met_ave_old(img_array, radius):
    """
     Calculate the mean value of a given image array within a specified radius. This was what the MetAve algorithm used

    :param img_array: The input image array.
    :param radius: The radius around each pixel to consider for mean calculation.
    :return: numpy.ndarray: The resulting mean array with the same dimensions as the input image array.
    """
    # padded_array = np.pad(array=img_array, pad_width=radius, mode='constant', constant_values=0)

    # Create an array of zeroes
    result = np.zeros_like(img_array, dtype=float)
    normalization_factor = (2*radius+1)**2

    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):

            # Find the min/max x and y coords accounting for boundaries which will only consider
            x_min, x_max = max(0, x-radius), min(img_array.shape[0], img_array.shape[0]+radius)
            y_min, y_max = max(0, y-radius), min(img_array.shape[1], img_array.shape[1]+radius)

            area = img_array[x_min:x_max+1, y_min:y_max+1]
            result[x, y] = np.sum(area)/normalization_factor

    return result

# def met_ave(img_array, radius):
#     """
#     Optimized mean calculation using an integral image for fast local averaging.
#
#     :param img_array: 2D numpy array (image).
#     :param radius: Neighborhood radius for averaging.
#     :return: 2D numpy array with local means.
#     """
#     h, w = img_array.shape
#     normalization_factor = (2 * radius + 1) ** 2
#
#     # Compute integral image (cumulative sum)
#     integral = np.pad(img_array, ((1, 0), (1, 0)), mode='constant', constant_values=0).cumsum(axis=0).cumsum(axis=1)
#
#     # Compute sums using the integral image
#     x1, y1 = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
#     x2, y2 = np.clip(x1 + radius + 1, 0, w), np.clip(y1 + radius + 1, 0, h)
#     x1, y1 = np.clip(x1 - radius, 0, w), np.clip(y1 - radius, 0, h)
#
#     result = (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]) / normalization_factor
#     return result


import numpy as np

def met_ave(img_array: np.ndarray, radius: int, *, ignore_zeros: bool = True, empty_value=np.nan) -> np.ndarray:
    """
    Local mean via integral images, excluding invalid pixels from the average.

    - Excludes NaNs always.
    - If ignore_zeros=True (default), excludes zeros as well (treats them like "no data"/padding).
    - Divides by the number of valid pixels actually inside the window (handles edges correctly).
    - If a window has no valid pixels, returns `empty_value` (default: np.nan).

    Parameters
    ----------
    img_array : (H, W) array-like
        Input image.
    radius : int
        Half-window radius; window size = (2*radius+1)^2.
    ignore_zeros : bool, default True
        Whether to exclude zeros from the mean.
    empty_value : scalar, default np.nan
        Value to place where a window has no valid pixels.

    Returns
    -------
    (H, W) float array
        Local means with invalids excluded.
    """
    a = np.asarray(img_array, dtype=np.float64)
    h, w = a.shape

    # Valid mask: finite and (nonzero if requested)
    valid = np.isfinite(a)
    if ignore_zeros:
        valid &= (a != 0)

    # Zero out invalids for the value-sum integral
    vals = np.where(valid, a, 0.0)

    # Build integral images for values and counts (pad 1 row/col up/left for easy box sums)
    def integral_image(x):
        return np.pad(x, ((1, 0), (1, 0)), mode='constant', constant_values=0).cumsum(axis=0).cumsum(axis=1)

    I_vals = integral_image(vals)
    I_cnts = integral_image(valid.astype(np.int32))

    # Coordinates for each pixel's window
    x0, y0 = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x1 = np.clip(x0 - radius, 0, w)
    y1 = np.clip(y0 - radius, 0, h)
    x2 = np.clip(x0 + radius + 1, 0, w)
    y2 = np.clip(y0 + radius + 1, 0, h)

    # Box sums via integral images
    sum_vals = I_vals[y2, x2] - I_vals[y1, x2] - I_vals[y2, x1] + I_vals[y1, x1]
    cnt_vals = I_cnts[y2, x2] - I_cnts[y1, x2] - I_cnts[y2, x1] + I_cnts[y1, x1]

    # Safe divide; where no valid pixels, put empty_value
    out = np.full((h, w), empty_value, dtype=np.float64)
    np.divide(sum_vals, cnt_vals, out=out, where=(cnt_vals > 0))

    return out


import os
import re



def scan_gerber_extrema(folder_path):
    COORD_RE = re.compile(r'^(?:G01)?(?:X(-?\d+))?(?:Y(-?\d+))?D0[13]\*$')
    FORMAT_RE = re.compile(r'%FS[L|T]X(\d)(\d)Y(\d)(\d)\*%')
    file_name = ""
    units = "mm"
    int_digits, dec_digits = 2, 4
    x_vals, y_vals = [], []
    xpoints, ypoints = {}, {}

    for file_path in os.listdir(folder_path):
        if not file_path.lower().endswith(('.gbr', '.ger', '.274x')):
            continue
        full_path = os.path.join(folder_path, file_path)
        try:
            with open(full_path, 'r') as f:
                for line in f:
                    if '%MOIN' in line:
                        units = "in"
                    elif '%MOMM' in line:
                        units = "mm"
                    elif line.startswith('%FS'):
                        match = FORMAT_RE.search(line)
                        if match:
                            int_digits = int(match.group(1))
                            dec_digits = int(match.group(2))

                    match = COORD_RE.match(line)
                    if match:
                        x_str = match.group(1)
                        y_str = match.group(2)
                        if x_str:
                            x = int(x_str) / (10 ** dec_digits)
                            x_vals.append(x)
                            xpoints[x] = x_str
                        if y_str:
                            y = int(y_str) / (10 ** dec_digits)
                            y_vals.append(y)
                            ypoints[y] = y_str
                        file_name = file_path
        except:
            continue

    return {
        "file_path": os.path.join(folder_path, file_name),
        "unit": units,
        "xmin": min(x_vals),
        "xmax": max(x_vals),
        "ymin": min(y_vals),
        "ymax": max(y_vals),
        "xval_dict": xpoints,
        "yval_dict": ypoints
    }

