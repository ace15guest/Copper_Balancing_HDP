from PIL import Image
import numpy as np


def bitmap_to_array(bitmap_path):
    # Open the image file
    with Image.open(bitmap_path) as img:
        # Convert the image data to a numpy array
        array = np.array(img)
    return array


def array_to_bitmap(array, output_path):
    # Convert the list of lists of tuples back to a numpy array
    array = np.array(array, dtype=np.uint8)

    # Create an image from the array
    img = Image.fromarray(array)

    # Save the image to a file
    img.save(output_path)


array = bitmap_to_array(
    r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\Design_Tif_Rip50\Design_Tif_Rip50\l7_signal_1oz_0.tif")
array_to_bitmap(array,
                r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\Design_Tif_Rip50\Design_Tif_Rip50\l7_signal_1oz_0_test.tif")
