import os
import PIL.ImageOps
from PIL import Image
import numpy as np


def bitmap_to_array(bitmap_path, ):
    # Open the image file
    print('Converting to array')
    array = None
    try:
        # TODO: Allow this to retry up to X times
        with Image.open(bitmap_path) as img:

                # Convert the image data to a numpy array
                gray = img.convert('L') # Convert the image to a gray scale
                array = np.array(gray)
                print(array.shape)
                print('Converted to array')
                img.close()
    except Exception as e:
        print(e)
    # os.remove(bitmap_path)
    array = -(array * 255.0 / np.max(array))+255
    return array

def open_multiple_bitmaps(folder):
    bit = {}
    for file in os.listdir(folder):
        f = os.path.join(folder, file)
        bit[file] = bitmap_to_array(f)

    return bit


def array_to_bitmap(array, output_path):
    # Convert the list of lists of tuples back to a numpy array
    array = np.array(array, dtype=np.uint8)

    # Create an image from the array
    img = Image.fromarray(array)

    # Save the image to a file
    img.save(output_path)

if __name__ == '__main__':
    file = r"C:\Users\6J2739897\Documents\projects\Projects4Others\HDP\Copper_Balancing_HDP\Assets\temp_tiff\P13.274x.png"
    bitmap_to_array(file)


