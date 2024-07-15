import os
import time
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from PIL import Image as pImage
from pygerber.gerberx3.api._v2 import OnParserErrorEnum
from pygerber.gerberx3.api.v2 import GerberFile
from pathlib import Path
from file_handling import find_inkscape
import subprocess
from wand.image import Image


def gerber_to_svg_python(file_path, save_path="output.svg", scale=10):
    GerberFile.from_file(file_path).parse(on_parser_error=OnParserErrorEnum.Ignore).render_svg(save_path, scale=scale)
    return


def gerber_to_svg_gerbv(file_path, save_name, D=50):
    error_log_path = fr"Assets\temp\{save_name}.log"
    Path(r"Assets\temp").mkdir(exist_ok=True, parents=True)
    Path(r"Assets\temp_svg").mkdir(exist_ok=True, parents=True)
    svg_name = fr"Assets\temp_svg\{save_name}.svg"
    cmd_line = fr'Assets\gerbv\gerbv -x svg -o "{svg_name}" "{file_path}" 2> {error_log_path}'
    subprocess.Popen(cmd_line)
    return svg_name, error_log_path


def svg_to_tiff(input_svg, output_tiff, width=None, height=None, error_log_path=None, res=100):
    with Image(filename=input_svg, resolution=res) as img:
        if width and height:
            img.resize(width, height)
        # img.format = 'tiff'
        img.save(filename=output_tiff)
        print("Saved tiff")
        os.remove(input_svg)
        # os.remove(error_log_path)


def svg_to_tiff_inkscape(input_svg, output_tiff, width=None, height=None, error_log_path=None, res=100):
    inkscape_path = find_inkscape()

    cmd_line = fr'"{inkscape_path}" -z -e "{output_tiff}" -w {width} -h {height} "{input_svg}"'
    subprocess.Popen(cmd_line)
    return


def gerber_to_274x(file_path, save_path):
    command = f'Assets\gerbv\gerbv -x rs274x -o "{save_path}" "{file_path}"'
    subprocess.Popen(command)
    return

def gerber_to_png_gerbv(gerb_file, save_folder, save_path, dpi=1500, scale=1, error_log_path="error.log", outline_file=None):
    """

    :param gerb_file:
    :param save_path:
    :param dpi: 0-2000
    :return:
    """
    save_path = f"{save_path}.tif"
    if outline_file is None:
        command = f'Assets\gerbv\gerbv -x png -a -D {dpi} -o "{save_path}" "{gerb_file}" 2> {error_log_path}.txt'
    else:
        command = f'Assets\gerbv\gerbv -x png -a -D {dpi} -o "{save_path}" "{gerb_file}" "{outline_file}" 2> {error_log_path}.txt'
    subprocess.Popen(command)
    return save_path
def gerber_to_pdf_gerbv(file_path, save_folder, save_path, D=50):
    """

    :param file_path:
    :param save_path:
    :param D: 0-750
    :return:
    """
    Path(save_folder).mkdir(exist_ok=True, parents=True)
    command = f'Assets\gerbv\gerbv -x pdf -D {D} -o "{save_path}.pdf" "{file_path}"'
    subprocess.Popen(command)
    return


def pdf_page_to_array(pdf_path, page_number=0, dpi=100):
    doc = fitz.open(pdf_path)

    # Select a specific page
    page = doc.load_page(page_number)

    # Render the page as an image at the specified DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi , dpi))

    # Convert the pixmap to a NumPy array
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)

    # Reshape the array to match the image dimensions
    img_array = img_array.reshape((pix.height, pix.width, pix.n))
    img = pImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    tiff_path = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\temp_tiff\test.tiff"
    img.save(tiff_path, "TIFF")
    # Close the PDF document
    doc.close()

    return img_array


def array_to_bitmap(array, output_path):
    # Convert the list of lists of tuples back to a numpy array
    array = np.array(array, dtype=np.uint8)

    # Create an image from the array
    img = pImage.fromarray(array)

    # Save the image to a file
    img.save("out.bmp")

def check_tiff_dimensions(folder_path):
    dimensions = None
    all_same_size = True

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):  # Ensure we're only checking TIFF files
            file_path = os.path.join(folder_path, file_name)
            with pImage.open(file_path) as img:
                if dimensions is None:
                    dimensions = img.size  # Set the initial dimensions to compare against
                elif img.size != dimensions:
                    all_same_size = False
                    print(f"File {file_name} has different dimensions: {img.size} compared to the initial {dimensions}")
                    break  # Exit the loop early as we've found an inconsistency

    if all_same_size:
        return True
    else:
        return False

