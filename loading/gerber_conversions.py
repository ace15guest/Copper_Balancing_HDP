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

def gerber_to_png_gerbv_windows_only(gerb_file, save_folder, save_path, dpi=1500, scale=1, error_log_path="error.log", outline_file=None):
    """
    Convert Gerber file to PNG using Gerbv.
    Parameters:
    - gerb_file (str): Path to the Gerber file.
    - save_folder (str): Path to the folder where the PNG file will be saved.
    - save_path (str): Name of the PNG file.
    - dpi (int, optional): DPI for the PNG file. Default is 1500.
    - scale (int, optional): Scale factor for the PNG file. Default is 1.
    - error_log_path (str, optional): Path to the error log file. Default is "error.log".
    - outline_file (str, optional): Path to the outline file. Default is None.

    Returns:
    str: Path to the saved PNG file.
    """
    save_path = f"{save_path}.tif"
    if outline_file is None:
        command = f'Assets\gerbv\gerbv -x png -a -D {dpi} -o "{save_path}" "{gerb_file}" 2> {error_log_path}.txt'
    else:
        # Merge the gerber file with the outline file
        command = f'Assets\gerbv\gerbv -x png -a -D {dpi} -o "{save_path}" "{gerb_file}" "{outline_file}" 2> {error_log_path}.txt'
    subprocess.Popen(command)
    return save_path

import os
import shutil
import subprocess
from pathlib import Path

import os
import shutil
import subprocess
from pathlib import Path

def gerber_to_png_gerbv(
    gerb_file,
    save_folder_temp,
    save_name,                 # no extension
    dpi=1500,
    outline_file=None,
    log_path=None,
    wait=True,                # async by default
    anti_alias=True
):
    # 1) Normalize paths (handle '~', make absolute)
    print(save_folder_temp, save_name)
    gerb_file = Path(gerb_file).expanduser().resolve()
    save_folder_new = Path(save_folder_temp)
    outline = Path(outline_file).expanduser().resolve() if outline_file else None

    Path(save_folder_temp).mkdir(parents=True, exist_ok=True)
    out_name = f"{save_name}.png"                 # pass just the name
    out_png  = f"{save_folder_temp}/{out_name}"             # this is where it will end up
    print(out_png)
    # 2) Locate gerbv
    gerbv = shutil.which("gerbv")
    if gerbv is None:
        candidate = Path("Assets") / "gerbv" / ("gerbv.exe" if os.name == "nt" else "gerbv")
        if candidate.exists():
            gerbv = str(candidate)
        else:
            raise FileNotFoundError("gerbv not found in PATH or Assets/gerbv/")

    # 3) Build argv (no shell), and set cwd=save_folder so output goes there
    cmd = [gerbv, "-x", "png", "-D", str(dpi)]
    if anti_alias:
        cmd.append("-a")
    cmd += ["-o", out_name, str(gerb_file)]
    if outline:
        cmd.append(str(outline))

    # 4) Run
    if wait:
        with open(log_path, "w") if log_path else subprocess.DEVNULL as logf:  # type: ignore
            subprocess.run(cmd, stdout=logf, stderr=logf, check=True)
        return str(out_png), None
    else:
        # For async, don't keep a file handle open; discard output
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out_png), proc


def gerber_to_pdf_gerbv(file_path, save_folder, save_path, D=50):
    """

    :param file_path:
    :param save_path:
    :param D: 0-750
    :return:
    """
    Path(save_folder).mkdir(exist_ok=True, parents=True)
    command = fr'Assets\gerbv\gerbv -x pdf -D {D} -o "{save_path}.pdf" "{file_path}"'
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

def check_tiff_dimensions(folder_path, raw_tiff=False):
    """
    This function checks to make sure all tiff dimensions are the same

    params
    folder_path: The path where the tiff are located
    raw_tiff: If the files being loaded in are raw tiff, True. If the function is being used on gerber that has been coverted to tiff this is false
    """
    dimensions = None
    all_same_size = True
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):  # Ensure we're only checking TIFF files
            file_path = os.path.join(folder_path, file_name)
            with pImage.open(file_path) as img:
                if dimensions is None:
                    dimensions = img.size  # Set the initial dimensions to compare against
                    file_names.append(file_name)
                elif img.size != dimensions:
                    all_same_size = False
                    break  # Exit the loop early as we've found an inconsistency
                else:
                    file_names.append(file_name)

    if all_same_size:
        if raw_tiff:
            return True, file_names
        else:
            return True
    else:
        return False

