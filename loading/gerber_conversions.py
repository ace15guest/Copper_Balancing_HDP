import os
import time

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
def gerber_to_pdf_gerbv(file_path, save_path, D=50):
    """

    :param file_path:
    :param save_path:
    :param D: 0-750
    :return:
    """
    command = f'Assets\gerbv\gerbv -x pdf -D {D} -o "{save_path}.pdf" "{file_path}"'
    subprocess.Popen(command)
    return

gerber_odb_in = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Example1\l6_plane_1oz.gdo"
gerber_274x = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Example1\l6_plane_1oz.274x"
svg_path = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Example1\l6_plane_1oz.svg"
tiff_path = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Example1\l6_plane_1oz.tiff"
gerber_to_274x(gerber_odb_in, gerber_274x)
gerber_to_svg_gerbv(gerber_274x, save_path="output.svg")
svg_to_tiff(svg_path, tiff_path, width=100, height=100)