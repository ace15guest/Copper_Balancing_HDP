from pygerber.gerberx3.api.v2 import GerberFile
from wand.image import Image


def gerber_to_svg(file_path, save_path="output.svg", scale=20):
    GerberFile.from_file(file_path).parse().render_svg(save_path, scale=20)
    return

def svg_to_tiff(svg_path, tiff_path):
    with Image(filename=svg_path) as img:
        img.format = 'tiff'
        img.save(filename=tiff_path)

# file = r"C:\Users\Asa Guest\Downloads\TOP.274x"
# gerber_to_svg(file_path=file)

# # Example usage
# svg_file = 'output.svg'
# tiff_file = 'file.tiff'
# svg_to_tiff(svg_file, tiff_file)