from pygerber.gerberx3.api.v2 import GerberFile
from wand.image import Image
import subprocess


def gerber_to_svg(file_path, save_path="output.svg", scale=20):
    GerberFile.from_file(file_path).parse().render_svg(save_path, scale=20)
    return


def svg_to_tiff(svg_path, tiff_path):
    with Image(filename=svg_path) as img:
        img.format = 'tiff'
        img.save(filename=tiff_path)


def verify_gerber(file_path: str) -> bool:
    """
    This function verifies that a gerber file is valid by running it through the gerbv application.
    :param file_path: The path to the gerber file
    :return: True if the gerber file is valid, False otherwise
    """
    cmd_line = f"Assets\gerbv\gerbv -x rs274x -o NUL \"{file_path}\""
    a = subprocess.run(cmd_line, shell=True, capture_output=True)

    failed_words = ['critical', 'error', 'warning', 'fail', 'invalid', 'not found', 'not supported', 'not recognized',
                    'not valid', 'not read', 'not exist']
    sub_out = str(a.stderr).lower()
    error = any(word in sub_out for word in failed_words)
    if error:
        return False
    return True


# file = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\loading\img2array.py"
# verify_gerber(file)

# # Example usage
# svg_file = 'output.svg'
# tiff_file = 'file.tiff'
# svg_to_tiff(svg_file, tiff_file)
