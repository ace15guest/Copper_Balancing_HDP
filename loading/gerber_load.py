import os

from wand.image import Image
import subprocess
import time
from pathlib import Path





def check_gerber(file_path: str) -> str:
    """
    This function verifies that a gerber file is valid by running it through the gerbv application.
    :param file_path: The path to the gerber file
    :return: True if the gerber file is valid, False otherwise

    """
    name = file_path.split('\\')[-1]
    Path(r"Assets\temp").mkdir(exist_ok=True, parents=True)
    log_file_name = fr"Assets\temp\{name}.log"
    cmd_line = fr'Assets\gerbv\gerbv -x rs274x -o NUL "{file_path}" 2>"{log_file_name}"'

    start_time = time.time()
    subprocess.Popen(cmd_line, shell=True)
    end_time = time.time()
    # TODO: Make Gerbv log
    # print(f"Time taken to run gerbv: {end_time - start_time}")


    return log_file_name

def verify_gerber(file_path: str) -> bool:
    failed_words = ['error', 'fail', 'invalid', 'not found', 'not supported', 'not recognized',
                    'not valid', 'not read', 'not exist']
    with open(file_path, 'r') as file:
        file_contents = file.read().lower()
    error = any(word in file_contents for word in failed_words)
    os.remove(file_path)
    if error:
        return False
    return True


# file = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\loading\img2array.py"
# verify_gerber(file)

# # Example usage
# svg_file = 'output.svg'
# tiff_file = 'file.tiff'
# svg_to_tiff(svg_file, tiff_file)
