import os
import subprocess
import time
from pathlib import Path





def check_gerber(file_path: str) -> str:
    """
    This function verifies that a gerber file is valid by running it through the gerbv application.
    :param file_path: The path to the gerber file
    :return: True if the gerber file is valid, False otherwise

    """
    name = Path(file_path).name
    Path(r"Assets\temp").mkdir(exist_ok=True, parents=True)
    log_file_name = fr"Assets\temp\{name}.log"
    cmd_line = fr'Assets\gerbv\gerbv -x rs274x -o NUL "{file_path}" 2>"{log_file_name}"'
    process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return log_file_name, process

def check_tiff(file_path:str) -> str:
    return

def verify_gerber(file_path: str) -> bool:
    """
    Verify the Gerber file generated from Gerbv.

    Args:
        file_path (str): The path to the Gerber file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    failed_words = ['error', 'fail', 'invalid', 'not found', 'not supported', 'not recognized',
                    'not valid', 'not read', 'not exist']
    with open(file_path, 'r') as file:
        file_contents = file.read().lower()
    error = any(word in file_contents for word in failed_words) # Check if any of the error words are in the File generated from gerbv
    os.remove(file_path) # Remove the file from the computer
    if error:
        return False
    return True


# file = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\loading\img2array.py"
# verify_gerber(file)

# # Example usage
# svg_file = 'output.svg'
# tiff_file = 'file.tiff'
# svg_to_tiff(svg_file, tiff_file)
