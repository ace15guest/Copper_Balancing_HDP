import os
import shutil
from pathlib import Path
import re
ini_global_path = f"{os.environ['LocalAppData']}/CuBalancing/Settings/config.ini"

def clear_folder(folder_path):
    """
    This function clears all files and subdirectories in a given folder.

    :param folder_path: The path to the folder to clear.
    """
    # Validate the input
    if not os.path.exists(folder_path):
        print(f"Error: The folder path {folder_path} does not exist.")
        os.makedirs(folder_path)
        return
    if not os.path.isdir(folder_path):
        print(f"Error: The path {folder_path} is not a directory.")
        return
    # Ensure we are only editing files within this directory -- not needed since we are creating a new folder in App data
    # if not os.path.abspath(folder_path).startswith(os.path.abspath('Assets')):
    #     print(f"Error: The path {folder_path} is not a safe path.")
    #     return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def create_folders(file_path:str):
    # Split the file path into a list of folders
    file_path = file_path.replace('\\','/').replace('//', '/')
    folders = file_path.split('/')

    # Create the folders if they do not exist
    file = Path(file_path)
    file.parent.mkdir(parents=True, exist_ok=True)

def find_inkscape():
    # Common directories where Inkscape might be installed
    windows_paths = [
        "C:/Program Files/Inkscape/bin/inkscape.exe",
        "C:/Program Files (x86)/Inkscape/inkscape.exe",
    ]
    macos_paths = [
        "/Applications/Inkscape.app/Contents/MacOS/inkscape",
    ]
    linux_paths = [
        "/usr/bin/inkscape",
        "/usr/local/bin/inkscape",
        "/snap/bin/inkscape",  # Snap package location on Ubuntu
    ]

    # Check paths based on the current operating system
    if os.name == 'nt':  # Windows
        for path in windows_paths:
            if os.path.exists(path):
                return path
    elif os.name == 'posix':  # macOS or Linux
        for path in macos_paths + linux_paths:
            if os.path.exists(path):
                return path

    return None  # Inkscape executable not found
def extract_format_spec(file_path):
    """Extract the %FS...*% line from the Gerber file."""
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('%FS') and line.endswith('*%\n'):
                return line.strip()
    # Default to 2.4 leading-zero omitted
    return '%FSLAX24Y24*%'
def create_outline_gerber_from_file(source_gerber, extrema, output_path):
    xmin = extrema["xval_dict"][extrema["xmin"]]
    xmax = extrema["xval_dict"][extrema["xmax"]]
    ymin = extrema["yval_dict"][extrema["ymin"]]
    ymax = extrema["yval_dict"][extrema["ymax"]]
    unit = extrema["unit"]
    format_line = extract_format_spec(source_gerber)
    # Determine scale from format (assumes 2.4 or 3.5, etc.)
    match = re.match(r'%FS[L|T]X(\d)(\d)Y(\d)(\d)\*%', format_line)
    if match:
        dec_digits = int(match.group(2))
    else:
        dec_digits = 4  # fallback default
    scale = 10 ** dec_digits
    def fmt(val): return f"{int(round(val * scale))}"
    if unit == "in":
        mo = "%MOIN*%"
        aperture = "%ADD10C,0.001*%"  # 1 mil
    else:
        mo = "%MOMM*%"
        aperture = "%ADD10C,0.1*%"    # 0.1 mm
    with open(output_path, "w") as f:
        f.write(format_line + "\n")
        f.write(mo + "\n")
        f.write("%IPPOS*%\n")
        f.write("%LPD*%\n")
        f.write("%AMOC8*5,1,8,0,0,1.08239X$1*%\n")
        f.write(aperture + "\n")
        f.write("D10*\n")
        f.write("G01*\n")
        # Draw rectangle
        f.write(f"X{xmin}Y{ymin}D02*\n")
        f.write(f"X{xmax}Y{ymin}D01*\n")
        f.write(f"X{xmax}Y{ymax}D01*\n")
        f.write(f"X{xmin}Y{ymax}D01*\n")
        f.write(f"X{xmin}Y{ymin}D01*\n")
        f.write("M02*\n")


