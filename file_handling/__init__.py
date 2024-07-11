import os
import shutil

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
    if not os.path.abspath(folder_path).startswith(os.path.abspath('Assets')):
        print(f"Error: The path {folder_path} is not a safe path.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


import os


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



