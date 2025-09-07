import os
import shutil
from pathlib import Path
import re
from typing import List, Tuple, Union
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


import os


def get_global_files(folder_path):
    """
    Return a list of files in the given folder that contain 'global' or 'Global'.

    Args:
        folder_path (str): Path to the folder

    Returns:
        list: List of matching filenames (full paths)
    """
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
           and ("global" in f.lower())  # makes it case-insensitive
    ]


def list_gerbers_with_weights(
    root: Union[str, Path],
    patterns: tuple[str, ...] = ("*.gbr", "*.GBR"),
) -> List[Tuple[str, float]]:
    """
    Scan 'root' recursively for Gerber files and infer copper weight from the filename.
    Works with names such as 'l4_plane_1oz_Q1.gbr', 'l6_plane_0_5oz_LL.gbr', 'sig_0.5oz.gbr', 'pwr_1.0oz.gbr'.

    Accepted weight formats in the name: '1oz', '0.5oz', '0_5oz', '1.0oz' (case-insensitive).
    Returns:
        List of (absolute_path, copper_oz)
    """
    # Match a number (optionally with . or _) immediately followed by 'oz' anywhere in the stem
    OZ_RE = re.compile(r'(\d+(?:[._]\d+)?)oz', re.IGNORECASE)

    root = Path(root).resolve()
    results: List[Tuple[str, float]] = []

    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))

    for p in sorted(set(files)):
        m = OZ_RE.search(p.stem)  # look only at stem (without extension)
        if not m:
            continue
        raw = m.group(1)
        # allow underscores as decimal separators (e.g., 0_5 -> 0.5)
        val = float(raw.replace("_", "."))
        results.append((str(p), val))

    return results

import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

def wait_for_folder_complete(
    folder: Union[str, Path],
    *,
    expected_names: Optional[Iterable[str]] = None,  # exact basenames (case-insensitive on Windows)
    expected_count: Optional[int] = None,            # number of files expected (after filtering)
    patterns: Tuple[str, ...] = ("*",),              # glob(s) to include, e.g. ("*.dat", "*.gbr")
    recursive: bool = False,                         # search subfolders too
    ignore_exts: Tuple[str, ...] = (".tmp_id", ".part", ".crdownload", ".download", ".~", ".swp"),
    quiet_period: float = 5.0,                       # seconds with no size/mtime changes
    poll_interval: float = 1.0,                      # seconds between checks
    timeout: Optional[float] = None,                 # None = wait forever
    verbose: bool = True                             # print progress
) -> List[str]:
    """
    Block until the folder appears "complete" per criteria, then return the file paths.

    Completion criteria (all must be satisfied):
      1) If expected_names is given: all those basenames exist.
      2) If expected_count is given: number of matched files >= expected_count.
      3) No *ignored* temp files are present.
      4) Folder is stable for `quiet_period` (no size/mtime changes to any matched file).

    Notes:
    - Uses only standard library; safe for network shares.
    - If neither expected_names nor expected_count is given, completion is just (3) and (4).
    - Returns the final list of matched files (absolute paths) when complete.
    - Raises TimeoutError if `timeout` elapses.
    """
    start = time.monotonic()
    folder = Path(folder)

    def _list_files() -> List[Path]:
        files: List[Path] = []
        pats = patterns if isinstance(patterns, (tuple, list)) else (patterns,)
        if recursive:
            for pat in pats:
                files.extend(folder.rglob(pat))
        else:
            for pat in pats:
                files.extend(folder.glob(pat))
        # files only
        files = [p for p in files if p.is_file()]
        # drop ignored temp extensions
        lowered_ign = tuple(e.lower() for e in ignore_exts)
        files = [p for p in files if not p.suffix.lower().endswith(lowered_ign)]
        return files

    def _has_ignored_temp() -> bool:
        pats = patterns if isinstance(patterns, (tuple, list)) else (patterns,)
        if recursive:
            it = (folder.rglob(p) for p in pats)
        else:
            it = (folder.glob(p) for p in pats)
        lowered_ign = tuple(e.lower() for e in ignore_exts)
        for gen in it:
            for p in gen:
                if p.is_file() and p.suffix.lower().endswith(lowered_ign):
                    return True
        return False

    def _snapshot(files: List[Path]):
        # map path -> (size, mtime)
        snap = {}
        for p in files:
            try:
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime)
            except FileNotFoundError:
                # file vanished between list and stat
                pass
        return snap

    # normalize expected names for case-insensitive filesystems
    expected_set = None
    if expected_names:
        expected_set = {str(n).strip() for n in expected_names}
        # If caller passed full paths, collapse to basenames
        expected_set = {Path(n).name for n in expected_set}

    last_change_t = time.monotonic()
    prev_snap = {}

    while True:
        if timeout is not None and (time.monotonic() - start) > timeout:
            raise TimeoutError("Timed out waiting for folder to complete.")

        files = _list_files()
        basenames = {p.name for p in files}

        # Criteria 1: all expected names exist
        if expected_set is not None:
            missing = expected_set - basenames
            have_expected = len(missing) == 0
        else:
            have_expected = True
            missing = set()

        # Criteria 2: count threshold
        if expected_count is not None:
            have_count = len(files) >= expected_count
        else:
            have_count = True

        # Criterion 3: no temp/partial files present
        no_temps = not _has_ignored_temp()

        # Criterion 4: stability (no size/mtime changes for quiet_period)
        snap = _snapshot(files)
        if snap != prev_snap:
            last_change_t = time.monotonic()
            prev_snap = snap

        stable = (time.monotonic() - last_change_t) >= quiet_period

        if verbose:
            msg = [
                f"files={len(files)}",
                f"expected_ok={have_expected}" + (f" (missing: {sorted(missing)})" if not have_expected else ""),
                f"count_ok={have_count}",
                f"no_temps={no_temps}",
                f"stable={stable}",
            ]
            print("[wait_for_folder_complete] " + " | ".join(msg))

        if have_expected and have_count and no_temps and stable:
            # Done — return absolute paths as strings
            return [str(p.resolve()) for p in files]

        time.sleep(poll_interval)

