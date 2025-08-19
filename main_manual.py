import re
from pathlib import Path
import re
from typing import List, Tuple
from loading.gerber_conversions import gerber_to_png_gerbv
from loading.img2array import bitmap_to_array
# Matches "..._1oz.gbr", "..._0.5oz.gbr", "..._0_5oz.gbr" (case-insensitive)
OZ_RE = re.compile(r'(\d+(?:[._]\d+)?)\s*oz\b', re.IGNORECASE)

def list_gerbers_with_weights(root: str | Path, patterns: tuple[str, ...] = ("*.gbr", "*.GBR")) -> List[Tuple[str, float]]:
    """
    Scan 'root' for Gerber files and infer copper weight from filename suffix like '_1oz'.
    Returns a list of (absolute_path, copper_oz).

    Accepted weight formats in name: '1oz', '0.5oz', '0_5oz', '1.0oz'
    """
    root = Path(root).resolve()
    results: List[Tuple[str, float]] = []

    files = []
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



if __name__ == "__main__":
    # Example: scan current dir
    mapping = list_gerbers_with_weights(r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber")
    temp_tiff_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\temp_tiff"
    folder = Path(temp_tiff_folder)
    arrays = {}


    for path, oz in mapping:
        print(f"{path} -> {oz} oz")
        name = path.split("\\")[-1]
        gerber_to_png_gerbv(gerb_file=path, save_folder=temp_tiff_folder,
                            save_path=rf"{temp_tiff_folder}\{name}.tiff", dpi=250, scale=1)
    input("Press Enter once everything is complete")
    for file in folder.iterdir():
        name = str(file).split("\\")[-1]
        arrays[name] = bitmap_to_array(file)
    print()

