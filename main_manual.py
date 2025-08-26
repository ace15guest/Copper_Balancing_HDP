import re
from pathlib import Path
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from loading.gerber_conversions import gerber_to_png_gerbv
from loading.img2array import bitmap_to_array
from calculations.multi_layer import multiple_layers
from plotting.comparing import plot_points_3d
from calculations.layer_calcs import *
from calculations.transformation import align_dat_to_gerber

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
    orig_path = r"C:\Users\Asa Guest\Downloads\files\CuBalanceDatFiles\TopDatFiles\ACCL-890K-01-Q1_Top_Global.dat"
    data = np.loadtxt(orig_path, delimiter="\t")
    data = np.where(data == 9999.0, np.nan, data)

    # Get grid coordinates
    nrows, ncols = data.shape
    y_idx, x_idx = np.indices((nrows, ncols))

    # Flatten into [x, y, z] triplets
    xyz = np.column_stack([x_idx.ravel(), y_idx.ravel(), data.ravel()])
    xyz = xyz[~np.isnan(xyz[:, 2])]

    # Convert to DataFrame if convenient
    df = pd.DataFrame(xyz, columns=["x", "y", "z"])
    mapping = list_gerbers_with_weights(r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber\UL")
    temp_tiff_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\temp_tiff"
    folder = Path(temp_tiff_folder)
    arrays = {}

    for path, oz in mapping:
        print(f"{path} -> {oz} oz")
        name = path.split("\\")[-1]
        gerber_to_png_gerbv(gerb_file=path, save_folder=temp_tiff_folder,
                            save_path=rf"{temp_tiff_folder}\{name}.tiff", dpi=1000, scale=1)
    input("Press Enter once everything is complete")
    for file in folder.iterdir():
        name = str(file).split("\\")[-1]
        arrays[name] = bitmap_to_array(file)
    layers_preblend = multiple_layers(arrays)
    # layers = blur_tiff_gauss(layers_preblend, sigma=20)
    layers = met_ave(layers_preblend, radius=400)
    layers = (layers * 55.0 / np.max(layers))

    a, b = align_dat_to_gerber(layers, data)
    print(a, b)
    outfile, stats = plot_points_3d(layers, backend="plotly", outfile="test.html")
    print(outfile, stats)
