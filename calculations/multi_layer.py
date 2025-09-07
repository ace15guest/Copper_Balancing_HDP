import numpy as np

def multiple_layers(layer_dict: dict) -> np.ndarray:
    """
    Crop all layers to the smallest common shape and sum them together.

    @param layer_dict: Dictionary of 2D numpy arrays
    @return: Summed 2D numpy array
    """
    try:
        # Step 1: Find the smallest shape
        min_rows = min(arr.shape[0] for arr in layer_dict.values())
        min_cols = min(arr.shape[1] for arr in layer_dict.values())

        # Step 2: Crop each array to the smallest shape
        cropped_layers = [
            arr[:min_rows, :min_cols] for arr in layer_dict.values()
        ]

        # Step 3: Sum the cropped arrays
        layer_array = np.sum(cropped_layers, axis=0)

    except Exception as e:
        print(f"Error while summing layers: {e}")
        layer_array = None

    return layer_array

import re
import numpy as np
from typing import Dict

# Match: 1oz, 0.5oz, 0_5oz, 1.0oz, optionally with spaces, and require '_' or end after 'oz'
_OZ_RE = re.compile(r'(\d+(?:[._]\d+)?)\s*oz(?=$|_)', re.IGNORECASE)

def multiple_layers_weighted(layer_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Weighted sum of 2D arrays using copper weight parsed from key names.
    Keys must contain a token like '..._1oz_...', '..._0.5oz_...', '..._0_5oz_...'.

    Args:
        layer_dict: {name: 2D np.ndarray}

    Returns:
        2D np.ndarray: weighted sum cropped to the smallest (rows, cols).

    Raises:
        ValueError: if any key lacks a parsable '<number>oz' token or arrays aren't 2D.
    """
    if not layer_dict:
        raise ValueError("layer_dict is empty.")

    # Smallest common shape
    min_rows = min(arr.shape[0] for arr in layer_dict.values())
    min_cols = min(arr.shape[1] for arr in layer_dict.values())

    out = np.zeros((min_rows, min_cols), dtype=np.float32)
    missing = []

    for name, arr in layer_dict.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"Layer '{name}' must be a 2D numpy array.")
        m = _OZ_RE.search(name)
        if not m:
            missing.append(name)
            continue
        w = float(m.group(1).replace("_", "."))  # "0_5" -> 0.5
        out += w * arr[:min_rows, :min_cols].astype(np.float32, copy=False)

    if missing:
        raise ValueError("Could not parse copper weight from keys: " + ", ".join(missing))

    return out



