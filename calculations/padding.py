import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Tuple

def fill_border(
    a: np.ndarray,
    *,
    method: str = "nearest",           # 'nearest' | 'idw' | 'biharmonic' | 'local_mean' | 'mean_percent' | 'max_percent'
    idw_k: int = 8,                    # for method='idw'
    idw_power: float = 2.0,            # for method='idw'
    local_mean_radius: Optional[int] = None,  # for method='local_mean' (auto if None)
    # for method='mean_percent' or 'max_percent'
    mean_percent: float = 1.0,         # multiplier for global mean
    max_percent: float = 1.0,          # multiplier for global max
    mean_ignore_zeros: bool = True,    # ignore zeros when computing stats
    return_mask: bool = False,         # also return the boolean mask of filled pixels
    out_dtype: Optional[np.dtype] = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Auto-fill border-connected zeros/NaNs ("padding") so smoothing/blending won't taper.

    - Detects all pixels that are (zero or NaN) AND connected to ANY image edge (8-connectivity).
    - Fills only those pixels; interior zeros remain untouched.

    Methods:
      - 'nearest'      : copy nearest valid pixel (fastest, crisp seams)
      - 'idw'          : inverse-distance weighting from nearest valid pixels (smooth seams)
      - 'biharmonic'   : PDE inpainting (very smooth; requires scikit-image)
      - 'local_mean'   : normalized box mean (if radius=None, auto-estimates from mask thickness)
      - 'mean_percent' : fill border with (mean_percent * global_mean_of_valid_pixels)
      - 'max_percent'  : fill border with (max_percent  * global_max_of_valid_pixels)
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("fill_border expects a 2D array.")

    # -------- helpers --------
    def _border_missing_mask(arr: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(arr) | (arr == 0)
        structure = np.ones((3, 3), bool)  # 8-connectivity
        labels, _ = ndi.label(missing, structure=structure)
        if labels.max() == 0:
            return np.zeros_like(missing, dtype=bool)
        edge_labels = set(np.unique(labels[0, :])) \
                    | set(np.unique(labels[-1, :])) \
                    | set(np.unique(labels[:, 0])) \
                    | set(np.unique(labels[:, -1]))
        edge_labels.discard(0)
        if not edge_labels:
            return np.zeros_like(missing, dtype=bool)
        return np.isin(labels, list(edge_labels))

    def _fill_constant(arr: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
        out = arr.astype(float, copy=True)
        out[mask] = value
        return out

    # --- existing helpers omitted here for brevity (nearest, idw, local_mean, biharmonic) ---

    # -------- main --------
    mask = _border_missing_mask(a)
    if not mask.any():
        return (a.copy(), mask) if return_mask else a.copy()

    method_lc = method.lower()
    if method_lc == "nearest":
        filled = _fill_nearest(a, mask)
    elif method_lc == "idw":
        filled = _fill_idw(a, mask, k=idw_k, power=idw_power)
    elif method_lc == "biharmonic":
        filled = _fill_biharmonic(a, mask)
    elif method_lc == "local_mean":
        filled = _fill_local_mean(a, mask, local_mean_radius)
    elif method_lc == "mean_percent":
        arrf = a.astype(float, copy=False)
        valid = np.isfinite(arrf)
        if mean_ignore_zeros:
            valid &= (arrf != 0)
        if not valid.any():
            filled = a.astype(float, copy=True)
        else:
            gmean = float(arrf[valid].mean())
            filled = _fill_constant(a, mask, mean_percent * gmean)
    elif method_lc == "max_percent":
        arrf = a.astype(float, copy=False)
        valid = np.isfinite(arrf)
        if mean_ignore_zeros:
            valid &= (arrf != 0)
        if not valid.any():
            filled = a.astype(float, copy=True)
        else:
            gmax = float(arrf[valid].max())
            filled = _fill_constant(a, mask, max_percent * gmax)
    else:
        raise ValueError("method must be one of: 'nearest', 'idw', 'biharmonic', 'local_mean', 'mean_percent', 'max_percent'")

    # Cast output dtype
    if out_dtype is not None:
        filled = filled.astype(out_dtype, copy=False)
    elif np.issubdtype(a.dtype, np.floating):
        filled = filled.astype(a.dtype, copy=False)
    else:
        filled = filled.astype(np.float32, copy=False)

    return (filled, mask) if return_mask else filled

