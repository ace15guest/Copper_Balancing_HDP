from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Tuple


from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi

def fill_border(
    a: npt.NDArray,
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
        """Missing == NaN/Inf or exactly 0. Only those connected to an edge are kept."""
        missing = ~np.isfinite(arr) | (arr == 0)
        if not missing.any():
            return np.zeros_like(missing, dtype=bool)
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

    def _fill_nearest(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Copy value from nearest valid pixel using EDT indices."""
        valid = np.isfinite(arr) & (arr != 0)
        if not valid.any():
            # Nothing valid to copy from; just return float array
            return arr.astype(float, copy=True)
        # distance_transform_edt on 'valid==0' to get nearest valid indices
        _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
        out = arr.astype(float, copy=True)
        out[mask] = arr[iy[mask], ix[mask]]
        return out

    def _fill_idw(arr: np.ndarray, mask: np.ndarray, *, k: int, power: float) -> np.ndarray:
        """Inverse-distance weighting from k nearest valid pixels."""
        from scipy.spatial import cKDTree
        valid = np.isfinite(arr) & (arr != 0)
        if not valid.any():
            return arr.astype(float, copy=True)
        yy, xx = np.nonzero(valid)
        vals = arr[valid].astype(float)
        tree = cKDTree(np.c_[yy, xx])

        my, mx = np.nonzero(mask)
        if my.size == 0:
            return arr.astype(float, copy=True)

        # Query up to k neighbors (cap at available points)
        k_eff = min(k, yy.size)
        dists, idxs = tree.query(np.c_[my, mx], k=k_eff)
        # Ensure 2D
        if k_eff == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Weights: 1 / d^p  (handle d=0 -> take that exact value)
        out_vals = np.empty(my.size, dtype=float)
        zero_mask = (dists == 0)
        if zero_mask.any(axis=1).any():
            # Where a zero-distance neighbor exists: just copy that value
            # (Pick the first zero-distance)
            for i in range(my.size):
                z = np.where(zero_mask[i])[0]
                if z.size:
                    out_vals[i] = vals[idxs[i, z[0]]]
                else:
                    # No exact; compute weighted average
                    w = 1.0 / np.power(dists[i], power)
                    out_vals[i] = np.sum(w * vals[idxs[i]]) / np.sum(w)
        else:
            w = 1.0 / np.power(dists, power)
            out_vals = np.sum(w * vals[idxs], axis=1) / np.sum(w, axis=1)

        out = arr.astype(float, copy=True)
        out[my, mx] = out_vals
        return out

    def _fill_local_mean(arr: np.ndarray, mask: np.ndarray, radius: Optional[int]) -> np.ndarray:
        """Box mean of nearby valid pixels (ignores zeros/NaNs). Auto radius from mask thickness if None."""
        arrf = arr.astype(float, copy=False)
        valid = np.isfinite(arrf) & (arrf != 0)

        if radius is None:
            # Auto radius based on the typical thickness of the masked band
            # Use median EDT inside the masked region (distance to nearest non-mask).
            if mask.any():
                thick = ndi.distance_transform_edt(mask)
                med = int(np.clip(np.median(thick[mask]), 1, max(1, min(arr.shape)//8)))
                radius = med
            else:
                radius = max(1, min(arr.shape)//100)

        ksize = 2 * int(radius) + 1

        # Convolve sums and counts separately, then divide
        # Use uniform_filter which is separable and fast
        sum_vals = ndi.uniform_filter(np.where(valid, arrf, 0.0), size=ksize, mode="nearest") * (ksize * ksize)
        cnt_vals = ndi.uniform_filter(valid.astype(float), size=ksize, mode="nearest") * (ksize * ksize)

        # Avoid division by zero; where count==0, fall back to nearest
        local_mean = np.where(cnt_vals > 0, sum_vals / np.maximum(cnt_vals, 1e-12), np.nan)

        out = arrf.copy()
        # Fill where mask & we have a mean; if any NaNs remain, backfill with nearest
        need = mask
        out[need] = local_mean[need]

        # Backfill remaining nans (e.g., completely empty neighborhood)
        bad = np.isnan(out)
        if bad.any():
            out = _fill_nearest(out, bad)

        return out

    def _fill_biharmonic(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use scikit-image biharmonic inpainting (very smooth)."""
        try:
            from skimage.restoration import inpaint_biharmonic
        except Exception as e:
            raise RuntimeError(
                "Biharmonic method requires scikit-image. "
                "Install it or choose another method."
            ) from e

        arrf = arr.astype(float, copy=False)
        result = inpaint_biharmonic(arrf, mask, channel_axis=None)
        return result

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


