import numpy as np
from scipy import ndimage as ndi
from typing import Optional, Tuple

def fill_border(
    a: np.ndarray,
    *,
    method: str = "nearest",           # 'nearest' | 'idw' | 'biharmonic' | 'local_mean'
    idw_k: int = 8,                    # for method='idw'
    idw_power: float = 2.0,            # for method='idw'
    local_mean_radius: Optional[int] = None,  # for method='local_mean' (auto if None)
    return_mask: bool = False,         # also return the boolean mask of filled pixels
    out_dtype: Optional[np.dtype] = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Auto-fill border-connected zeros/NaNs ("padding") so smoothing/blending won't taper.

    - Detects all pixels that are (zero or NaN) AND connected to ANY image edge (8-connectivity).
    - Fills only those pixels; interior zeros remain untouched.

    Methods:
      - 'nearest'    : copy nearest valid pixel (fastest, crisp seams)
      - 'idw'        : inverse-distance weighting from nearest valid pixels (smooth seams)
      - 'biharmonic' : PDE inpainting (very smooth; requires scikit-image)
      - 'local_mean' : normalized box mean (if radius=None, auto-estimates from mask thickness)

    Args:
        a                 : 2D array (float recommended if NaNs present)
        method            : fill strategy (see above)
        idw_k             : number of neighbors for IDW (1..N)
        idw_power         : distance exponent for IDW (>=1)
        local_mean_radius : half window for local mean; if None, chooses automatically
        return_mask       : if True, also returns the filled-pixel boolean mask
        out_dtype         : cast result to this dtype; default preserves float dtype else float32

    Returns:
        filled array (and optionally the boolean mask of filled pixels)
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

    def _fill_nearest(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = arr.astype(float, copy=True)
        valid = np.isfinite(out) & (out != 0)
        if not valid.any():
            return out
        ri, rj = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out[mask] = out[ri[mask], rj[mask]]
        return out

    def _fill_idw(arr: np.ndarray, mask: np.ndarray, k: int, power: float) -> np.ndarray:
        from scipy.spatial import cKDTree
        out = arr.astype(float, copy=True)
        valid = np.isfinite(out) & (out != 0) & (~mask)
        if not valid.any():
            return _fill_nearest(out, mask)
        pts_valid = np.column_stack(np.nonzero(valid))
        vals_valid = out[valid]
        pts_query = np.column_stack(np.nonzero(mask))
        k = max(1, min(k, len(pts_valid)))
        tree = cKDTree(pts_valid)
        dist, idx = tree.query(pts_query, k=k, workers=-1)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        w = 1.0 / np.maximum(dist, 1e-6) ** power
        w /= w.sum(axis=1, keepdims=True)
        out[pts_query[:, 0], pts_query[:, 1]] = (w * vals_valid[idx]).sum(axis=1)
        return out

    def _local_mean(arr: np.ndarray, radius: int) -> np.ndarray:
        from scipy.ndimage import uniform_filter
        arr = arr.astype(float, copy=False)
        valid = np.isfinite(arr) & (arr != 0)
        arr0 = np.where(valid, arr, 0.0)
        size = 2 * radius + 1
        num = uniform_filter(arr0, size=size, mode='reflect')
        den = uniform_filter(valid.astype(float), size=size, mode='reflect')
        out = np.empty_like(num)
        np.divide(num, np.maximum(den, 1e-12), out=out)
        out[den <= 1e-12] = np.nan
        return out

    def _fill_local_mean(arr: np.ndarray, mask: np.ndarray, radius: Optional[int]) -> np.ndarray:
        out = arr.astype(float, copy=True)
        if radius is None:
            # Auto radius: ~80th percentile of distance-to-valid within the missing border,
            # clamped to [3, 25]. Provides a reasonable local context.
            valid = np.isfinite(out) & (out != 0)
            # distance to nearest VALID pixel (inside mask region)
            dist = ndi.distance_transform_edt(~valid)
            d_mask = dist[mask]
            if d_mask.size:
                est = int(np.clip(np.percentile(d_mask, 80), 3, 25))
            else:
                est = 7
            radius = est
        lm = _local_mean(out, radius)
        # where local mean is NaN (no support), fall back to nearest
        fallback = np.isnan(lm) & mask
        out[mask] = lm[mask]
        if fallback.any():
            out = _fill_nearest(out, fallback)
        return out

    def _fill_biharmonic(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            from skimage.restoration import inpaint_biharmonic
        except Exception as e:
            raise ImportError("Method 'biharmonic' requires scikit-image: pip install scikit-image") from e
        a0 = arr.astype(float, copy=True)
        a0 = np.nan_to_num(a0, nan=0.0)
        return inpaint_biharmonic(a0, mask, channel_axis=None)

    # -------- main --------
    mask = _border_missing_mask(a)
    if not mask.any():
        return (a.copy(), mask) if return_mask else a.copy()

    method = method.lower()
    if method == "nearest":
        filled = _fill_nearest(a, mask)
    elif method == "idw":
        filled = _fill_idw(a, mask, k=idw_k, power=idw_power)
    elif method == "biharmonic":
        filled = _fill_biharmonic(a, mask)
    elif method == "local_mean":
        filled = _fill_local_mean(a, mask, local_mean_radius)
    else:
        raise ValueError("method must be one of: 'nearest', 'idw', 'biharmonic', 'local_mean'")

    # Cast output dtype
    if out_dtype is not None:
        filled = filled.astype(out_dtype, copy=False)
    elif np.issubdtype(a.dtype, np.floating):
        filled = filled.astype(a.dtype, copy=False)
    else:
        filled = filled.astype(np.float32, copy=False)

    return (filled, mask) if return_mask else filled
