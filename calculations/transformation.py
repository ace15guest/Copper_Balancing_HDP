import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate, rescale
from skimage.registration import phase_cross_correlation
import numpy as np
from skimage.transform import SimilarityTransform, warp, rotate, rescale

# ----------------- Helpers -----------------

def sanitize(a):
    """Ensure finite float32 array [0..1]."""
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    amin, amax = a.min(), a.max()
    if np.isfinite(amax) and amax > 0:
        a = a / amax
    return a


def binarize_robust(a, sigma=1.0):
    """Blur + Otsu threshold, safe on flat images."""
    a = sanitize(a)
    a = gaussian_filter(a, sigma=sigma)
    if not np.isfinite(a).any() or np.all(a == a.flat[0]):
        return (a > 0).astype(np.float32)
    try:
        t = threshold_otsu(a)
    except Exception:
        t = np.nanmedian(a) if np.isfinite(a).any() else 0.0
    return (a >= t).astype(np.float32)


def dominant_angle_safe(binary):
    """Estimate orientation from image moments; safe on sparse data."""
    y, x = np.nonzero(binary)
    if len(x) < 100:
        return 0.0
    x = x - x.mean();
    y = y - y.mean()
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def auto_scale(ref, mov, scales, preblur=1.0):
    """Pick scale that maximizes NCC vs ref."""
    ref_b = binarize_robust(ref, sigma=preblur)
    best = (1.0, -np.inf)
    for s in scales:
        m = rescale(mov, s, anti_aliasing=True, preserve_range=True)
        m_b = binarize_robust(m, sigma=preblur)
        H = min(ref_b.shape[0], m_b.shape[0])
        W = min(ref_b.shape[1], m_b.shape[1])
        if H < 32 or W < 32:
            continue
        rb = ref_b[:H, :W] - ref_b[:H, :W].mean()
        mb = m_b[:H, :W] - m_b[:H, :W].mean()
        denom = (np.linalg.norm(rb) * np.linalg.norm(mb) + 1e-8)
        ncc = float((rb * mb).sum() / denom)
        if ncc > best[1]:
            best = (s, ncc)
    return best[0]


def phase_translate(ref, mov):
    """Return (dy, dx) shift to align mov onto ref."""
    if ref.size == 0 or mov.size == 0:
        return 0.0, 0.0
    try:
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
        return float(shift[0]), float(shift[1])
    except Exception:
        return 0.0, 0.0


# ----------------- Main aligner -----------------

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate, rescale, SimilarityTransform, warp
from skimage.registration import phase_cross_correlation

def align_dat_to_gerber(
    gerber_arr,
    dat_arr,
    px_per_mm_gerber=None,
    px_per_mm_dat=None,
    scale_search=(0.7, 1.4, 21),
    flip_x=False,
    flip_y=False,
    # NEW: hole handling
    fill_mode="edge",        # 'edge' | 'reflect' | 'constant'
    fill_cval=np.nan,        # used only if fill_mode == 'constant'
    return_valid_mask=True,
):
    """
    Align DAT -> Gerber by rotation+scale+translation and resample directly
    onto the Gerber canvas WITHOUT zero padding.

    fill_mode:
      - 'edge'     : extend nearest edge values (no zeros, no NaNs)
      - 'reflect'  : reflect about the border
      - 'constant' : fill with fill_cval (e.g., np.nan to mask later)

    If return_valid_mask=True, also returns a boolean mask of pixels that
    come from valid DAT samples (True) vs extrapolated/fill (False).
    """
    # ---- helpers ----
    def sanitize(a):
        a = np.asarray(a, dtype=np.float32)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    def binarize_robust(a, sigma=1.0):
        a = sanitize(a)
        a = gaussian_filter(a, sigma=sigma)
        if not np.isfinite(a).any() or np.all(a == a.flat[0]):
            return (a > 0).astype(np.float32)
        try:
            t = threshold_otsu(a)
        except Exception:
            t = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
        return (a >= t).astype(np.float32)

    def dominant_angle_safe(binary):
        y, x = np.nonzero(binary)
        if len(x) < 100:
            return 0.0
        x = x - x.mean(); y = y - y.mean()
        cov = np.cov(np.vstack([x, y]))
        if not np.all(np.isfinite(cov)):
            return 0.0
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def auto_scale(ref, mov, scales, preblur=1.0):
        ref_b = binarize_robust(ref, sigma=preblur)
        best = (1.0, -np.inf)
        for s in scales:
            m = rescale(mov, s, anti_aliasing=True, preserve_range=True)
            m_b = binarize_robust(m, sigma=preblur)
            H = min(ref_b.shape[0], m_b.shape[0])
            W = min(ref_b.shape[1], m_b.shape[1])
            if H < 32 or W < 32:
                continue
            rb = ref_b[:H, :W]; mb = m_b[:H, :W]
            rb0 = rb - rb.mean(); mb0 = mb - mb.mean()
            denom = (np.linalg.norm(rb0) * np.linalg.norm(mb0) + 1e-8)
            ncc = float((rb0 * mb0).sum() / denom)
            if ncc > best[1]:
                best = (s, ncc)
        return best[0]

    def phase_translate(ref, mov):
        try:
            shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
            return float(shift[0]), float(shift[1])
        except Exception:
            return 0.0, 0.0

    # ---- inputs ----
    G = sanitize(gerber_arr)
    D = sanitize(dat_arr)
    if flip_x: D = np.fliplr(D)
    if flip_y: D = np.flipud(D)
    if not np.any(G): raise ValueError("Gerber array has no finite data.")
    if not np.any(D): raise ValueError("DAT array has no finite data.")

    # ---- orientation + scale ----
    angG = dominant_angle_safe(binarize_robust(G))
    angD = dominant_angle_safe(binarize_robust(D))
    rot_needed = angG - angD

    # IMPORTANT: rotate with mode + cval so we don't introduce zeros
    Drot = rotate(
        D, angle=rot_needed, resize=True, preserve_range=True,
        order=1, mode=fill_mode, cval=fill_cval
    )

    if px_per_mm_gerber and px_per_mm_dat:
        scale = float(px_per_mm_gerber) / float(px_per_mm_dat)
    else:
        smin, smax, n = scale_search
        scales = np.linspace(smin, smax, n)
        scale = auto_scale(G, Drot, scales)

    Drs = rescale(Drot, scale, anti_aliasing=True, preserve_range=True, order=1)

    # ---- translation on overlap ----
    H = min(G.shape[0], Drs.shape[0])
    W = min(G.shape[1], Drs.shape[1])
    Gc = gaussian_filter(G[:H, :W], 1.0)
    Dc = gaussian_filter(Drs[:H, :W], 1.0)
    dy, dx = phase_translate(Gc, Dc)

    # ---- compose translation as a warp (no manual paste) ----
    # Use warp(..., mode=fill_mode, cval=fill_cval) so we never insert zeros.
    tform = SimilarityTransform(scale=1.0, rotation=0.0, translation=(dx, dy))
    aligned = warp(
        Drs, inverse_map=tform.inverse, output_shape=G.shape,
        order=1, mode=fill_mode, cval=fill_cval, preserve_range=True
    ).astype(np.float32)

    # ---- optional validity mask (where pixels truly came from DAT) ----
    valid_mask = None
    if return_valid_mask:
        ones = np.ones_like(D, dtype=np.float32)
        ones_rot = rotate(
            ones, angle=rot_needed, resize=True, preserve_range=True,
            order=0, mode='constant', cval=0.0
        )
        ones_rs = rescale(ones_rot, scale, anti_aliasing=False, preserve_range=True, order=0)
        mask = warp(
            ones_rs, inverse_map=tform.inverse, output_shape=G.shape,
            order=0, mode='constant', cval=0.0, preserve_range=True
        )
        valid_mask = (mask > 0.5)

    params = {
        "rotation_deg": float(rot_needed),
        "scale": float(scale),
        "shift": (float(dy), float(dx)),
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
    }

    if return_valid_mask:
        return aligned, params, valid_mask
    else:
        return aligned, params

def apply_alignment(mov_arr, params, out_shape, order=1, cval=0.0):
    """
    Apply a previously computed Gerber↔DAT alignment to another array.

    Parameters
    ----------
    mov_arr : 2D numpy array
        The "moving" dataset you want to transform (e.g. another DAT layer).
        This should be in the *same native coordinate system* as the one you
        originally aligned with `align_dat_to_gerber`.

    params : dict
        The dictionary returned by `align_dat_to_gerber`. Expected keys are:
          - 'rotation_deg' : float
              Rotation angle in degrees applied to DAT → Gerber.
          - 'scale' : float
              Scale factor applied to DAT → Gerber.
          - 'shift' : (dy, dx)
              Translation (row shift, col shift) needed to align to Gerber.
              Positive dy = down, positive dx = right.
          - 'flip_x' : bool (optional)
              Whether a left-right mirror was applied during alignment.
          - 'flip_y' : bool (optional)
              Whether an up-down mirror was applied during alignment.

    out_shape : tuple of ints
        Desired output shape (rows, cols). Usually you pass
        `gerber_arr.shape` so the aligned result is on the Gerber canvas.

    order : int, optional (default=1)
        Interpolation order used by `skimage.transform.warp`.
        0=nearest, 1=bilinear, 3=cubic. Higher = smoother but slower.

    cval : float, optional (default=0.0)
        Fill value used for areas that move outside the canvas after transform.

    Returns
    -------
    aligned : 2D numpy array, dtype=float32
        The input `mov_arr`, transformed by the rotation/scale/translation
        stored in `params` and resampled onto `out_shape`.

    Notes
    -----
    - Use this to apply *the exact same alignment* you computed once with
      `align_dat_to_gerber` to other arrays from the same DAT dataset.
    - That way all your DAT layers line up with the Gerber reference.
    """

    a = np.asarray(mov_arr, dtype=np.float32)
    p = {**{"rotation_deg":0.0,"scale":1.0,"shift":(0.0,0.0),
            "flip_x":False,"flip_y":False}, **params}

    # Apply flips if required
    if p["flip_x"]:
        a = np.fliplr(a)
    if p["flip_y"]:
        a = np.flipud(a)

    # Apply rotation (with resize so nothing gets cropped)
    if abs(p["rotation_deg"]) > 1e-6:
        a = rotate(a, angle=p["rotation_deg"], resize=True,
                   preserve_range=True, order=order, cval=cval)

    # Apply scaling
    if abs(p["scale"] - 1.0) > 1e-6:
        a = rescale(a, p["scale"], anti_aliasing=True,
                    preserve_range=True, order=order)

    # Apply translation
    dy, dx = p["shift"]
    tform = SimilarityTransform(scale=1.0, rotation=0.0, translation=(dx, dy))
    aligned = warp(a, inverse_map=tform.inverse, output_shape=out_shape,
                   order=order, cval=cval, preserve_range=True)
    return aligned.astype(np.float32)

import numpy as np

def fill_nans_nd(arr, method="linear", max_iters=200, tol=1e-5, kernel_size=3):
    """
    Fill NaNs in an N-D array using:
      - method='linear'   : scipy.interpolate.griddata with nearest fallback
      - method='nearest'  : scipy.ndimage.distance_transform_edt
      - method='iterative': N-D neighbor averaging via scipy.ndimage.convolve

    Parameters
    ----------
    arr : np.ndarray, shape (D0, D1, ..., Dn)
        N-D array with NaNs to fill.
    method : {'linear','iterative','nearest'}
    max_iters : int
        For 'iterative', max passes.
    tol : float
        For 'iterative', stop when max |Δ| on formerly-NaN entries < tol.
    kernel_size : int
        For 'iterative', size of the averaging window along each axis (odd).

    Returns
    -------
    filled : np.ndarray
        Copy of arr with NaNs filled.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy ndarray")

    if arr.ndim < 1:
        raise ValueError("arr must be at least 1-D")

    filled = arr.astype(np.float64, copy=True)  # work in float for NaNs
    nan_mask = np.isnan(filled)

    if not np.any(nan_mask):
        return arr.copy()

    if np.all(nan_mask):
        raise ValueError("All values are NaN; nothing to interpolate from.")

    if method == "nearest":
        from scipy.ndimage import distance_transform_edt
        known = ~nan_mask
        # distance_transform_edt measures distance to nearest zero; pass (~known)
        _, ind = distance_transform_edt(~known, return_indices=True)
        filled[nan_mask] = filled[tuple(ind[:, nan_mask])]
        return filled.astype(arr.dtype, copy=False)

    if method == "linear":
        from scipy.interpolate import griddata

        # Get coordinates of known and missing points in N-D
        coords_known = np.array(np.nonzero(~nan_mask)).T  # (M, N)
        vals_known = filled[~nan_mask]
        coords_miss  = np.array(np.nonzero(nan_mask)).T   # (K, N)

        lin_vals = griddata(coords_known, vals_known, coords_miss, method="linear")

        # Fallback for points outside convex hull
        need_nn = np.isnan(lin_vals)
        if np.any(need_nn):
            nn_vals = griddata(coords_known, vals_known, coords_miss[need_nn], method="nearest")
            lin_vals[need_nn] = nn_vals

        filled[nan_mask] = lin_vals
        return filled.astype(arr.dtype, copy=False)

    if method == "iterative":
        from scipy.ndimage import convolve

        if kernel_size % 2 != 1 or kernel_size < 3:
            raise ValueError("kernel_size should be an odd integer >= 3")

        data = filled.copy()
        data[nan_mask] = 0.0
        weights = (~nan_mask).astype(np.float64)

        # N-D uniform kernel (e.g., 3x3x...x3)
        kernel = np.ones((kernel_size,) * arr.ndim, dtype=np.float64)

        prev_vals = data[nan_mask]
        for _ in range(max_iters):
            num = convolve(data, kernel, mode="nearest")
            den = convolve(weights, kernel, mode="nearest")

            upd_mask = nan_mask & (den > 0)
            new_vals = num[upd_mask] / den[upd_mask]

            data[upd_mask] = new_vals
            weights[upd_mask] = 1.0  # become "known" for subsequent passes

            cur_vals = data[nan_mask]
            max_change = np.nanmax(np.abs(cur_vals - prev_vals))
            if max_change < tol:
                break
            prev_vals = cur_vals

        filled[nan_mask] = data[nan_mask]
        return filled.astype(arr.dtype, copy=False)

    raise ValueError("method must be 'linear', 'iterative', or 'nearest'")
from skimage.transform import resize

def shrink_array(arr, out_shape, order=1):
    """
    Resize an array to out_shape.
    Good for downsampling when there are no NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    out_shape : (H, W)
        Target shape.
    order : int
        Interpolation order: 1=bilinear (smooth), 3=bicubic (sharper).

    Returns
    -------
    np.ndarray
        Resized array with values scaled to preserve range.
    """
    return resize(
        arr, out_shape,
        order=order,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(arr.dtype)

def rescale_to_shared_minmax(a, b):
    """
    Affinely rescale A and B so both end up with the same min/max:
      min(A') = min(B') = min(A,B)
      max(A') = max(B') = max(A,B)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    gmin = min(np.nanmin(a), np.nanmin(b))
    gmax = max(np.nanmax(a), np.nanmax(b))
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise ValueError("Inputs contain no finite values.")
    if gmax == gmin:
        raise ValueError("Global range is zero; cannot rescale.")

    def scale(x):
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if xmax == xmin:
            raise ValueError("One input has zero range; cannot map to a nonzero global range.")
        return (x - xmin) / (xmax - xmin) * (gmax - gmin) + gmin

    a2 = scale(a)
    b2 = scale(b)
    return a2, b2, (gmin, gmax)

# ----------------- Example -----------------
if __name__ == "__main__":
    # dummy example arrays
    g = np.random.rand(400, 600)
    d = np.random.rand(200, 300)

    aligned, info = align_dat_to_gerber(g, d)
    print("Transform params:", info)
    print("Output shape:", aligned.shape)
