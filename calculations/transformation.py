import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate, rescale
from skimage.registration import phase_cross_correlation


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

def align_dat_to_gerber(
        gerber_arr,
        dat_arr,
        px_per_mm_gerber=None,
        px_per_mm_dat=None,
        scale_search=(0.7, 1.4, 21)
):
    """
    Align dat_arr onto gerber_arr's grid via rotation+scale+translation.

    Returns:
        aligned_dat : np.ndarray, same shape as gerber_arr
        params : dict with {'rotation_deg','scale','shift':(dy,dx)}
    """
    G = sanitize(gerber_arr)
    D = sanitize(dat_arr)

    if not np.any(G):
        raise ValueError("Gerber array has no finite data.")
    if not np.any(D):
        raise ValueError("DAT array has no finite data.")

    # --- 1) Orientation ---
    angG = dominant_angle_safe(binarize_robust(G))
    angD = dominant_angle_safe(binarize_robust(D))
    rot_needed = angG - angD
    Drot = rotate(D, angle=rot_needed, resize=True, preserve_range=True)

    # --- 2) Scale ---
    if px_per_mm_gerber and px_per_mm_dat:
        scale = float(px_per_mm_gerber) / float(px_per_mm_dat)
    else:
        smin, smax, n = scale_search
        scales = np.linspace(smin, smax, n)
        scale = auto_scale(G, Drot, scales)
    Drs = rescale(Drot, scale, anti_aliasing=True, preserve_range=True)

    # --- 3) Translation ---
    H = min(G.shape[0], Drs.shape[0])
    W = min(G.shape[1], Drs.shape[1])
    Gc = gaussian_filter(G[:H, :W], 1.0)
    Dc = gaussian_filter(Drs[:H, :W], 1.0)
    dy, dx = phase_translate(Gc, Dc)

    # --- 4) Paste onto Gerber-sized canvas ---
    out = np.zeros_like(G, dtype=np.float32)
    sy, sx = int(round(-dy)), int(round(-dx))
    y0, x0 = max(0, sy), max(0, sx)
    y1, x1 = min(G.shape[0], sy + Drs.shape[0]), min(G.shape[1], sx + Drs.shape[1])
    Dy0, Dx0 = max(0, -sy), max(0, -sx)
    Dy1, Dx1 = Dy0 + (y1 - y0), Dx0 + (x1 - x0)

    if (y1 > y0) and (x1 > x0):
        out[y0:y1, x0:x1] = sanitize(Drs[Dy0:Dy1, Dx0:Dx1])

    params = {
        "rotation_deg": float(rot_needed),
        "scale": float(scale),
        "shift": (float(dy), float(dx)),
    }
    return out, params


# ----------------- Example -----------------
if __name__ == "__main__":
    # dummy example arrays
    g = np.random.rand(400, 600)
    d = np.random.rand(200, 300)

    aligned, info = align_dat_to_gerber(g, d)
    print("Transform params:", info)
    print("Output shape:", aligned.shape)
