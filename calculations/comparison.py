import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- 1) Metrics ----------
import numpy as np
from typing import Dict, Tuple, Optional

# ---------- helpers ----------
def _as_points(arr: np.ndarray, ignore_zeros: bool) -> np.ndarray:
    """Turn a 2D array into Nx3 points (x,y,z) on its index grid."""
    z = np.asarray(arr, float)
    m = np.isfinite(z)
    if ignore_zeros:
        m &= (z != 0)
    if not np.any(m):
        raise ValueError("No finite (and non-zero) samples found.")
    y, x = np.indices(z.shape)
    return np.column_stack((x[m], y[m], z[m]))

def _kabsch_umeyama(P: np.ndarray, Q: np.ndarray, with_scaling: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Kabsch/Umeyama rigid alignment: find R,t,(s) minimizing ||s R P + t - Q||_F.
    P,Q are Nx3 with 1-1 correspondence.
    Returns (R, t, s) so that P_aligned = s*(P @ R.T) + t
    """
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must be Nx3 with same N.")
    # centroids
    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    X = P - muP
    Y = Q - muQ
    # covariance
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # ensure right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if with_scaling:
        # Umeyama isotropic scale
        varP = np.sum(np.sum(X**2, axis=1))
        s = float(np.sum(S) / max(varP, 1e-12))
    else:
        s = 1.0
    t = muQ - s * (R @ muP)
    return R, t, s

def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares plane z ≈ ax + by + c."""
    A = np.column_stack([x, y, np.ones_like(x)])
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
    return float(a), float(b), float(c)

def _detrend_plane(points: np.ndarray) -> np.ndarray:
    """Subtract best-fit plane from the z of Nx3 points (x,y,z)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    a, b, c = _fit_plane(x, y, z)
    z_d = z - (a*x + b*y + c)
    out = points.copy()
    out[:, 2] = z_d
    return out

def _downsample_pairs(P: np.ndarray, Q: np.ndarray, maxN: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly keep up to maxN corresponding rows."""
    N = P.shape[0]
    if N <= maxN:
        return P, Q
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=maxN, replace=False)
    return P[idx], Q[idx]

# ---------- main: align + stats ----------
def align_and_compare(
    A: np.ndarray,
    B: np.ndarray,
    *,
    ignore_zeros: bool = True,
    detrend: bool = True,          # remove each surface’s best-fit plane before stats
    with_scaling: bool = False,    # allow uniform scale in Kabsch (default off)
    maxN_align: int = 20000        # downsample correspondences for alignment speed
) -> Dict[str, float]:
    """
    Align A->B with rigid SVD (optionally scaling), then compute stats.
    Assumes A and B represent the same grid (1-1 correspondence by index).
    Returns a dict of metrics and the text block for annotation.
    """
    # 1) Build point sets
    P = _as_points(A, ignore_zeros=ignore_zeros)  # Nx3 (x,y,z_A)
    Q = _as_points(B, ignore_zeros=ignore_zeros)  # Nx3 (x,y,z_B)

    # 2) Use common indices only (in case masks differed)
    # Map (x,y) -> z and intersect keys
    kP = {(int(px), int(py)) for px, py in P[:, :2]}
    kQ = {(int(qx), int(qy)) for qx, qy in Q[:, :2]}
    common = kP & kQ
    if not common:
        raise ValueError("No overlapping (x,y) samples between A and B after masking.")
    # Build aligned arrays in the same (x,y) order
    def _points_from_keys(arr, keys):
        z = arr.astype(float)
        pts = []
        for (x, y) in keys:
            pts.append((x, y, z[y, x]))
        return np.array(pts, float)

    keys_sorted = sorted(common)  # deterministic order
    Pfull = _points_from_keys(A, keys_sorted)
    Qfull = _points_from_keys(B, keys_sorted)

    # 3) (Optional) downsample for alignment only
    Pfit, Qfit = _downsample_pairs(Pfull, Qfull, maxN=maxN_align, seed=123)

    # 4) Kabsch / Umeyama (rigid, optional scale)
    R, t, s = _kabsch_umeyama(Pfit, Qfit, with_scaling=with_scaling)

    # Apply transform to ALL correspondences
    Paligned = s * (Pfull @ R.T) + t  # Nx3

    # 5) (Optional) detrend each surface before stats
    if detrend:
        Paligned = _detrend_plane(Paligned)
        Qd        = _detrend_plane(Qfull)
    else:
        Qd = Qfull

    # 6) Metrics
    diff3d = Paligned - Qd
    dists = np.linalg.norm(diff3d, axis=1)
    dz = Paligned[:, 2] - Qd[:, 2]

    def _pct(a, p): return float(np.percentile(a, p))

    rmse_3d = float(np.sqrt(np.mean(dists**2)))
    mae_3d  = float(np.mean(np.abs(dists)))
    p95_3d  = _pct(dists, 95)
    max_3d  = float(np.max(dists))

    rmse_z = float(np.sqrt(np.mean(dz**2)))
    mae_z  = float(np.mean(np.abs(dz)))
    p95_z  = _pct(np.abs(dz), 95)
    max_z  = float(np.max(np.abs(dz)))

    # Correlation + linear fit of heights
    zA = Paligned[:, 2]
    zB = Qd[:, 2]
    # Pearson r
    r = float(np.corrcoef(zA, zB)[0, 1]) if zA.size > 1 else np.nan
    # Linear regression zB ~ a*zA + b
    Areg = np.column_stack([zA, np.ones_like(zA)])
    a, b = np.linalg.lstsq(Areg, zB, rcond=None)[0]
    # R^2
    yhat = a * zA + b
    ss_res = float(np.sum((zB - yhat)**2))
    ss_tot = float(np.sum((zB - np.mean(zB))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    stats = {
        # transform
        "scale": s,
        "R00": float(R[0,0]), "R01": float(R[0,1]), "R02": float(R[0,2]),
        "R10": float(R[1,0]), "R11": float(R[1,1]), "R12": float(R[1,2]),
        "R20": float(R[2,0]), "R21": float(R[2,1]), "R22": float(R[2,2]),
        "t_x": float(t[0]), "t_y": float(t[1]), "t_z": float(t[2]),
        # distances
        "rmse_3d": rmse_3d, "mae_3d": mae_3d, "p95_3d": p95_3d, "max_3d": max_3d,
        # vertical-only
        "rmse_z": rmse_z, "mae_z": mae_z, "p95_z": p95_z, "max_z": max_z,
        # height relationship
        "pearson_r": r, "slope": float(a), "intercept": float(b), "r2": r2,
        # counts
        "n": int(Paligned.shape[0]),
        "detrended": bool(detrend),
        "with_scaling": bool(with_scaling),
    }

    # One-line summary you can drop into your Plotly annotation:
    stats_text = (
        f"n={stats['n']} | rmse₃ᴅ={stats['rmse_3d']:.4g} | p95₃ᴅ={stats['p95_3d']:.4g} | "
        f"rmse_z={stats['rmse_z']:.4g} | p95_z={stats['p95_z']:.4g} | "
        f"r={stats['pearson_r']:.3f} | zB≈{stats['slope']:.3f}·zA+{stats['intercept']:.3f} | "
        f"R²={stats['r2']:.3f} | scale={stats['scale']:.4f}"
        + (" | detrended" if detrend else "")
    )

    stats["text"] = stats_text
    return stats


# ---------- 3) Optional: residual heatmap for 2D arrays ----------
def residual_heatmap_2d(A, B, title="Residual heatmap (B - A)"):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("residual_heatmap_2d expects 2D arrays.")
    if A.shape != B.shape:
        raise ValueError(f"Shapes must match, got {A.shape} vs {B.shape}")

    diff = B - A
    vmax = float(np.nanmax(np.abs(diff)))
    fig = go.Figure(go.Heatmap(z=diff, colorscale="RdBu", zmin=-vmax, zmax=vmax,
                               colorbar=dict(title="B - A")))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


import numpy as np
if __name__ == '__main__':
    def make_flatness_example(H=300, W=450, Lx_mm=150.0, Ly_mm=100.0, seed=0):
        """
        Generate two PCB-like flatness maps (A_um, B_um) in micrometers (µm),
        slightly offset from each other. Also returns (dx_mm, dy_mm) spacing.

        H, W   : grid size
        Lx_mm  : board length in X (mm)
        Ly_mm  : board length in Y (mm)
        seed   : RNG seed for reproducible noise
        """
        rng = np.random.default_rng(seed)

        # Grid + normalized coords
        y, x = np.mgrid[0:H, 0:W]
        x_n = (x - W/2) / (W/2)
        y_n = (y - H/2) / (H/2)

        # Base bow/twist/tilt (smooth, low-frequency) in µm
        bow   = 60.0 * (0.9*x_n**2 + 1.2*y_n**2)
        twist = 35.0 * (x_n * y_n)
        tilt  = 10.0 * (0.02*x_n - 0.015*y_n)

        # Edge curl
        edge_curl = 12.0 * (
            np.exp(-((np.abs(x_n)-1.0)/0.10)**2) +
            np.exp(-((np.abs(y_n)-1.0)/0.10)**2)
        )

        # Localized bump/dimple
        def gaussian2d(xn, yn, x0, y0, sx, sy, amp):
            return amp * np.exp(-(((xn-x0)**2)/(2*sx**2) + ((yn-y0)**2)/(2*sy**2)))

        bump_A = gaussian2d(x_n, y_n, x0= 0.10, y0=-0.12, sx=0.20, sy=0.16, amp=14.0)
        bump_B = gaussian2d(x_n, y_n, x0= 0.13, y0=-0.10, sx=0.21, sy=0.17, amp=13.0)

        # A: baseline + ripple + small noise
        A_um = bow + twist + tilt + bump_A + edge_curl
        A_um += 1.5 * np.sin(2*np.pi*x/(W/18))
        A_um += 0.8 * rng.standard_normal((H, W))

        # B: slight twist change + small global offset + shifted bump + ripple + noise
        B_um = (bow + 0.95*twist + tilt + bump_B + edge_curl) + 5.0  # +5 µm offset
        B_um += 1.5 * np.sin(2*np.pi*(x+6)/(W/18))
        B_um += 0.8 * rng.standard_normal((H, W))

        dx_mm, dy_mm = Lx_mm / W, Ly_mm / H
        return A_um.astype(np.float32), B_um.astype(np.float32), (dx_mm, dy_mm)

    A, B, (dx, dy) = make_flatness_example()
    # 1) Get numeric metrics
    metrics = compare_arrays_metrics(A, B)
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")

    # 2) See scatter + residual histogram
    fig = comparison_scatter_and_hist(A, B, title="DAT vs Gerber-scaled")
    fig.show()

    # 3) If 2D arrays, also inspect spatial residuals
    # fig_res = residual_heatmap_2d(A, B)
    # fig_res.show()
