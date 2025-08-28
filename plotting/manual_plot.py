import numpy as np
import matplotlib.pyplot as plt

# ---------- utilities ----------

def _sanitize(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def _valid_pair(x, y, use_nonzero_mask=True):
    """Return finite, optional-nonzero overlapping pixels as 1D vectors."""
    x = _sanitize(x)
    y = _sanitize(y)
    m = np.isfinite(x) & np.isfinite(y)
    if use_nonzero_mask:
        m &= (x != 0) | (y != 0)
    xv = x[m].ravel()
    yv = y[m].ravel()
    if xv.size < 10:
        raise ValueError("Not enough overlapping finite pixels to compare.")
    return xv, yv

# ---------- SVD + R^2 ----------

def svd_and_r2(arr_ref, arr_mov, use_nonzero_mask=True, standardize_for_svd=True):
    """
    Compute:
      - SVD of the 2×N data matrix (rows = variables)
      - Linear regression y = a*x + b and R^2

    Returns dict with:
      - s: singular values (length 2, descending)
      - var_explained: tuple (pc1, pc2) variance explained ratios
      - slope, intercept
      - r (Pearson), r2
      - n (points used)
    """
    x, y = _valid_pair(arr_ref, arr_mov, use_nonzero_mask=use_nonzero_mask)
    n = x.size

    # Pearson correlation
    x_mean, y_mean = x.mean(), y.mean()
    x_std,  y_std  = x.std(ddof=1), y.std(ddof=1)
    if x_std == 0 or y_std == 0:
        raise ValueError("One of the arrays is constant over the compared region.")
    r = float(np.cov(x, y, ddof=1)[0,1] / (x_std * y_std))

    # Linear regression y = a*x + b (closed form)
    slope = r * (y_std / x_std)
    intercept = y_mean - slope * x_mean
    # R^2 from correlation for simple linear reg with intercept
    r2 = float(r**2)

    # SVD on centered (optionally standardized) 2×N matrix
    X = np.vstack([x, y]).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    if standardize_for_svd:
        sds = X.std(axis=1, ddof=1, keepdims=True)
        sds[sds == 0] = 1.0
        X = X / sds

    # SVD of X (no 1/sqrt(n-1) scaling here)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Convert singular values to covariance eigenvalues if desired:
    # eigenvalues = (S**2) / (n - 1)
    eig = (S**2) / max(n - 1, 1)
    total = eig.sum()
    var_explained = tuple((eig / total).tolist())  # (pc1, pc2)

    return {
        "s": S.tolist(),
        "var_explained": var_explained,
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r2": float(r2),
        "n": int(n),
    }

# ---------- plotting ----------
def plot_arrays_3d(gerber_arr, dat_arr, max_points=20000):
    """
    3D scatter plot of two aligned arrays with different markers/colors.

    - gerber_arr: reference 2D array
    - dat_arr: aligned 2D array (same shape as gerber_arr)
    """
    g = np.asarray(gerber_arr, dtype=np.float32)
    d = np.asarray(dat_arr, dtype=np.float32)

    # Make sure they're same shape
    H = min(g.shape[0], d.shape[0])
    W = min(g.shape[1], d.shape[1])
    g = g[:H, :W]
    d = d[:H, :W]

    # Coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten
    Xf, Yf = X.ravel(), Y.ravel()
    Gf, Df = g.ravel(), d.ravel()

    # Downsample to avoid millions of points
    N = Xf.size
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        Xf, Yf, Gf, Df = Xf[idx], Yf[idx], Gf[idx], Df[idx]

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Gerber points
    ax.scatter(Xf, Yf, Gf, c='blue', marker='o', s=5, alpha=0.6, label='Gerber')

    # DAT points
    ax.scatter(Xf, Yf, Df, c='red', marker='^', s=5, alpha=0.6, label='DAT')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")
    ax.set_title("Gerber vs. DAT 3D Plot")
    ax.legend()

    plt.tight_layout()
    plt.show()
def plot_overlay_scatter(arr_ref, arr_mov, title=None, use_nonzero_mask=True,
                         point_alpha=0.25, max_points=50000):
    """
    Scatter plot of pixel pairs with regression line and y=x line.
    """
    x, y = _valid_pair(arr_ref, arr_mov, use_nonzero_mask=use_nonzero_mask)

    # Downsample for speed/clarity if huge
    if x.size > max_points:
        idx = np.random.choice(x.size, size=max_points, replace=False)
        x = x[idx]; y = y[idx]

    # Fit line
    x_mean, y_mean = x.mean(), y.mean()
    x_std,  y_std  = x.std(ddof=1), y.std(ddof=1)
    r = float(np.cov(x, y, ddof=1)[0,1] / (x_std * y_std))
    slope = r * (y_std / x_std)
    intercept = y_mean - slope * x_mean
    r2 = r**2

    # Plot
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(x, y, s=2, alpha=point_alpha)
    # y=x line (identity)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1)

    # Regression line
    xr = np.array([lo, hi], dtype=np.float64)
    yr = slope * xr + intercept
    plt.plot(xr, yr, linewidth=2)

    plt.xlabel("Gerber (reference)")
    plt.ylabel("DAT (aligned)")
    if title is None:
        title = "Gerber vs. DAT"
    plt.title(title)
    plt.grid(True, alpha=0.25)

    # Nice annotation
    txt = f"slope={slope:.3f}\nintercept={intercept:.3f}\nR²={r2:.4f}\nr={r:.4f}\nN={x.size}"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   va='top', ha='left', bbox=dict(boxstyle="round", alpha=0.15))

    plt.tight_layout()
    plt.show()

# ---------- example usage ----------
if __name__ == "__main__":
    # Suppose you already created aligned_dat = align_dat_to_gerber(gerber_arr, dat_arr)[0]
    # Demo with synthetic data:
    rng = np.random.default_rng(0)
    g = rng.random((400, 600)).astype(np.float32)
    d = 0.9 * g + 0.05 * rng.standard_normal(g.shape).astype(np.float32) + 0.02  # correlated

    stats = svd_and_r2(g, d, use_nonzero_mask=True, standardize_for_svd=True)
    print(stats)
    rng = np.random.default_rng(0)
    gerber_arr = rng.random((100, 150))
    dat_arr = 0.8 * gerber_arr + 0.2 * rng.random((100, 150))
    plot_overlay_scatter(g, d, title="Gerber vs DAT (scatter+fit)")
    plot_arrays_3d(gerber_arr, dat_arr)
