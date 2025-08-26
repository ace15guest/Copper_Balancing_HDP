import math
import numpy as np
from pathlib import Path

# Optional: Plotly only needed for backend="plotly"
try:
    import plotly.express as px
    from plotly.offline import plot as plot_offline
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def grid_to_points(grid: np.ndarray,
                   pitch_x: float = 1.0,
                   pitch_y: float = 1.0,
                   origin_x: float = 0.0,
                   origin_y: float = 0.0,
                   downsample: int = 1):
    """
    Convert a 2D grid of Z values -> 3 arrays (X, Y, Z) as points.
    NaNs in Z are dropped automatically.

    pitch_x, pitch_y: map index -> physical units (e.g., mm). Leave at 1.0 for index space.
    origin_x, origin_y: optional offsets in the same units as pitch.
    downsample: take every Nth point along both axes.
    """
    if downsample < 1:
        downsample = 1
    z = grid[::downsample, ::downsample]
    nrows, ncols = z.shape
    yy, xx = np.indices((nrows, ncols))
    X = origin_x + xx * pitch_x * downsample
    Y = origin_y + yy * pitch_y * downsample
    Z = z

    # Flatten and drop NaNs
    Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
    m = ~np.isnan(Zf)
    return Xf[m], Yf[m], Zf[m]


def plot_points_3d(grid: np.ndarray,
                   *,
                   pitch_x: float = 1.0,
                   pitch_y: float = 1.0,
                   origin_x: float = 0.0,
                   origin_y: float = 0.0,
                   backend: str = "plotly",
                   downsample: int | None = None,
                   max_points: int = 200_000,
                   outfile: str | None = None,
                   colorbar_title: str = "Z"):
    """
    Plot a combined 2D array as a 3D POINT CLOUD (no mesh).

    backend: "matplotlib" (PNG) or "plotly" (interactive HTML).
    downsample: if None, auto-choose based on max_points; else use the factor you provide.
    max_points: target cap for performance (used when downsample=None).
    outfile: path to save ('.png' for matplotlib, '.html' for plotly). If None, auto-names next to CWD.
    Returns: (outfile_path, stats)
    """
    nrows, ncols = grid.shape
    total = nrows * ncols
    if downsample is None:
        downsample = 1 if total <= max_points else max(1, int(math.sqrt(total / max_points)))

    X, Y, Z = grid_to_points(grid, pitch_x, pitch_y, origin_x, origin_y, downsample)
    if X.size == 0:
        raise ValueError("No valid points to plot (all NaN after masking).")

    stats = {"shape": (nrows, ncols), "downsample": downsample, "points": int(X.size), "nan_count": int(np.isnan(grid).sum())}

    if backend.lower() == "plotly":
        if not _HAS_PLOTLY:
            raise RuntimeError("Plotly is not installed. `pip install plotly` or use backend='matplotlib'.")
        fig = px.scatter_3d(x=X, y=Y, z=Z, color=Z, opacity=0.9)
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title=colorbar_title),
            coloraxis_colorbar=dict(title=colorbar_title),
            title=f"Points • step={downsample} • points={X.size:,}"
        )
        if outfile is None:
            outfile = "combined_points3d.html"
        plot_offline(fig, filename=outfile, auto_open=False, include_plotlyjs="cdn")
        return outfile, stats

    # Matplotlib backend (PNG)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X, Y, Z, c=Z, s=2)  # don't specify cmap to keep defaults
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(colorbar_title)
    fig.colorbar(sc, shrink=0.6, label=colorbar_title)
    ax.view_init(elev=25, azim=45)
    if outfile is None:
        outfile = "combined_points3d.png"
    plt.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outfile, stats
