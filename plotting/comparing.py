import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple

def _prep_xyz_from_array(arr: np.ndarray, ignore_zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten arr -> x,y,z and return finite mask."""
    arr = np.asarray(arr)
    ny, nx = arr.shape
    y, x = np.indices((ny, nx))
    z = arr.astype(float, copy=False)
    mask = np.isfinite(z)
    if ignore_zeros:
        mask &= (z != 0)
    return x[mask].ravel(), y[mask].ravel(), z[mask].ravel(), mask

def _downsample_xyz(x, y, z, max_points: int, seed: int = 0):
    n = z.size
    if n <= max_points:
        return x, y, z
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], z[idx]

def plot_pointclouds_and_heatmaps(
    A: np.ndarray,
    B: np.ndarray,
    out_folder: str,
    out_name: str,
    *,
    max_points: int = 150_000,          # downsample cap for each 3D cloud
    colorscale: str = "Viridis",
    marker_size: int = 2,
    ignore_zeros: bool = False,         # set True if 0 = "no data"
    stats_text: str = "⟶ Stats placeholder: add later",
    open_in_browser: bool = False
) -> str:
    """
    Create a 2x2 Plotly HTML: [A 3D point cloud, B 3D point cloud; A heatmap, B heatmap].
    All plots share identical color mapping. Returns the saved HTML path.

    Args:
        A, B            : 2D arrays (same or different shapes are OK)
        out_folder      : folder to save into (created if missing)
        out_name        : file name ('.html' added if missing)
        max_points      : max points per 3D cloud for speed
        colorscale      : Plotly colorscale name
        marker_size     : 3D point size
        ignore_zeros    : if True, drop zeros from plots (useful if zeros mean padding)
        stats_text      : a placeholder annotation at the top
        open_in_browser : if True, open in default browser

    Returns:
        str: full path to the generated HTML file
    """
    # --- ensure output path ---
    os.makedirs(out_folder, exist_ok=True)
    if not out_name.lower().endswith(".html"):
        out_name += ".html"
    out_path = os.path.join(out_folder, out_name)

    # --- prepare data ---
    # Global color range (finite, optionally ignore zeros)
    def _finite_vals(arr):
        m = np.isfinite(arr)
        if ignore_zeros:
            m &= (arr != 0)
        return arr[m]

    finite_A = _finite_vals(np.asarray(A, float))
    finite_B = _finite_vals(np.asarray(B, float))
    if finite_A.size == 0 or finite_B.size == 0:
        raise ValueError("A and/or B have no finite (and non-zero if ignore_zeros=True) values to plot.")

    vmin = float(np.min([finite_A.min(), finite_B.min()]))
    vmax = float(np.max([finite_A.max(), finite_B.max()]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallback to per-array mins/maxs if identical; still keep same range
        vmin = float(min(np.nanmin(A), np.nanmin(B)))
        vmax = float(max(np.nanmax(A), np.nanmax(B)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            raise ValueError("Could not determine a valid color range.")

    # Build XYZ for 3D clouds (index grid -> coordinates)
    xA, yA, zA, _ = _prep_xyz_from_array(A, ignore_zeros=ignore_zeros)
    xB, yB, zB, _ = _prep_xyz_from_array(B, ignore_zeros=ignore_zeros)

    # Downsample for speed
    xA, yA, zA = _downsample_xyz(xA, yA, zA, max_points=max_points, seed=1)
    xB, yB, zB = _downsample_xyz(xB, yB, zB, max_points=max_points, seed=2)

    # --- figure with 4 subplots ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.06, vertical_spacing=0.12,
        subplot_titles=("Simulation — Point Cloud", "Akro — Point Cloud", "Simulation — Heatmap", "Akro — Heatmap"),
    )

    # 3D point clouds (share colors)
    fig.add_trace(
        go.Scatter3d(
            x=xA, y=yA, z=zA,
            mode="markers",
            marker=dict(size=marker_size, color=zA, colorscale=colorscale, cmin=vmin, cmax=vmax, opacity=1.0),
            name="A"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=xB, y=yB, z=zB,
            mode="markers",
            marker=dict(size=marker_size, color=zB, colorscale=colorscale, cmin=vmin, cmax=vmax, opacity=1.0),
            name="B",
            showlegend=False
        ),
        row=1, col=2
    )

    # Heatmaps (share the same zmin/zmax)
    fig.add_trace(
        go.Heatmap(
            z=np.asarray(A, float),
            colorscale=colorscale, zmin=vmin, zmax=vmax,
            colorbar=dict(title="Z", x=0.46, len=0.4)  # show one colorbar on left-bottom
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=np.asarray(B, float),
            colorscale=colorscale, zmin=vmin, zmax=vmax,
            showscale=False
        ),
        row=2, col=2
    )

    # Scenes & layout
    fig.update_layout(
        title=dict(text=f"{out_name}", x=0.5, y=0.98),
        height=950, width=1400,
        margin=dict(l=70, r=30, t=110, b=70),
    )
    # Equal-ish aspect for both 3D scenes
    fig.update_scenes(
        xaxis_title="X (col)",
        yaxis_title="Y (row)",
        zaxis_title="Z",
        aspectmode="cube",
        camera=dict(eye=dict(x=1.2, y=1.2, z=0.9))
    )

    # Add a stats placeholder area (top-center annotation)
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper", x=0.5, y=1.08,
        showarrow=False, align="center", font=dict(size=12, color="gray")
    )

    # Save
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    if open_in_browser:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out_path))

    return out_path

# -------------------
# Example usage:
# out = plot_pointclouds_and_heatmaps(A, B, r"C:\temp\plots", "ab_compare.html",
#                                     max_points=120_000, colorscale="Viridis",
#                                     ignore_zeros=True,
#                                     stats_text="RMSE: —  |  ρ: —  |  n: —",
#                                     open_in_browser=True)
# print("Saved:", out)
