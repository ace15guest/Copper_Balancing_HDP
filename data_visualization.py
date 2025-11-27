import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Helpers: tidy + filtering
# =========================
def _filter(df, quartile=None, edgefill=None, dpi=None):
    m = pd.Series(True, index=df.index)
    if quartile is not None:
        m &= df["Quartile"].isin([quartile] if isinstance(quartile, str) else quartile)
    if edgefill is not None:
        m &= df["EdgeFill"].isin([edgefill] if isinstance(edgefill, str) else edgefill)
    if dpi is not None:
        # Handle int, float, and NumPy integer types
        if isinstance(dpi, (int, float, np.integer)):
            m &= df["DPI"].isin([dpi])
        else:
            m &= df["DPI"].isin(dpi)
    return df[m].copy()

# ==========================================
# 1) Heatmap: R² by EdgeFill × BlurRadius
#    (one heatmap per DPI)
# ==========================================
def heatmap_r2_by_edgefill_blur(df, dpi, quartile=None, annotate=True):
    """Draw R² heatmap for a specific DPI (optionally a specific Quartile)."""
    d = _filter(df, quartile=quartile, dpi=dpi)
    if d.empty:
        print(f"[heatmap] No data for DPI={dpi}, Quartile={quartile}")
        return None

    pivot = d.pivot_table(values="R2", index="EdgeFill", columns="BlurRadius", aggfunc="mean")
    vals = pivot.values

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, label="Mean R²")
    title = f"R² by EdgeFill × BlurRadius — DPI={dpi}" + (f", Quartile={quartile}" if quartile else "")
    ax.set_title(title)
    ax.set_xlabel("BlurRadius")
    ax.set_ylabel("EdgeFill")

    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()
    return pivot

# ====================================================
# 2) Lines: R² (or r) vs BlurRadius grouped by DPI
#    (one panel per EdgeFill)
# ====================================================
def lines_metric_vs_blur_by_dpi(df, metric="R2", edgefills=None, quartile=None):
    """For each EdgeFill, plot metric vs BlurRadius with one line per DPI."""
    d = _filter(df, quartile=quartile, edgefill=edgefills)
    if d.empty:
        print(f"[lines] No data for given filters")
        return

    for ef, g in d.groupby("EdgeFill"):
        fig = plt.figure()
        ax = plt.gca()
        for dpi, gd in g.groupby("DPI"):
            # Ensure sorted BlurRadius for lines
            gd = gd.sort_values("BlurRadius")
            ax.plot(gd["BlurRadius"], gd[metric], marker="o", label=f"DPI={dpi}")
        ax.set_title(f"{metric} vs BlurRadius — EdgeFill={ef}" + (f", Quartile={quartile}" if quartile else ""))
        ax.set_xlabel("BlurRadius")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1 if metric in ("R2", "r") else None)
        ax.legend(title="DPI", fontsize=8)
        plt.tight_layout()
        plt.show()

# =====================================================
# 3) Scatter: R² vs r, colored by EdgeFill, shaped by DPI
#    (filter to one BlurRadius for crisp comparison)
# =====================================================
def scatter_r2_vs_r(df, blur_radius, quartile=None):
    d = _filter(df, quartile=quartile)
    d = d[d["BlurRadius"] == blur_radius]
    if d.empty:
        print(f"[scatter] No data for BlurRadius={blur_radius}, Quartile={quartile}")
        return

    fig = plt.figure()
    ax = plt.gca()
    # Keep simple styling: one marker set per EdgeFill×DPI
    for (ef, dpi), g in d.groupby(["EdgeFill", "DPI"]):
        ax.scatter(g["R2"], g["r"], label=f"{ef}, {dpi}dpi", s=24)
    ax.set_xlabel("R²")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"R² vs r — BlurRadius={blur_radius}" + (f", Quartile={quartile}" if quartile else ""))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()

# =====================================================
# 4) Tables: top-k combinations by R² within each DPI
# =====================================================
def topk_by_r2(df, k=5, quartile=None):
    d = _filter(df, quartile=quartile)
    out = []
    for dpi, g in d.groupby("DPI"):
        g2 = g.sort_values("R2", ascending=False).head(k)
        g2 = g2[["DPI","EdgeFill","BlurRadius","R2","r"]]
        out.append(g2.assign(Rank=range(1, len(g2)+1)))
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["DPI","EdgeFill","BlurRadius","R2","r","Rank"])

# ============================
# ---------- Usage -----------
# Assuming you already have df with:
# ["Quartile","EdgeFill","DPI","BlurRadius","R2","r"]
# ============================
df = pd.read_excel(r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\Output\results - Copy.xlsx")
# Example 1: Heatmaps (one per DPI)
for dpi_val in sorted(df["DPI"].unique()):
    heatmap_r2_by_edgefill_blur(df, dpi=dpi_val, quartile="Q1")  # or quartile=None

# Example 2: Metric vs blur lines per EdgeFill, grouped by DPI
lines_metric_vs_blur_by_dpi(df, metric="R2", quartile="Q2")
lines_metric_vs_blur_by_dpi(df, metric="r", quartile="Q2")

# Example 3: R² vs r scatter at a chosen blur
scatter_r2_vs_r(df, blur_radius=100, quartile="Q2")

# Example 4: See the top 5 combos by R² for each DPI
best = topk_by_r2(df, k=5, quartile="Q2")
print(best)
