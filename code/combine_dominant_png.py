"""Plot the optimized RF dominance map with an explicit low-skill class.

Strength is the full-period MDI leading factor.  The plus/minus annotation is
the sign of the corresponding full-period marginal Pearson correlation and is
not presented as an RF partial-effect direction.  Pixels with OOB R2 <= 0 are
shown as low-skill/uncertain instead of being forced into a dominant class.
The SMOS-IC analysis grid uses equal-area cells, so global percentages use
equal cell weights.
"""

from __future__ import annotations

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import xarray as xr


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")
RF_RESULT = os.path.join(RESULT, "RandomForest Importance")
FS = 14
matplotlib.rcParams["font.size"] = FS

MERGED = os.path.join(RF_RESULT, "pearsonr_importancenolag.csv")
df = pd.read_csv(MERGED)
required = {"OOB_R2", "SM Importance", "VPD Importance", "PRE Importance", "ET Importance"}
missing = sorted(required.difference(df.columns))
if missing:
    raise RuntimeError(f"Run the optimized RandomForest.py and combine.py first; missing {missing}")

CATEGORIES = [
    ("SM", "+", "#E69F00"),
    ("SM", "-", "#A66F00"),
    ("PRE", "+", "#56B4E9"),
    ("PRE", "-", "#0072B2"),
    ("VPD", "+", "#F07F5A"),
    ("VPD", "-", "#D55E00"),
    ("ET", "+", "#009E73"),
    ("ET", "-", "#006B4F"),
    ("Low OOB skill", "", "#BDBDBD"),
]
FEATURES = ["SM", "VPD", "PRE", "ET"]
MDI_COLUMNS = [f"{name} Importance" for name in FEATURES]
CORR_COLUMNS = ["VOD_SM", "VOD_VPD", "VOD_PRE", "VOD_ET"]

mdi = df[MDI_COLUMNS].to_numpy(float)
corr = df[CORR_COLUMNS].to_numpy(float)
leader = np.argmax(mdi, axis=1)
row = np.arange(len(df))
leader_corr = corr[row, leader]
supported = (df["OOB_R2"].to_numpy(float) > 0) & np.isfinite(leader_corr) & (leader_corr != 0)

# Factor codes are arranged as positive/negative pairs in CATEGORIES.
factor_to_pair = np.array([0, 4, 2, 6])  # FEATURES order -> first code of each pair
codes = np.full(len(df), 8, dtype=np.int16)
codes[supported] = factor_to_pair[leader[supported]] + (leader_corr[supported] < 0).astype(np.int16)

with xr.open_dataset(os.path.join(DATA, "2010_06_svvpe_zv.nc")) as ds:
    grid_lats = ds["lat"].values.astype(float)
    grid_lons = ds["lon"].values.astype(float)


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def nearest_index(centers, values):
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    index = np.searchsorted(centers, values, side="left")
    index = np.clip(index, 1, len(centers) - 1)
    left = index - 1
    choose_right = np.abs(centers[index] - values) < np.abs(values - centers[left])
    return np.where(choose_right, index, left)


lat_index = nearest_index(grid_lats, df["lat"].to_numpy(float))
lon_index = nearest_index(grid_lons, df["lon"].to_numpy(float))
grid_code = np.full((len(grid_lats), len(grid_lons)), -1, dtype=np.int16)
grid_code[lat_index, lon_index] = codes

colors = [category[2] for category in CATEGORIES]
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
masked = np.ma.masked_where(grid_code < 0, grid_code)

# SMOS-IC cells are equal area; applying cosine(latitude) would double-weight
# the latitude dependence already built into the grid geometry.
area_weights = np.ones(len(df), dtype=float)
total_weight = area_weights.sum()
summary_rows = []
for code, (factor, sign, _color) in enumerate(CATEGORIES):
    mask = codes == code
    label = f"{factor} ({sign})" if sign else factor
    summary_rows.append(
        {
            "code": code,
            "category": label,
            "pixel_count": int(mask.sum()),
            "pixel_percent": float(100 * mask.mean()),
            "area_weighted_percent": float(100 * area_weights[mask].sum() / total_weight),
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(RF_RESULT, "RF_dominance_summary.csv"), index=False)

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -60.01, 60.01], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#F2F2F2", edgecolor="black")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.LAKES, facecolor="#D9EEF7")
ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#555555", linewidth=0.6)
gl = ax.gridlines(draw_labels=True, linewidth=0.8, color="gray", alpha=0.5, linestyle="--")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": FS}
gl.ylabel_style = {"size": FS}

ax.pcolormesh(
    centers_to_edges(grid_lons),
    centers_to_edges(grid_lats),
    masked,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    shading="flat",
)

boxes = [
    (-123, -105, 30, 48),
    (56, 83, 45, 54),
    (140, 153, -38, -20),
    (25, 35, -35, -20),
    (16, 23, -18, -9),
    (98, 120, -5, 16),
    (-78, -59, -6, 7),
]
for box_i, (lon0, lon1, lat0, lat1) in enumerate(boxes):
    ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
            color="#800080", linewidth=1.7, transform=ccrs.PlateCarree())
    ax.text(
        lon0 + 0.8,
        lat1 - 1.2,
        chr(ord('A') + box_i),
        color="#800080",
        fontsize=FS,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ccrs.PlateCarree(),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
    )

handles = []
for code, (factor, sign, color) in enumerate(CATEGORIES):
    percent = summary.loc[summary["code"] == code, "area_weighted_percent"].iloc[0]
    if sign:
        label = f"{factor} dominant ({sign}) ({percent:.1f}%)"
    else:
        label = f"{factor} ({percent:.1f}%)"
    handles.append(mpatches.Patch(color=color, label=label))
ax.legend(
    handles=handles,
    loc="center left",
    bbox_to_anchor=(0.00, 0.33),
    ncol=1,
    fontsize=FS,
    framealpha=0.82,
    edgecolor="#555555",
    facecolor="white",
)
ax.set_title("Random-forest leading factors with OOB-skill screening", fontsize=15, pad=20)

output = os.path.join(RF_RESULT, "World_Max_Importance_nolag.png")
plt.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(summary.to_string(index=False))
print(f"Saved {output}")
