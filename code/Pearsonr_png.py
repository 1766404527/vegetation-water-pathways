"""Create Figure 3: global VOD-driver Pearson-correlation maps."""

from __future__ import annotations

import os
import shutil

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import xarray as xr


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")
OUT_DIR = os.path.join(RESULT, "Pearsonr")
OUTPUT = os.path.join(OUT_DIR, "Figure_3_correlations.png")
FINAL_OUTPUT = os.path.join(BASE, "figures", "Figure_3.png")

FS = 17
TICK_FS = 16
TITLE_FS = 18
matplotlib.rcParams["font.size"] = FS

PANELS = [
    ("VOD_SM", "a) VOD-SM"),
    ("VOD_PRE", "b) VOD-PRE"),
    ("VOD_VPD", "c) VOD-VPD"),
    ("VOD_ET", "d) VOD-ET"),
]

RED = np.array([250, 127, 111]) / 255.0
WHITE = np.ones(3)
BLUE = np.array([130, 176, 210]) / 255.0
CMAP = LinearSegmentedColormap.from_list("correlation", [RED, WHITE, BLUE])
NORM = Normalize(vmin=-1.0, vmax=1.0)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def nearest_index(centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    index = np.clip(np.searchsorted(centers, values, side="left"), 1, len(centers) - 1)
    left = index - 1
    choose_right = np.abs(centers[index] - values) < np.abs(values - centers[left])
    return np.where(choose_right, index, left)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    data = pd.read_csv(os.path.join(RESULT, "Worlds_nolag.csv"))
    with xr.open_dataset(os.path.join(DATA, "2010_06_svvpe_zv.nc")) as dataset:
        lat = dataset["lat"].values.astype(float)
        lon = dataset["lon"].values.astype(float)

    lat_index = nearest_index(lat, data["lat"].to_numpy(float))
    lon_index = nearest_index(lon, data["lon"].to_numpy(float))
    lat_edges = centers_to_edges(lat)
    lon_edges = centers_to_edges(lon)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 4.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    mesh = None
    for panel_i, (column, title) in enumerate(PANELS):
        row, col = divmod(panel_i, 2)
        ax = axes[row, col]
        grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
        grid[lat_index, lon_index] = data[column].to_numpy(np.float32)

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(grid),
            cmap=CMAP,
            norm=NORM,
            transform=ccrs.PlateCarree(),
            shading="flat",
        )
        ax.set_extent([-180, 180, -60.01, 60.01], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#D9D9D9", edgecolor="black", linewidth=0.45)
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.LAKES, facecolor="#D9EEF7")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#555555", linewidth=0.4)
        gridlines = ax.gridlines(
            draw_labels=True,
            xlocs=[-120, -60, 0, 60, 120],
            ylocs=[-60, -40, -20, 0, 20, 40, 60],
            linewidth=0.5,
            color="#999999",
            alpha=0.55,
            linestyle="--",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.left_labels = col == 0
        gridlines.bottom_labels = row == 1
        gridlines.xlabel_style = {"size": TICK_FS}
        gridlines.ylabel_style = {"size": TICK_FS}
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=7)

    colorbar = fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.026,
        pad=0.025,
        aspect=32,
    )
    colorbar.set_label("Pearson r", fontsize=FS)
    colorbar.ax.tick_params(labelsize=TICK_FS)

    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(OUTPUT, FINAL_OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Updated {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
