"""Create Figure 4: five global Hamed-Rao trend maps."""

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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import xarray as xr


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
MK_DIR = os.path.join(BASE, "result1", "mk")
OUTPUT = os.path.join(MK_DIR, "Figure_4_trends.png")
FINAL_OUTPUT = os.path.join(BASE, "figures", "Figure_4.png")

FS = 17
TICK_FS = 15
TITLE_FS = 18
matplotlib.rcParams["font.size"] = FS

PANELS = [
    ("VOD", "a) VOD"),
    ("SM", "b) SM"),
    ("PRE", "c) PRE"),
    ("VPD", "d) VPD"),
    ("ET", "e) ET"),
]

RED = np.array([250, 127, 111]) / 255.0
WHITE = np.ones(3)
BLUE = np.array([130, 176, 210]) / 255.0
CMAP = LinearSegmentedColormap.from_list("trend", [RED, WHITE, BLUE])


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


def trend_grid(variable: str, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, Normalize]:
    data = pd.read_csv(os.path.join(MK_DIR, f"{variable.lower()}_mk.csv"))
    significant = data[data["Trend"] == True].copy()  # noqa: E712
    if significant.empty:
        raise RuntimeError(f"No significant {variable} trend pixels")

    lower = float(significant["Slope"].quantile(0.05))
    upper = float(significant["Slope"].quantile(0.95))
    significant["Slope"] = significant["Slope"].clip(lower, upper)
    limit = max(abs(lower), abs(upper))

    lat_index = nearest_index(lat, significant["Latitude"].to_numpy(float))
    lon_index = nearest_index(lon, significant["Longitude"].to_numpy(float))
    grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    grid[lat_index, lon_index] = significant["Slope"].to_numpy(np.float32)
    return grid, Normalize(vmin=-limit, vmax=limit)


def main() -> None:
    with xr.open_dataset(os.path.join(DATA, "2010_06_svvpe_zv.nc")) as dataset:
        lat = dataset["lat"].values.astype(float)
        lon = dataset["lon"].values.astype(float)
    lat_edges = centers_to_edges(lat)
    lon_edges = centers_to_edges(lon)

    fig = plt.figure(figsize=(12, 9.6))
    layout = fig.add_gridspec(3, 4)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.07, top=0.95, wspace=0.42, hspace=0.40)
    specifications = [
        layout[0, 1:3],
        layout[1, 0:2],
        layout[1, 2:4],
        layout[2, 0:2],
        layout[2, 2:4],
    ]

    for (variable, title), specification in zip(PANELS, specifications):
        ax = fig.add_subplot(specification, projection=ccrs.PlateCarree())
        grid, norm = trend_grid(variable, lat, lon)
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.ma.masked_invalid(grid),
            cmap=CMAP,
            norm=norm,
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
        gridlines.xlabel_style = {"size": TICK_FS}
        gridlines.ylabel_style = {"size": TICK_FS}
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=7)

        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            fraction=0.075,
            pad=0.13,
            aspect=28,
        )
        colorbar.ax.tick_params(labelsize=TICK_FS)
        colorbar.set_label(r"Sen slope (z-score month$^{-1}$)", fontsize=TICK_FS)

    no_trend = mpatches.Patch(facecolor="#D9D9D9", edgecolor="#777777", label="No significant trend")
    first_ax = fig.axes[0]
    first_ax.legend(
        handles=[no_trend],
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#777777",
        facecolor="white",
        fontsize=TICK_FS,
    )
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(OUTPUT, FINAL_OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Updated {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
