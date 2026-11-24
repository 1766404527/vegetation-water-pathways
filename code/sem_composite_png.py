"""Create the final seven-panel Figure 9 with one shared coefficient scale."""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import xarray as xr


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")
OUT_DIR = os.path.join(RESULT, "sem_s")

FS = 16
TICK_FS = 15
TITLE_FS = 17
matplotlib.rcParams["font.size"] = FS

PANELS = [
    ("SM Direct", "a) SM direct"),
    ("PRE Direct", "b) PRE direct"),
    ("VPD Direct", "c) VPD direct"),
    ("ET Direct", "d) ET direct"),
    ("PRE Total", "e) PRE total"),
    ("VPD Total", "f) VPD total"),
    ("ET Total", "g) ET total"),
]

red = np.array([250, 127, 111]) / 255.0
white = np.ones(3)
blue = np.array([130, 176, 210]) / 255.0
CMAP = LinearSegmentedColormap.from_list("path_effect", [red, white, blue])
NORM = Normalize(vmin=-0.8, vmax=0.8)


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def nearest_index(centers, values):
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    index = np.searchsorted(centers, values)
    index = np.clip(index, 1, len(centers) - 1)
    left = index - 1
    return np.where(
        np.abs(centers[index] - values) < np.abs(values - centers[left]), index, left
    )


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(RESULT, "PLS_SEM_ETnolag.csv"))
    with xr.open_dataset(os.path.join(DATA, "2010_06_svvpe_zv.nc")) as ds:
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
    lat_index = nearest_index(lat, data["lat"].to_numpy(float))
    lon_index = nearest_index(lon, data["lon"].to_numpy(float))

    fig, axes = plt.subplots(
        4, 2, figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    mesh = None
    for panel_i, (column, title) in enumerate(PANELS):
        row, col = divmod(panel_i, 2)
        ax = axes[row, col]
        grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
        grid[lat_index, lon_index] = data[column].to_numpy(np.float32)
        mesh = ax.pcolormesh(
            centers_to_edges(lon),
            centers_to_edges(lat),
            np.ma.masked_invalid(grid),
            cmap=CMAP,
            norm=NORM,
            shading="flat",
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent([-180, 180, -60.01, 60.01])
        ax.add_feature(cfeature.LAND, facecolor="#D9D9D9", edgecolor="black", linewidth=0.35)
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3)
        gl = ax.gridlines(
            draw_labels=True,
            xlocs=[-120, -60, 0, 60, 120],
            ylocs=[-60, -40, -20, 0, 20, 40, 60],
            linewidth=0.35,
            color="#999999",
            alpha=0.55,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = col == 0
        gl.bottom_labels = row == 3
        gl.xlabel_style = {"size": TICK_FS}
        gl.ylabel_style = {"size": TICK_FS}
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)

    axes[3, 1].set_axis_off()
    axes[3, 1].text(
        0.5,
        0.57,
        "Total-effect decomposition\n\n"
        "PRE: direct + PRE→SM→VOD\n"
        "ET: direct + ET→SM→VOD\n"
        "VPD: direct + VPD→SM→VOD\n"
        "       + VPD→ET→VOD\n"
        "       + VPD→ET→SM→VOD",
        transform=axes[3, 1].transAxes,
        ha="center",
        va="center",
        fontsize=FS,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F7F7F7", "edgecolor": "#777777"},
    )
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.055, top=0.965, wspace=0.06, hspace=0.16)
    color_ax = fig.add_axes([0.925, 0.09, 0.018, 0.82])
    colorbar = fig.colorbar(mesh, cax=color_ax)
    colorbar.set_label("Standardized path coefficient", fontsize=FS)
    colorbar.ax.tick_params(labelsize=TICK_FS)

    output = os.path.join(OUT_DIR, "Figure_9_path_coefficients.png")
    plt.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")
