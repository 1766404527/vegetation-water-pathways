"""Figure 8: regional VOD-driver correlations as point-interval estimates.

For each representative region and driver, the figure reports the equal-weight
mean of 2-degree spatial-block means and a 95% spatial-block bootstrap
confidence interval. Only pixels with positive full-period RF OOB R2 are used.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RandomForest_regional_validation import REGIONS


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "result1")
OUT_DIR = os.path.join(RESULT, "Pearsonr")
INPUT = os.path.join(RESULT, "RandomForest Importance", "pearsonr_importancenolag.csv")
OUTPUT = os.path.join(OUT_DIR, "Fig8_region_boxplots.png")
SUMMARY = os.path.join(OUT_DIR, "Fig8_region_point_intervals.csv")

FS = 17
BOOTSTRAP_REPEATS = 1000
BLOCK_DEGREES = 2.0
Y_LIM = (-0.8, 0.8)
Y_TICKS = np.arange(-0.75, 0.76, 0.25)

PANELS = [
    ("a", "SM", "VOD_SM"),
    ("b", "VPD", "VOD_VPD"),
    ("c", "ET", "VOD_ET"),
    ("d", "PRE", "VOD_PRE"),
]
COLORS = {"SM": "#E69F00", "VPD": "#D55E00", "ET": "#009E73", "PRE": "#0072B2"}


def spatial_block_estimate(
    data: pd.DataFrame, column: str, seed: int
) -> tuple[float, float, float, int, int]:
    """Return block-mean estimate, bootstrap CI, pixel count, and block count."""
    work = data[["lat", "lon", column]].dropna().copy()
    n_pixels = len(work)
    if work.empty:
        return np.nan, np.nan, np.nan, 0, 0

    work["lat_block"] = np.floor((work["lat"] + 60.0) / BLOCK_DEGREES).astype(int)
    work["lon_block"] = np.floor((work["lon"] + 180.0) / BLOCK_DEGREES).astype(int)
    block_means = work.groupby(["lat_block", "lon_block"])[column].mean().to_numpy()
    estimate = float(block_means.mean())

    if block_means.size == 1:
        return estimate, estimate, estimate, n_pixels, 1

    rng = np.random.default_rng(seed)
    bootstrap_means = rng.choice(
        block_means,
        size=(BOOTSTRAP_REPEATS, block_means.size),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return estimate, float(low), float(high), n_pixels, int(block_means.size)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel_i, (_letter, factor, column) in enumerate(PANELS):
        for region_i, (region, region_name, lat_min, lat_max, lon_min, lon_max) in enumerate(REGIONS):
            subset = data[
                data["lat"].between(lat_min, lat_max)
                & data["lon"].between(lon_min, lon_max)
            ]
            estimate, low, high, n_pixels, n_blocks = spatial_block_estimate(
                subset,
                column,
                seed=20260716 + 100 * panel_i + region_i,
            )
            rows.append(
                {
                    "region": region,
                    "region_name": region_name,
                    "factor": factor,
                    "n_pixels_positive_OOB": n_pixels,
                    "n_spatial_blocks": n_blocks,
                    "block_mean_Pearson_r": estimate,
                    "CI95_low": low,
                    "CI95_high": high,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    data = pd.read_csv(INPUT)
    if "OOB_R2" not in data.columns:
        raise RuntimeError("Run RandomForest.py and combine.py before Pearsonr_box.py")
    data = data[data["OOB_R2"] > 0].copy()

    summary = build_summary(data)
    if summary[["block_mean_Pearson_r", "CI95_low", "CI95_high"]].isna().any().any():
        raise RuntimeError("At least one regional factor estimate is missing")
    summary.to_csv(SUMMARY, index=False)

    matplotlib.rcParams["font.size"] = FS
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True, layout="constrained")
    x = np.arange(len(REGIONS))
    region_labels = [region[0] for region in REGIONS]

    for ax, (letter, factor, _column) in zip(axes.flat, PANELS):
        panel = summary[summary["factor"].eq(factor)].set_index("region").loc[region_labels]
        values = panel["block_mean_Pearson_r"].to_numpy()
        lower = values - panel["CI95_low"].to_numpy()
        upper = panel["CI95_high"].to_numpy() - values

        ax.errorbar(
            x,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=7,
            markerfacecolor=COLORS[factor],
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            ecolor=COLORS[factor],
            elinewidth=1.8,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )
        ax.axhline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_title(f"{letter}) {factor}", loc="left", fontweight="bold")
        ax.set_xlim(-0.55, len(REGIONS) - 0.45)
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
        ax.set_xticks(x)
        ax.set_xticklabels(region_labels)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supylabel("Pearson r with VOD")
    fig.supxlabel("Representative region (A-G)")
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
