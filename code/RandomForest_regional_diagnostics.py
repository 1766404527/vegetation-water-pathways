"""Figure 7: full-period MDI, OOB drop-column importance, and RF diagnostics.

The regional analysis uses the regularised full-period model, measures the loss
of OOB R2 after dropping each predictor, and summarizes spatial uncertainty with
a block bootstrap.
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

import RandomForest as rf_main
from RandomForest_regional_validation import REGIONS


FS = 18
matplotlib.rcParams["font.size"] = FS
BOOTSTRAP_REPEATS = 1000
BLOCK_DEGREES = 2.0

FACTORS = ["SM", "VPD", "ET", "PRE"]
COLORS = {"SM": "#E69F00", "VPD": "#D55E00", "ET": "#009E73", "PRE": "#0072B2"}


def spatial_block_mean_ci(data: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    """Equal-area-ish 2-degree block bootstrap to avoid pixel-level pseudo-precision."""
    work = data[["lat", "lon", column]].dropna().copy()
    if work.empty:
        return np.nan, np.nan, np.nan
    work["lat_block"] = np.floor((work["lat"] + 60.0) / BLOCK_DEGREES).astype(int)
    work["lon_block"] = np.floor((work["lon"] + 180.0) / BLOCK_DEGREES).astype(int)
    blocks = work.groupby(["lat_block", "lon_block"])[column].mean().to_numpy()
    estimate = float(blocks.mean())
    if blocks.size == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draw = rng.choice(blocks, size=(BOOTSTRAP_REPEATS, blocks.size), replace=True).mean(axis=1)
    low, high = np.quantile(draw, [0.025, 0.975])
    return estimate, float(low), float(high)


def collect(panel: str, pixels: pd.DataFrame):
    values = {factor: [] for factor in FACTORS}
    lower = {factor: [] for factor in FACTORS}
    upper = {factor: [] for factor in FACTORS}
    for region_i, (letter, _name, *_bounds) in enumerate(REGIONS):
        region = pixels[(pixels["region"] == letter) & (pixels["OOB_R2"] > 0)]
        for factor_i, factor in enumerate(FACTORS):
            column = f"{factor} Importance" if panel == "mdi" else f"{factor} DropCol OOB"
            estimate, lo, hi = spatial_block_mean_ci(
                region, column, seed=20260711 + 100 * region_i + factor_i
            )
            values[factor].append(estimate)
            lower[factor].append(estimate - lo)
            upper[factor].append(hi - estimate)
    return values, lower, upper


def draw_grouped(ax, values, lower, upper, ylabel, panel_letter):
    x = np.arange(len(REGIONS))
    width = 0.19
    offsets = {factor: (i - 1.5) * width for i, factor in enumerate(FACTORS)}
    matrix = np.asarray([values[factor] for factor in FACTORS])
    top = np.nanargmax(matrix, axis=0)
    for factor_i, factor in enumerate(FACTORS):
        y = np.asarray(values[factor], dtype=float)
        errors = np.vstack([lower[factor], upper[factor]])
        bars = ax.bar(
            x + offsets[factor],
            y,
            width,
            color=COLORS[factor],
            edgecolor="#555555",
            linewidth=0.5,
            yerr=errors,
            capsize=2,
            error_kw={"elinewidth": 0.8},
            label=factor,
        )
        for region_i, bar in enumerate(bars):
            if top[region_i] == factor_i:
                yv = bar.get_height()
                span = max(0.01, np.nanmax(matrix) - min(0.0, np.nanmin(matrix)))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yv + 0.045 * span,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    fontweight="bold",
                )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([region[0] for region in REGIONS])
    ax.set_ylabel(ylabel)
    ax.text(0.01, 0.96, panel_letter, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    low_value = float(np.nanmin(matrix - np.asarray([lower[f] for f in FACTORS])))
    high_value = float(np.nanmax(matrix + np.asarray([upper[f] for f in FACTORS])))
    span = max(0.01, high_value - min(0.0, low_value))
    ax.set_ylim(min(0.0, low_value - 0.08 * span), high_value + 0.18 * span)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10), framealpha=0.75)


if __name__ == "__main__":
    path = os.path.join(rf_main.RF_RESULT, "RF_regional_validation_pixels.csv")
    pixels = pd.read_csv(path)

    mdi, mdi_lo, mdi_hi = collect("mdi", pixels)
    drop, drop_lo, drop_hi = collect("drop", pixels)

    summary_rows = []
    for j, (letter, name, *_bounds) in enumerate(REGIONS):
        for factor in FACTORS:
            summary_rows.append(
                {
                    "region": letter,
                    "region_name": name,
                    "factor": factor,
                    "MDI_block_mean": mdi[factor][j],
                    "MDI_CI_low": mdi[factor][j] - mdi_lo[factor][j],
                    "MDI_CI_high": mdi[factor][j] + mdi_hi[factor][j],
                    "DropCol_block_mean": drop[factor][j],
                    "DropCol_CI_low": drop[factor][j] - drop_lo[factor][j],
                    "DropCol_CI_high": drop[factor][j] + drop_hi[factor][j],
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(rf_main.RF_RESULT, "RF_regional_figure_summary.csv"), index=False
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 12.5), sharex=True)
    draw_grouped(axes[0], mdi, mdi_lo, mdi_hi, "Full-period MDI importance", "a)")
    draw_grouped(
        axes[1],
        drop,
        drop_lo,
        drop_hi,
        r"OOB drop-column importance ($\Delta R^2_{OOB}$)",
        "b)",
    )

    x = np.arange(len(REGIONS))
    positive = []
    agreement = []
    for letter, _name, *_bounds in REGIONS:
        region = pixels[pixels["region"] == letter]
        positive.append(100.0 * (region["OOB_R2"] > 0).mean())
        agreement.append(100.0 * region["Dominant_agreement_early_full"].dropna().mean())
    axes[2].bar(x - 0.18, positive, 0.36, color="#4C78A8", edgecolor="#555555", label=r"OOB $R^2>0$")
    axes[2].bar(
        x + 0.18,
        agreement,
        0.36,
        color="#9C755F",
        edgecolor="#555555",
        label="Early/full leading-factor agreement",
    )
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Pixels (%)")
    axes[2].set_xlabel("Representative region (A-G)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([region[0] for region in REGIONS])
    axes[2].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), framealpha=0.75)
    axes[2].text(0.01, 0.96, "c)", transform=axes[2].transAxes, ha="left", va="top", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout()
    output = os.path.join(rf_main.RF_RESULT, "MDI_vs_OOB_dropcolumn_importance.png")
    plt.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")

    print("\nRegion  MDI-top  Drop-column-top  OOB-positive(%)  Early/full agreement(%)")
    for j, (letter, _name, *_bounds) in enumerate(REGIONS):
        mdi_top = FACTORS[int(np.nanargmax([mdi[f][j] for f in FACTORS]))]
        drop_top = FACTORS[int(np.nanargmax([drop[f][j] for f in FACTORS]))]
        print(f"{letter:>3s} {mdi_top:>8s} {drop_top:>16s} {positive[j]:16.1f} {agreement[j]:24.1f}")
