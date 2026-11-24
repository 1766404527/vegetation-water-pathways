"""Complete-case pixel-wise correlations on the common n >= 30 analysis mask."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import RandomForest as rf_main


PAIR_DEFINITIONS = [
    ("VOD_SM", 0, 1),
    ("VOD_PRE", 0, 3),
    ("VOD_ET", 0, 4),
    ("VOD_VPD", 0, 2),
    ("SM_PRE", 1, 3),
    ("SM_ET", 1, 4),
    ("SM_VPD", 1, 2),
    ("PRE_ET", 3, 4),
    ("PRE_VPD", 3, 2),
    ("ET_VPD", 4, 2),
]


def pairwise_complete_correlations(cube: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # RandomForest.load_pixel_cube returns (pixel, time, variable).  All five
    # variables must be finite in a month, matching RF and structural analysis.
    complete = np.isfinite(cube).all(axis=2)
    n = complete.sum(axis=1).astype(np.int32)
    clean = np.where(complete[:, :, None], cube, 0.0)

    sums = clean.sum(axis=1, dtype=np.float64)
    sum_squares = (clean * clean).sum(axis=1, dtype=np.float64)
    centered_ss = sum_squares - sums * sums / n[:, None]
    centered_ss = np.maximum(centered_ss, 0.0)

    output = {}
    for name, i, j in PAIR_DEFINITIONS:
        cross = (clean[:, :, i] * clean[:, :, j]).sum(axis=1, dtype=np.float64)
        centered_cross = cross - sums[:, i] * sums[:, j] / n
        denominator = np.sqrt(centered_ss[:, i] * centered_ss[:, j])
        with np.errstate(invalid="ignore", divide="ignore"):
            output[name] = centered_cross / denominator
        output[name][denominator == 0] = np.nan
    return n, output


if __name__ == "__main__":
    cube, lat, lon, _dates = rf_main.load_pixel_cube()
    n, correlations = pairwise_complete_correlations(cube)

    table = pd.DataFrame({"lat": lat, "lon": lon, "n_months_corr": n, **correlations})
    if len(table) != 99_758:
        print(f"Note: eligible-pixel count is {len(table):,}; verify if the input dataset changed.")
    if table[["lat", "lon"]].duplicated().any():
        raise RuntimeError("Duplicate correlation coordinates")

    output = os.path.join(rf_main.RESULT, "Worlds_nolag.csv")
    table.to_csv(output, index=False)
    print(f"Correlation results saved to {output} ({len(table):,} rows)")
