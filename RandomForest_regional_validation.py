"""Regional OOB drop-column validation for the full-period RF importance map.

MDI can allocate shared signal arbitrarily among correlated predictors.  As a
complement, this script refits the same regularised forest after dropping one
predictor at a time and measures the decrease in OOB R2.  The calculation is
restricted to the seven pre-defined representative regions used in the paper;
the global map itself remains based on the final full-period MDI model.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

import RandomForest as rf_main


REGIONAL_TREES = 120
N_JOBS = int(os.environ.get("RF_N_JOBS", min(14, os.cpu_count() or 1)))
CHUNK_SIZE = 150

REGIONS = [
    ("A", "Western North America", 30, 48, -123, -105),
    ("B", "Interior of Eurasia", 45, 54, 56, 83),
    ("C", "Southeastern Australia", -38, -20, 140, 153),
    ("D", "Southeastern Africa", -35, -20, 25, 35),
    ("E", "African Moist Savanna", -18, -9, 16, 23),
    ("F", "Southeast Asia", -5, 16, 98, 120),
    ("G", "Amazon Rainforest", -6, 7, -78, -59),
]


def _forest(n_features: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=REGIONAL_TREES,
        min_samples_leaf=rf_main.MIN_SAMPLES_LEAF,
        # Preserve two candidate variables per split after a column is dropped;
        # otherwise a 0.5 fraction would change from 2/4 to only 1/3.
        max_features=min(2, n_features),
        max_samples=rf_main.MAX_SAMPLES,
        bootstrap=True,
        oob_score=True,
        random_state=rf_main.RANDOM_STATE,
        n_jobs=1,
    )


def select_pixels(final_rf: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for letter, name, lat0, lat1, lon0, lon1 in REGIONS:
        part = final_rf[
            final_rf["lat"].between(lat0, lat1)
            & final_rf["lon"].between(lon0, lon1)
        ].copy()
        part.insert(0, "region", letter)
        part.insert(1, "region_name", name)
        pieces.append(part)
    selected = pd.concat(pieces, ignore_index=True)
    if selected[["lat", "lon"]].duplicated().any():
        raise RuntimeError("Representative regions overlap on the RF grid")
    return selected


def load_selected_cube(selected: pd.DataFrame) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(rf_main.DATA, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {rf_main.DATA}")
    first, lat, lon = rf_main._read_flat(files[0])

    lat_index = np.abs(lat[:, None].astype(float) - selected["lat"].to_numpy()[None, :]).argmin(axis=0)
    lon_index = np.abs(lon[:, None].astype(float) - selected["lon"].to_numpy()[None, :]).argmin(axis=0)
    if not np.allclose(lat[lat_index], selected["lat"], atol=1e-5) or not np.allclose(
        lon[lon_index], selected["lon"], atol=1e-5
    ):
        raise ValueError("RF coordinates do not align with the NetCDF grid")

    flat = lat_index * lon.size + lon_index
    cube = np.full((len(selected), len(files), len(rf_main.ALL_VARS)), np.nan, dtype=np.float32)
    cube[:, 0, :] = first[flat]
    for i, path in enumerate(files[1:], start=1):
        matrix, lat_i, lon_i = rf_main._read_flat(path)
        if not np.array_equal(lat_i, lat) or not np.array_equal(lon_i, lon):
            raise ValueError(f"Grid mismatch in {path}")
        cube[:, i, :] = matrix[flat]
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"Loaded {i + 1}/{len(files)} months", flush=True)
    return cube


def validate_one(series: np.ndarray) -> list[float]:
    matrix = series[np.isfinite(series).all(axis=1)].astype(float)
    y = matrix[:, 0]
    x = matrix[:, 1:]

    baseline = _forest(x.shape[1])
    baseline.fit(x, y)
    baseline_r2 = float(baseline.oob_score_)

    drop_scores = []
    for j in range(len(rf_main.FEATURES)):
        reduced_x = np.delete(x, j, axis=1)
        reduced = _forest(reduced_x.shape[1])
        reduced.fit(reduced_x, y)
        drop_scores.append(baseline_r2 - float(reduced.oob_score_))
    return [baseline_r2, *drop_scores]


def validate_chunk(indices: np.ndarray, cube: np.ndarray):
    return [validate_one(cube[i]) for i in indices]


if __name__ == "__main__":
    started = time.time()
    final_path = os.path.join(rf_main.RESULT, "Worlds_importancenolag.csv")
    final_rf = pd.read_csv(final_path)
    required = {"OOB_R2", "Dominant_agreement_early_full", "MDI_full_top_margin"}
    missing = sorted(required.difference(final_rf.columns))
    if missing:
        raise RuntimeError(
            f"{final_path} is not the optimized full-period RF output; missing columns {missing}"
        )

    selected = select_pixels(final_rf)
    cube = load_selected_cube(selected)
    indices = np.arange(len(selected))
    chunks = [indices[i : i + CHUNK_SIZE] for i in range(0, len(indices), CHUNK_SIZE)]
    print(f"Validating {len(selected):,} regional pixels ...", flush=True)
    nested = Parallel(n_jobs=N_JOBS, verbose=10, max_nbytes="10M", mmap_mode="r")(
        delayed(validate_chunk)(chunk, cube) for chunk in chunks
    )
    values = np.asarray([row for block in nested for row in block], dtype=float)

    selected["OOB_R2_regional_refit"] = values[:, 0]
    for j, feature in enumerate(rf_main.FEATURES):
        selected[f"{feature} DropCol OOB"] = values[:, j + 1]

    keep = [
        "region",
        "region_name",
        "lat",
        "lon",
        "n_months",
        "OOB_R2",
        "OOB_R2_regional_refit",
        "Dominant_agreement_early_full",
        "MDI_full_top_margin",
        *[f"{name} Importance" for name in rf_main.FEATURES],
        *[f"{name} DropCol OOB" for name in rf_main.FEATURES],
    ]
    pixel_out = selected[keep]

    summary_rows = []
    for letter, name, *_ in REGIONS:
        region = pixel_out[pixel_out["region"].eq(letter)]
        skillful = region[region["OOB_R2"] > 0]
        row = {
            "region": letter,
            "region_name": name,
            "n_pixels": int(len(region)),
            "n_pixels_positive_OOB": int(len(skillful)),
            "fraction_positive_OOB": float((region["OOB_R2"] > 0).mean()),
            "median_OOB_R2": float(region["OOB_R2"].median()),
            "early_full_dominant_agreement": float(
                region["Dominant_agreement_early_full"].dropna().mean()
            ),
        }
        for feature in rf_main.FEATURES:
            row[f"{feature}_MDI_mean_positive_OOB"] = float(
                skillful[f"{feature} Importance"].mean()
            )
            row[f"{feature}_DropCol_mean_positive_OOB"] = float(
                skillful[f"{feature} DropCol OOB"].mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    out_dir = rf_main.RF_RESULT
    pixel_path = os.path.join(out_dir, "RF_regional_validation_pixels.csv")
    summary_path = os.path.join(out_dir, "RF_regional_validation_summary.csv")
    pixel_out.to_csv(pixel_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Elapsed minutes: {(time.time() - started) / 60:.1f}", flush=True)
    print(f"Saved {pixel_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
