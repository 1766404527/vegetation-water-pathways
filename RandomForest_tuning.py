"""Reproducible global tuning experiment for the pixel-wise random forest.

This script samples pixels across latitude and record-length strata, evaluates a
small pre-declared regularisation grid with OOB R2, and writes both the sampled
coordinates and aggregate scores.  It deliberately does not tune each pixel:
one global configuration is fixed before the full attribution run.
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


SAMPLE_SEED = 8128
SAMPLE_PER_STRATUM = 20
TUNING_TREES = 120
N_JOBS = int(os.environ.get("RF_N_JOBS", min(14, os.cpu_count() or 1)))
SELECTED_LABEL = "leaf_0.075_features_0.50_samples_0.50"

LATITUDE_BREAKS = [-60, -30, 0, 30, 60.001]
MONTH_BREAKS = [29, 59, 99, 139, 157]

CONFIGS = [
    ("default_deep", {"min_samples_leaf": 1, "max_features": 1.0}),
    ("leaf_0.03", {"min_samples_leaf": 0.03, "max_features": 1.0}),
    ("leaf_0.05", {"min_samples_leaf": 0.05, "max_features": 1.0}),
    ("leaf_0.075", {"min_samples_leaf": 0.075, "max_features": 1.0}),
    ("leaf_0.10", {"min_samples_leaf": 0.10, "max_features": 1.0}),
    ("leaf_0.125", {"min_samples_leaf": 0.125, "max_features": 1.0}),
    ("leaf_0.15", {"min_samples_leaf": 0.15, "max_features": 1.0}),
    ("leaf_0.05_features_0.50", {"min_samples_leaf": 0.05, "max_features": 0.50}),
    ("leaf_0.075_features_0.50", {"min_samples_leaf": 0.075, "max_features": 0.50}),
    ("leaf_0.10_features_0.50", {"min_samples_leaf": 0.10, "max_features": 0.50}),
    ("leaf_0.125_features_0.50", {"min_samples_leaf": 0.125, "max_features": 0.50}),
    ("leaf_0.10_features_0.75", {"min_samples_leaf": 0.10, "max_features": 0.75}),
    (
        "leaf_0.075_features_0.50_samples_0.50",
        {"min_samples_leaf": 0.075, "max_features": 0.50, "max_samples": 0.50},
    ),
    (
        "leaf_0.10_features_0.50_samples_0.70",
        {"min_samples_leaf": 0.10, "max_features": 0.50, "max_samples": 0.70},
    ),
]


def load_stratified_sample():
    files = sorted(glob.glob(os.path.join(rf_main.DATA, "*.nc")))
    first, lat, lon = rf_main._read_flat(files[0])
    counts = np.isfinite(first).all(axis=1).astype(np.uint16)
    for i, path in enumerate(files[1:], start=1):
        matrix, lat_i, lon_i = rf_main._read_flat(path)
        if not np.array_equal(lat_i, lat) or not np.array_equal(lon_i, lon):
            raise ValueError(f"Grid mismatch in {path}")
        counts += np.isfinite(matrix).all(axis=1)
        if (i + 1) % 40 == 0 or i + 1 == len(files):
            print(f"Counted {i + 1}/{len(files)} months", flush=True)

    eligible = np.flatnonzero(counts >= rf_main.MIN_MONTHS)
    n_lon = lon.size
    lat_idx, lon_idx = np.divmod(eligible, n_lon)
    table = pd.DataFrame(
        {
            "flat_index": eligible,
            "lat": lat[lat_idx].astype(float),
            "lon": lon[lon_idx].astype(float),
            "n_months": counts[eligible].astype(int),
        }
    )
    table["lat_stratum"] = pd.cut(table["lat"], LATITUDE_BREAKS, include_lowest=True)
    table["month_stratum"] = pd.cut(table["n_months"], MONTH_BREAKS, include_lowest=True)

    rng = np.random.default_rng(SAMPLE_SEED)
    selected = []
    for _, group in table.groupby(["lat_stratum", "month_stratum"], observed=True):
        n_take = min(SAMPLE_PER_STRATUM, len(group))
        selected.extend(rng.choice(group.index.to_numpy(), n_take, replace=False))
    sample = table.loc[sorted(selected)].reset_index(drop=True)

    cube = np.full((len(files), len(sample), len(rf_main.ALL_VARS)), np.nan, dtype=np.float32)
    flat = sample["flat_index"].to_numpy()
    for i, path in enumerate(files):
        matrix, _, _ = rf_main._read_flat(path)
        cube[i] = matrix[flat]
        if (i + 1) % 40 == 0 or i + 1 == len(files):
            print(f"Loaded {i + 1}/{len(files)} months", flush=True)
    dates = np.array([rf_main._file_date(path) for path in files])
    complete = np.isfinite(cube).all(axis=2)
    sample["n_tuning_months"] = complete[dates < rf_main.HOLDOUT_START].sum(axis=0)
    return sample, cube, dates


def evaluate_pixel(series: np.ndarray, dates: np.ndarray) -> dict[str, float]:
    complete = np.isfinite(series).all(axis=1)
    matrix = series[complete].astype(float)
    valid_dates = dates[complete]
    reference = valid_dates < rf_main.HOLDOUT_START
    standardized = rf_main.standardize_from_reference_period(matrix, valid_dates, reference)
    use = reference & np.isfinite(standardized).all(axis=1)
    matrix = standardized[use]
    if len(matrix) < rf_main.MIN_TRAIN_MONTHS:
        return {label: np.nan for label, _ in CONFIGS}
    y = matrix[:, 0].astype(float)
    x = matrix[:, 1:].astype(float)
    scores = {}
    for label, params in CONFIGS:
        model = RandomForestRegressor(
            n_estimators=TUNING_TREES,
            bootstrap=True,
            oob_score=True,
            random_state=rf_main.RANDOM_STATE,
            n_jobs=1,
            **params,
        )
        model.fit(x, y)
        scores[label] = float(model.oob_score_)
    return scores


if __name__ == "__main__":
    started = time.time()
    sample, cube, dates = load_stratified_sample()
    print(
        f"Evaluating {len(CONFIGS)} configurations on {len(sample)} pixels "
        f"using only dates before {rf_main.HOLDOUT_START} ...",
        flush=True,
    )
    rows = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(evaluate_pixel)(cube[:, i, :], dates) for i in range(cube.shape[1])
    )
    scores = pd.DataFrame(rows)

    summary_rows = []
    for label, params in CONFIGS:
        values = scores[label].dropna()
        summary_rows.append(
            {
                "configuration": label,
                "parameters": repr(params),
                "n_pixels": int(len(values)),
                "median_OOB_R2": float(values.median()),
                "mean_OOB_R2": float(values.mean()),
                "fraction_OOB_R2_above_zero": float((values > 0).mean()),
                "q25_OOB_R2": float(values.quantile(0.25)),
                "q75_OOB_R2": float(values.quantile(0.75)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("median_OOB_R2", ascending=False)
    summary["rank_by_median_OOB_R2"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["configuration"].eq(SELECTED_LABEL)
    if summary["selected"].sum() != 1:
        raise RuntimeError(f"Selected configuration {SELECTED_LABEL!r} is not unique")

    out_dir = rf_main.RF_RESULT
    sample_out = os.path.join(out_dir, "RF_tuning_sample.csv")
    summary_out = os.path.join(out_dir, "RF_tuning_summary.csv")
    sample.drop(columns=["lat_stratum", "month_stratum"]).to_csv(sample_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(summary.to_string(index=False), flush=True)
    print(f"Elapsed minutes: {(time.time() - started) / 60:.1f}", flush=True)
    print(f"Saved {sample_out}", flush=True)
    print(f"Saved {summary_out}", flush=True)
