"""Pixel-wise random-forest attribution with OOB and temporal diagnostics.

The manuscript uses the forest as a retrospective, contemporaneous attribution
model rather than as a future-forecasting model.  The full-period MDI importance map is
therefore fitted with every valid month.  Model skill is *not* estimated from the
same fitted values: two clearly labelled diagnostics are saved instead:

1. out-of-bag (OOB) R2, which measures internal association-recovery skill under
   bootstrap resampling; and
2. a fixed-calendar temporal stress period (2019-07--2023-06), which diagnoses
   non-stationarity/temporal transfer.  Calendar-month means and standard
   deviations are re-estimated from the pre-stress period only and then applied
   to both periods.  This single late-period split is reported as a stress
   diagnostic, not as evidence of general future-forecasting performance.

The regularisation settings were selected from a reproducible, pre-stress-period
stratified tuning experiment (see RandomForest_tuning.py): a leaf must contain at
least 7.5% of the available observations, two of four predictors are considered
per split, and each tree bootstraps 50% of the observations.  These settings limit
tree complexity for short, autocorrelated records while retaining nonlinear
response flexibility.
"""

from __future__ import annotations

import glob
import json
import os
import time
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")
RF_RESULT = os.path.join(RESULT, "RandomForest Importance")
os.makedirs(RF_RESULT, exist_ok=True)

TARGET = "VOD"
FEATURES = ["SM", "VPD", "PRE", "ET"]
ALL_VARS = [TARGET] + FEATURES

MIN_MONTHS = 30
HOLDOUT_START = np.datetime64("2019-07-01")
MIN_TRAIN_MONTHS = 24
MIN_TEST_MONTHS = 12

N_TREES = 200
STRESS_TREES = 100
MIN_SAMPLES_LEAF = 0.075
MAX_FEATURES = 0.50
MAX_SAMPLES = 0.50
RANDOM_STATE = 42

N_JOBS = int(os.environ.get("RF_N_JOBS", min(14, os.cpu_count() or 1)))
CHUNK_SIZE = 250


def _file_date(path: str) -> np.datetime64:
    name = os.path.basename(path)
    date = datetime.strptime(name.split("_")[0] + "-" + name.split("_")[1], "%Y-%m")
    return np.datetime64(date.date())


def _read_flat(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one month as (grid_cell, variable), preserving the common grid."""
    with xr.open_dataset(path) as ds:
        missing = [name for name in ALL_VARS if name not in ds]
        if missing:
            raise ValueError(f"{path} is missing variables: {missing}")
        lat = ds["lat"].values.astype(np.float32, copy=True)
        lon = ds["lon"].values.astype(np.float32, copy=True)
        matrix = np.column_stack(
            [ds[name].values.astype(np.float32, copy=False).reshape(-1) for name in ALL_VARS]
        )
    return matrix, lat, lon


def load_pixel_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only grid cells with at least MIN_MONTHS complete observations.

    A two-pass reader keeps peak memory below 1 GB on the 16-GB reference
    workstation and avoids expanding the complete 508 x 1388 x 157 grid into
    a long in-memory table.
    """
    files = sorted(glob.glob(os.path.join(DATA, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {DATA}")

    dates = np.array([_file_date(path) for path in files])
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("Monthly input files are not in strictly increasing order")
    months = dates.astype("datetime64[M]")
    expected = np.arange(months[0], months[-1] + np.timedelta64(1, "M"), dtype="datetime64[M]")
    if not np.array_equal(months, expected):
        raise ValueError("Monthly input sequence contains a gap")

    first, lat, lon = _read_flat(files[0])
    n_cells = first.shape[0]
    counts = np.isfinite(first).all(axis=1).astype(np.uint16)

    print(f"Pass 1/2: counting complete months in {len(files)} files ...", flush=True)
    for i, path in enumerate(files[1:], start=1):
        matrix, lat_i, lon_i = _read_flat(path)
        if matrix.shape[0] != n_cells or not np.array_equal(lat_i, lat) or not np.array_equal(lon_i, lon):
            raise ValueError(f"Grid mismatch in {path}")
        counts += np.isfinite(matrix).all(axis=1)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"  counted {i + 1}/{len(files)}", flush=True)

    flat_index = np.flatnonzero(counts >= MIN_MONTHS)
    # Pixel-major layout makes each worker's time series contiguous on disk/RAM.
    cube = np.full((flat_index.size, len(files), len(ALL_VARS)), np.nan, dtype=np.float32)

    print(f"Pass 2/2: loading {flat_index.size:,} eligible pixels ...", flush=True)
    for i, path in enumerate(files):
        matrix, _, _ = _read_flat(path)
        cube[:, i, :] = matrix[flat_index]
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"  loaded {i + 1}/{len(files)}", flush=True)

    n_lon = lon.size
    lat_index, lon_index = np.divmod(flat_index, n_lon)
    pixel_lat = lat[lat_index].astype(float)
    pixel_lon = lon[lon_index].astype(float)
    return cube, pixel_lat, pixel_lon, dates


def _new_forest(*, oob_score: bool = True, n_estimators: int = N_TREES) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        max_samples=MAX_SAMPLES,
        bootstrap=True,
        oob_score=oob_score,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        return np.nan
    if np.var(y_true) == 0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 3 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    return float(pearsonr(y_true, y_pred).statistic)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def standardize_from_reference_period(
    matrix: np.ndarray,
    dates: np.ndarray,
    reference_mask: np.ndarray,
    min_reference_per_calendar_month: int = 3,
) -> np.ndarray:
    """Recompute calendar-month anomalies using the reference period only.

    ``matrix`` may already contain full-period z scores.  A second z transform
    within each calendar month is algebraically identical to standardising the
    unavailable raw values with that reference period's mean and standard
    deviation, because the first transform is affine within each month.
    """
    matrix = np.asarray(matrix, dtype=float)
    dates = np.asarray(dates)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    transformed = np.full_like(matrix, np.nan, dtype=float)
    calendar_month = dates.astype("datetime64[M]").astype(np.int64) % 12

    for month in range(12):
        reference = reference_mask & (calendar_month == month)
        apply = calendar_month == month
        if reference.sum() < min_reference_per_calendar_month:
            continue
        mean = matrix[reference].mean(axis=0)
        std = matrix[reference].std(axis=0, ddof=0)
        if not np.isfinite(std).all() or np.any(std <= 0):
            continue
        transformed[apply] = (matrix[apply] - mean) / std
    return transformed


def rf_one(lat: float, lon: float, series: np.ndarray, dates: np.ndarray) -> list[object]:
    complete = np.isfinite(series).all(axis=1)
    matrix = series[complete].astype(float, copy=False)
    y = matrix[:, 0]
    x = matrix[:, 1:]
    valid_dates = dates[complete]
    n_months = y.size

    final_model = _new_forest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        final_model.fit(x, y)

    mdi = final_model.feature_importances_.astype(float)
    oob_pred = np.asarray(final_model.oob_prediction_, dtype=float)
    oob_r2 = _safe_r2(y, oob_pred)
    oob_rmse = _rmse(y, oob_pred)
    oob_r = _safe_pearson(y, oob_pred)

    order = np.sort(mdi)
    importance_margin = float(order[-1] - order[-2])
    dominant_full_idx = int(np.argmax(mdi))

    train = valid_dates < HOLDOUT_START
    test = ~train
    chrono_r2 = np.nan
    chrono_rmse = np.nan
    chrono_r = np.nan
    early_mdi = np.full(len(FEATURES), np.nan, dtype=float)
    dominant_early = "NA"
    rank_stable = np.nan
    n_train_stress = 0
    n_test_stress = 0

    stress_matrix = standardize_from_reference_period(matrix, valid_dates, train)
    stress_complete = np.isfinite(stress_matrix).all(axis=1)
    train_stress = train & stress_complete
    test_stress = test & stress_complete
    n_train_stress = int(train_stress.sum())
    n_test_stress = int(test_stress.sum())

    if n_train_stress >= MIN_TRAIN_MONTHS and n_test_stress >= MIN_TEST_MONTHS:
        y_stress = stress_matrix[:, 0]
        x_stress = stress_matrix[:, 1:]
        early_model = _new_forest(oob_score=False, n_estimators=STRESS_TREES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            early_model.fit(x_stress[train_stress], y_stress[train_stress])
        pred = early_model.predict(x_stress[test_stress])
        chrono_r2 = _safe_r2(y_stress[test_stress], pred)
        chrono_rmse = _rmse(y_stress[test_stress], pred)
        chrono_r = _safe_pearson(y_stress[test_stress], pred)
        early_mdi = early_model.feature_importances_.astype(float)
        dominant_early_idx = int(np.argmax(early_mdi))
        dominant_early = FEATURES[dominant_early_idx]
        rank_stable = float(dominant_early_idx == dominant_full_idx)

    return [
        lat,
        lon,
        int(n_months),
        *mdi.tolist(),
        oob_r2,
        oob_rmse,
        oob_r,
        chrono_r2,
        chrono_rmse,
        chrono_r,
        n_train_stress,
        n_test_stress,
        *early_mdi.tolist(),
        FEATURES[dominant_full_idx],
        dominant_early,
        rank_stable,
        importance_margin,
    ]


def rf_chunk(indices: np.ndarray, cube: np.ndarray, lats: np.ndarray, lons: np.ndarray, dates: np.ndarray):
    return [rf_one(lats[j], lons[j], cube[j], dates) for j in indices]


def _summary(df: pd.DataFrame, elapsed_minutes: float) -> dict[str, object]:
    oob = df["OOB_R2"].dropna()
    chrono = df["Temporal_stress_R2"].dropna()
    stable = df["Dominant_agreement_early_full"].dropna()
    return {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "elapsed_minutes": elapsed_minutes,
        "n_pixels": int(len(df)),
        "configuration": {
            "n_trees": N_TREES,
            "temporal_stress_n_trees": STRESS_TREES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "max_samples": MAX_SAMPLES,
            "random_state": RANDOM_STATE,
            "minimum_complete_months": MIN_MONTHS,
            "temporal_stress_start": str(HOLDOUT_START),
            "minimum_train_months": MIN_TRAIN_MONTHS,
            "minimum_test_months": MIN_TEST_MONTHS,
        },
        "oob_r2": {
            "n": int(oob.size),
            "median": float(oob.median()),
            "mean": float(oob.mean()),
            "fraction_above_zero": float((oob > 0).mean()),
            "q25": float(oob.quantile(0.25)),
            "q75": float(oob.quantile(0.75)),
        },
        "temporal_stress_r2": {
            "n": int(chrono.size),
            "median": float(chrono.median()),
            "mean": float(chrono.mean()),
            "fraction_above_zero": float((chrono > 0).mean()),
            "q25": float(chrono.quantile(0.25)),
            "q75": float(chrono.quantile(0.75)),
        },
        "early_full_dominant_rank_agreement": float(stable.mean()),
        "median_MDI_full_top_margin": float(df["MDI_full_top_margin"].median()),
    }


if __name__ == "__main__":
    started = time.time()
    cube, lats, lons, dates = load_pixel_cube()

    indices = np.arange(cube.shape[0])
    chunks = [indices[i : i + CHUNK_SIZE] for i in range(0, indices.size, CHUNK_SIZE)]
    print(
        f"Fitting {indices.size:,} pixels with {N_TREES} trees, "
        f"min_samples_leaf={MIN_SAMPLES_LEAF}, max_features={MAX_FEATURES}, "
        f"max_samples={MAX_SAMPLES} ...",
        flush=True,
    )
    nested = Parallel(n_jobs=N_JOBS, verbose=10, max_nbytes="10M", mmap_mode="r")(
        delayed(rf_chunk)(chunk, cube, lats, lons, dates) for chunk in chunks
    )
    rows = [row for block in nested for row in block]

    columns = [
        "lat",
        "lon",
        "n_months",
        "SM Importance",
        "VPD Importance",
        "PRE Importance",
        "ET Importance",
        "OOB_R2",
        "OOB_RMSE",
        "OOB_Pearson_r",
        "Temporal_stress_R2",
        "Temporal_stress_RMSE",
        "Temporal_stress_Pearson_r",
        "n_early_period",
        "n_stress_period",
        "SM Importance early",
        "VPD Importance early",
        "PRE Importance early",
        "ET Importance early",
        "Dominant_full",
        "Dominant_early",
        "Dominant_agreement_early_full",
        "MDI_full_top_margin",
    ]
    output = pd.DataFrame(rows, columns=columns).sort_values(["lat", "lon"]).reset_index(drop=True)

    mdi_columns = [f"{name} Importance" for name in FEATURES]
    mdi_sum = output[mdi_columns].sum(axis=1)
    if not np.allclose(mdi_sum, 1.0, atol=1e-6):
        raise RuntimeError("MDI importances do not sum to one")
    if output[["lat", "lon"]].duplicated().any():
        raise RuntimeError("Duplicate pixel coordinates in RF output")

    out_csv = os.path.join(RESULT, "Worlds_importancenolag.csv")
    output.to_csv(out_csv, index=False)
    elapsed = (time.time() - started) / 60.0

    summary = _summary(output, elapsed)
    summary_path = os.path.join(RF_RESULT, "RF_model_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {out_csv} ({len(output):,} rows)", flush=True)
    print(f"Saved {summary_path}", flush=True)
