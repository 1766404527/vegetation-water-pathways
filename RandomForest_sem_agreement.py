"""Quantify agreement between RF leading factors and SEM direct/total effects."""

import os

import numpy as np
import pandas as pd

import RandomForest as rf_main
from RandomForest_regional_validation import REGIONS


if __name__ == "__main__":
    rf = pd.read_csv(os.path.join(rf_main.RESULT, "Worlds_importancenolag.csv"))
    sem = pd.read_csv(os.path.join(rf_main.RESULT, "PLS_SEM_ETnolag.csv"))
    data = rf.merge(sem, on=["lat", "lon"], how="inner", validate="one_to_one")
    if len(data) != len(rf) or len(data) != len(sem):
        raise RuntimeError(
            "RF/SEM coordinate mismatch: "
            f"RF={len(rf)}, SEM={len(sem)}, merged={len(data)}"
        )
    factors = np.asarray(rf_main.FEATURES)
    rf_leader = factors[data[[f"{f} Importance" for f in factors]].to_numpy().argmax(axis=1)]
    direct_leader = factors[
        np.abs(data[[f"{f} Direct" for f in factors]].to_numpy()).argmax(axis=1)
    ]
    total_leader = factors[
        np.abs(data[[f"{f} Total" for f in factors]].to_numpy()).argmax(axis=1)
    ]
    # SMOS-IC cells are equal area, so global and regional summaries use equal
    # cell weights rather than applying a second latitude correction.
    weights = np.ones(len(data), dtype=float)

    masks = {
        "all_analyzable": np.ones(len(data), dtype=bool),
        "positive_OOB": data["OOB_R2"].to_numpy() > 0,
        "positive_OOB_and_early_full_agreement": (
            (data["OOB_R2"].to_numpy() > 0)
            & (data["Dominant_agreement_early_full"].to_numpy() == 1)
        ),
    }
    rows = []
    for name, mask in masks.items():
        rows.append(
            {
                "subset": name,
                "n_pixels": int(mask.sum()),
                "RF_SEM_direct_pixel_agreement": float((rf_leader[mask] == direct_leader[mask]).mean()),
                "RF_SEM_direct_area_weighted_agreement": float(
                    np.sum(weights[mask] * (rf_leader[mask] == direct_leader[mask])) / weights[mask].sum()
                ),
                "RF_SEM_total_area_weighted_agreement": float(
                    np.sum(weights[mask] * (rf_leader[mask] == total_leader[mask])) / weights[mask].sum()
                ),
            }
        )

    supported = masks["positive_OOB"]
    for letter, region_name, lat0, lat1, lon0, lon1 in REGIONS:
        mask = (
            supported
            & data["lat"].between(lat0, lat1).to_numpy()
            & data["lon"].between(lon0, lon1).to_numpy()
        )
        rows.append(
            {
                "subset": f"region_{letter}_{region_name}",
                "n_pixels": int(mask.sum()),
                "RF_SEM_direct_pixel_agreement": float((rf_leader[mask] == direct_leader[mask]).mean()),
                "RF_SEM_direct_area_weighted_agreement": float(
                    np.sum(weights[mask] * (rf_leader[mask] == direct_leader[mask])) / weights[mask].sum()
                ),
                "RF_SEM_total_area_weighted_agreement": float(
                    np.sum(weights[mask] * (rf_leader[mask] == total_leader[mask])) / weights[mask].sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(rf_main.RF_RESULT, "RF_SEM_agreement_summary.csv")
    summary.to_csv(summary_path, index=False)

    confusion = pd.crosstab(
        pd.Series(rf_leader[supported], name="RF_leader"),
        pd.Series(direct_leader[supported], name="SEM_direct_leader"),
        normalize="index",
    )
    confusion.to_csv(os.path.join(rf_main.RF_RESULT, "RF_SEM_direct_confusion_positive_OOB.csv"))
    print(summary.to_string(index=False))
    print(f"Saved {summary_path}")
