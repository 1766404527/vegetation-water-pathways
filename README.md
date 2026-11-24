# Hydroclimatic pathways of global vegetation water content

This repository provides Python code and supporting numerical results for:

“How indirect pathways shape the attribution of global vegetation water
content anomalies to soil moisture and atmospheric dryness.”

The study analyzes monthly standardized anomalies of L-band vegetation
optical depth (VOD), soil moisture (SM), precipitation (PRE), vapor pressure
deficit (VPD), and evapotranspiration (ET) from June 2010 to June 2023.

The workflow includes data preprocessing, correlation and trend analyses,
pixel-wise random forests, structural path analysis, forest–non-forest
comparisons, independent SMAP comparisons, and figure generation.

Complete analysis-ready datasets are distributed separately. See
data/README.md for download instructions and the required directory structure.

## Folder structure

```text
AFM/
├── Manuscript_AFM.docx
├── README.md
├── requirements.txt
├── code/        # analysis and figure scripts
├── data/        # analysis-ready inputs and optional preprocessing scripts
├── result1/     # numerical results and analysis figures
└── figures/     # Figures 1-11 (PNG); PDF/SVG for Figures 1-2; editable SVG for Figure 11
```

The directory name `result1` is used consistently by the supplied scripts as the output location.

## Environment

The analysis was run with Python 3.12.2 on Windows. Install the pinned packages from the archive root:

```powershell
"C:\Program Files\Python312\python.exe" -m pip install -r requirements.txt
```

Equivalent `python -m pip` commands can be used on other systems. Cartopy requires GEOS and PROJ. If a pip wheel is unavailable, Cartopy 0.24.1 can be installed from conda-forge before installing the remaining packages.

The RF and path-analysis scripts accept optional worker-count variables:

```powershell
$env:RF_N_JOBS = 8
$env:SEM_N_JOBS = 8
```

If these variables are unset, the scripts use at most 14 workers.

`code/build_figure11.py` updates the editable Figure 11 SVG directly from the
forest and non-forest path tables. Both panels include four direct and five
indirect effects, with VPD->ET->SM->VOD drawn separately from VPD->ET->VOD.
An editable copy is also saved as `figures/Figure_11.svg`.
SVG generation uses the Python dependencies
listed above. Direct PNG export uses Microsoft Edge in headless mode; use
`python code/build_figure11.py --skip-png` when only the editable SVG is
required.

## Data

`data/svvpesv/` contains 157 contiguous monthly NetCDF files from June 2010 through June 2023 on a common 508 x 1388 grid. Each file contains calendar-month standardized anomalies of VOD, SM, VPD, PRE, and ET. A vegetation pixel enters the common multivariate analysis when at least 30 months contain finite values for all five variables. This definition yields 99,758 analyzed pixels.

`data/smap/` contains 97 monthly files for the independent SMAP comparison from June 2015 through June 2023. `data/ccy2019.nc` contains the 2019 MODIS MCD12Q1 land-cover mask used for the forest/non-forest comparison.

The analysis-ready inputs needed by the manuscript analyses are included. The preprocessing scripts under `data/` can be used when the corresponding raw product archives are obtained separately and their local paths are configured.

| Variable | Product |
|---|---|
| VOD and SM | SMOS-IC V2 |
| PRE | PERSIANN-CDR |
| VPD | ERA5-Land temperature and dew-point temperature |
| ET | GLEAM v4.1a |
| Land cover | MODIS MCD12Q1 Collection 6.1, IGBP scheme, 2019 |
| Independent SM and VOD comparison | SMAP L3 SPL3SMP Version 8 |

## Analysis sequence

Run the commands from the `AFM` directory. The sequence below makes file dependencies explicit.

| Step | Command | Main output |
|---:|---|---|
| 1 | `python code/mk.py` | trend tables in `result1/mk/` |
| 2 | `python code/mk_png.py` | trend maps |
| 3 | `python code/pearsonr.py` | `result1/Worlds_nolag.csv` |
| 4 | `python code/Pearsonr_png.py` | VOD-driver correlation maps |
| 5 | `python code/smap_validation.py` | SMAP comparison table |
| 6 | `python code/smap_validation_png.py` | SMAP comparison figure |
| 7 | `python code/RandomForest_tuning.py` | RF tuning summaries |
| 8 | `python code/RandomForest.py` | pixel RF results and model summary |
| 9 | `python code/combine.py` | merged RF and correlation table |
| 10 | `python code/combine_dominant_png.py` | OOB-screened RF dominance map |
| 11 | `python code/RandomForest_regional_validation.py` | regional OOB drop-column results |
| 12 | `python code/RandomForest_regional_diagnostics.py` | regional RF diagnostic figure |
| 13 | `python code/Pearsonr_box.py` | spatial-block regional correlation point estimates |
| 14 | `python code/sem.py` | pixel path-analysis table |
| 15 | `python code/sem_composite_png.py` | direct and total path-coefficient figure |
| 16 | `python code/sem_dominant_png.py` | path dominance maps and summaries |
| 17 | `python code/sem_dominant_composite.py` | path-dominance composite figure |
| 18 | `python code/RandomForest_sem_agreement.py` | RF-path agreement diagnostics |
| 19 | `python code/f_sem.py` | forest and non-forest path subsets |
| 20 | `python code/build_figure11.py` | data-driven Figure 11 SVG and PNG |

## Statistical implementation

### Correlation and trends

Correlation, RF, and path analysis use the same complete-case definition. Pairwise Pearson coefficients are calculated on the common 99,758-pixel mask. The Mann-Kendall analysis uses each variable's available monthly record and applies the Hamed-Rao correction for serial correlation. Theil-Sen slopes are reported in standardized-anomaly units per month for gap-free records.

### Random forest attribution

Parameter tuning and skill screening are separate steps. `RandomForest_tuning.py` selects one global regularized configuration on the stratified pre-stress sample; after the final full-period models are fitted, `OOB_R2 > 0` is used as the minimum internal skill screen for assigning RF dominant factors.

The pixel RF is used for retrospective contemporaneous attribution. The full-period model contains 200 trees, a minimum leaf size of 7.5% of complete monthly observations, two candidate predictors per split, a 50% bootstrap sample fraction, and random seed 42. Mean decrease in impurity (MDI) ranks predictor strength. Out-of-bag R2 provides an internal association-recovery diagnostic, and pixels with OOB R2 less than or equal to zero are classified as low OOB skill in the dominance map.

A fixed temporal stress test trains a 100-tree model on observations before July 2019 and evaluates July 2019-June 2023. Calendar-month means and standard deviations are estimated from the earlier period and applied to both periods. This diagnostic evaluates temporal transfer under non-stationarity and is separate from the full-period attribution map.

For seven geographic regions, predictor support is also assessed by the loss in OOB R2 after dropping one predictor at a time. Confidence intervals use 1,000 resamples of 2-degree spatial-block means. The SMOS-IC analysis grid uses equal-area cells, so global area summaries use equal cell weights.

The main RF diagnostics are:

| Quantity | Value |
|---|---:|
| Analyzed pixels | 99,758 |
| Median full-period OOB R2 | 0.1148 |
| Pixels with OOB R2 > 0 | 84.12% |
| OOB-supported analyzed area | 84.12% |
| Median temporal-stress R2 | -0.1906 |
| Temporal-stress pixels with R2 > 0 | 28.62% |
| Early/full leading-factor agreement | 71.01% |

MDI is unsigned. The positive or negative sign displayed for a leading factor is taken from its full-period marginal Pearson correlation with VOD and is not a conditional RF partial effect. OOB skill is not a spatially or temporally independent validation score.

### Path analysis

The standardized OLS equations are:

```text
VOD ~ SM + VPD + PRE + ET
SM  ~ PRE + ET + VPD
ET  ~ VPD
```

The analysis includes the indirect paths PRE->SM->VOD, ET->SM->VOD, VPD->SM->VOD, VPD->ET->VOD, and VPD->ET->SM->VOD. Indirect and total effects use 300 nonparametric monthly bootstrap resamples for confidence intervals and p-values. RF-path agreement is a cross-method concordance analysis based on common inputs, rather than independent validation or causal proof.

## Main files

- `result1/Worlds_nolag.csv`: pixel correlation coefficients.
- `result1/Worlds_importancenolag.csv`: pixel RF importance and diagnostics.
- `result1/PLS_SEM_ETnolag.csv`: direct, indirect, and total path effects.
- `result1/RandomForest Importance/`: RF tuning, regional validation, dominance, and RF-path agreement results.
- `result1/sem_s/`: path maps and dominance summaries.
- `result1/smap_validation/`: independent SMAP comparison.
- `figures/`: the eleven PNG manuscript figures, numbered in manuscript order.

Figures 1 and 2 also have font-embedded PDF and editable SVG versions.
Run `python code/build_framework_figures.py` to regenerate their PNG, PDF,
and SVG files. Figure 1 shows three complementary analyses of shared
preprocessed data. Figure 2 uses the same effect-decomposition layout as
Figure 11: four direct effects and five complete indirect effects, including
the separate VPD->ET->VOD and serial VPD->ET->SM->VOD pathways. Dashed shafts
end in complete, solid arrowheads, matching the direct-effect arrows.

This repository provides the Python code associated with the study. Input datasets and precomputed results are not included. Users need to obtain the required datasets and configure the input paths before running the scripts.
