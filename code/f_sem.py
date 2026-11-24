import os
import xarray as xr
import pandas as pd
import numpy as np

# ======================================================
# 统一相对路径。按 2019 年土地覆盖把 SEM 结果拆成森林 / 非森林两组。
# 注意：用单一 2019 年静态分类代表 2010-2023，属于已知局限（见论文 Limitations）。
# ======================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "result1")
landcover_nc = os.path.join(BASE, "data", "ccy2019.nc")
sem_csv = os.path.join(RESULT, "PLS_SEM_ETnolag.csv")
out_dir = os.path.join(RESULT, "f_sem")
os.makedirs(out_dir, exist_ok=True)

FOREST_CODES = [1, 2, 3, 4, 5]      # IGBP 森林类型

lc = xr.open_dataset(landcover_nc)["data"]
lat_grid = lc["lat"].values
lon_grid = lc["lon"].values
lc_vals = lc.values.astype(np.int16)

df = pd.read_csv(sem_csv)
latv = df["lat"].to_numpy()
lonv = df["lon"].to_numpy()


def nearest_index(coord, x):
    coord = np.asarray(coord)
    x = np.asarray(x)
    idx = np.searchsorted(coord, x)
    idx = np.clip(idx, 1, len(coord) - 1)
    left = coord[idx - 1]
    right = coord[idx]
    idx = idx - (np.abs(x - left) <= np.abs(right - x))
    return idx


ilat = nearest_index(lat_grid, latv)
ilon = nearest_index(lon_grid, lonv)
lc_at_pts = lc_vals[ilat, ilon]

is_forest = np.isin(lc_at_pts, FOREST_CODES)

df_forest = df.loc[is_forest].copy()
df_forest["lc_type"] = lc_at_pts[is_forest]
out_f = os.path.join(out_dir, "f_SEM_ETnolag.csv")
df_forest.to_csv(out_f, index=False)

df_nonforest = df.loc[~is_forest].copy()
df_nonforest["lc_type"] = lc_at_pts[~is_forest]
out_nf = os.path.join(out_dir, "nf_SEM_ETnolag.csv")
df_nonforest.to_csv(out_nf, index=False)

print(f"Saved: {out_f}, rows={len(df_forest)}")
print(f"Saved: {out_nf}, rows={len(df_nonforest)}")

# ===== 森林 vs 非森林平均路径系数（供 Fig 10 / 论文数字）=====
path_cols = ['SM Direct', 'VPD Direct', 'PRE Direct', 'ET Direct',
             'PRE-SM-VOD Indirect', 'ET-SM-VOD Indirect', 'VPD-SM-VOD Indirect', 'VPD-ET-VOD Indirect',
             'VPD-ET-SM-VOD Indirect',
             'SM Total', 'VPD Total', 'PRE Total', 'ET Total']
print("\n=== mean path coefficients (forest vs non-forest) ===")
print(f"{'path':22s} {'forest':>9s} {'non-forest':>11s}")
for c in path_cols:
    print(f"{c:22s} {df_forest[c].mean():>9.3f} {df_nonforest[c].mean():>11.3f}")
