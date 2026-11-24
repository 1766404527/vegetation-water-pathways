# -*- coding: utf-8 -*-
"""
SMAP 月均处理 —— 第二阶段
1) 把第一阶段的 EASE2 36km 月均（SM, VOD）双线性重采样到 SMOS/SVVPESV 的 0.25° 网格；
2) 套用与 svvpesv 相同的植被掩膜（ccy2019.nc，剔除非植被类 0/13/15/16）；
3) 按 svvpes.py 的方法计算逐月 z-score 异常（逐像元、逐日历月、跨年）；
4) 逐月输出 nc 到 data/smap，命名 {YYYY}_{MM}_smap_zv.nc，含变量 SM、VOD（异常），
   网格与 svvpesv 完全一致，便于与 SMOS 直接逐像元交叉验证。
"""
import os
import numpy as np
import xarray as xr
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根 D:\Desktop\source
DATA = os.path.join(BASE, "data")
NPZ = os.path.join(DATA, "smap", "smap_monthly_ease2.npz")
REF = os.path.join(DATA, "svvpesv", "2010_06_svvpe_zv.nc")          # 目标网格参考
CCY = os.path.join(DATA, "ccy2019.nc")                              # 植被分类
OUT = os.path.join(DATA, "smap")
NONVEG = [0, 13, 15, 16]

# 1) 读 EASE2 月均
d = np.load(NPZ)
sm, vod = d["sm"], d["vod"]                  # (n, 406, 964)
years, months = d["years"], d["months"]
lat1, lon1 = d["lat1"].astype("float64"), d["lon1"].astype("float64")

# lat 递减 -> 翻成递增（interp 更稳）
if lat1[0] > lat1[-1]:
    lat1 = lat1[::-1]; sm = sm[:, ::-1, :]; vod = vod[:, ::-1, :]

times = pd.to_datetime([f"{y:04d}-{m:02d}-15" for y, m in zip(years, months)])
ease = xr.Dataset(
    {"SM": (("time", "lat", "lon"), sm), "VOD": (("time", "lat", "lon"), vod)},
    coords={"time": times, "lat": lat1, "lon": lon1},
)
print("EASE2 dataset:", dict(ease.sizes))

# 2) 目标网格 + 双线性重采样
ref = xr.open_dataset(REF)
tlat = ref["lat"].values.astype("float64")
tlon = ref["lon"].values.astype("float64")
res = ease.interp(lat=tlat, lon=tlon, method="linear")
print("resampled to target grid:", dict(res.sizes))

# 3) 植被掩膜（与 svvpesv 一致）
ccy = xr.open_dataset(CCY)
ccy_var = "data" if "data" in ccy.data_vars else list(ccy.data_vars)[0]
veg = (~ccy[ccy_var].isin(NONVEG)) & ccy[ccy_var].notnull()
try:
    veg_t = veg.interp(lat=tlat, lon=tlon, method="nearest")
except Exception:
    veg_t = veg.reindex(lat=tlat, lon=tlon, method="nearest")
res = res.where(veg_t)

# 4) 逐月 z-score 异常（逐日历月气候态）
clim_mean = res.groupby("time.month").mean("time")
clim_std = res.groupby("time.month").std("time")
clim_std = clim_std.where(clim_std > 0)
anom = res.groupby("time.month") - clim_mean
z = anom.groupby("time.month") / clim_std

# 5) 逐月写出
os.makedirs(OUT, exist_ok=True)
n = 0
for t in z.time.values:
    ds_t = z.sel(time=t)
    if "month" in ds_t.coords:
        ds_t = ds_t.drop_vars("month")
    ds_t = ds_t.drop_vars("time", errors="ignore")
    dt = pd.to_datetime(str(t))
    out_name = f"{dt.year:04d}_{dt.month:02d}_smap_zv.nc"
    ds_t[["SM", "VOD"]].to_netcdf(os.path.join(OUT, out_name))
    n += 1
print(f"wrote {n} monthly SMAP anomaly files to {OUT}")

# 简要校验
chk = z.isel(time=slice(0, None))
print("VOD anomaly std over time (should ~1):", float(chk["VOD"].std()))
print("SM  anomaly std over time (should ~1):", float(chk["SM"].std()))
print("valid pixels (a month):", int(np.isfinite(z["VOD"].isel(time=12)).sum()))
