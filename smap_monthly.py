# -*- coding: utf-8 -*-
"""
SMAP L3 SM_P (SPL3SMP, EASE-Grid 2.0 36 km) 月均处理 —— 第一阶段
读取 E:\\SMAP_L3_SM_P 下 2015-06 ~ 2023-06 的逐日文件，
对 soil_moisture (= DCA) 与 vegetation_opacity (= DCA L-VOD) 计算逐月均值，
AM+PM 两次过境取平均，仅用缺测掩膜（剔除 -9999，保留植被密集区以与 SMOS-IC 可比）。
结果（EASE2 网格，406x964）存为中间文件 smap_monthly_ease2.npz，供第二阶段重采样/求异常。
"""
import os, glob, time
import numpy as np
import h5py
from joblib import Parallel, delayed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SMAP_DIR：原始 SMAP L3 逐日 HDF5 文件夹（未随仓库提供）—— 请改为本机/外置盘实际路径
SMAP_DIR = r"E:\SMAP_L3_SM_P"
OUT_NPZ = os.path.join(BASE, "data", "smap", "smap_monthly_ease2.npz")  # 仓库内置输出
FILL = -9999.0
GROUPS = ["Soil_Moisture_Retrieval_Data_AM", "Soil_Moisture_Retrieval_Data_PM"]
VARS = ["soil_moisture", "vegetation_opacity"]      # = DCA SM / L-VOD
N_JOBS = 6

# 目标时段：2015-06 ~ 2023-06
MONTHS = []
for y in range(2015, 2024):
    for m in range(1, 13):
        if (y, m) < (2015, 6) or (y, m) > (2023, 6):
            continue
        MONTHS.append((y, m))


def list_files_by_month():
    files = glob.glob(os.path.join(SMAP_DIR, "SMAP_L3_SM_P_*.h5"))
    by = {}
    for fp in files:
        base = os.path.basename(fp)
        ymd = base.split("_")[4]          # YYYYMMDD
        y, m = int(ymd[:4]), int(ymd[4:6])
        by.setdefault((y, m), []).append(fp)
    return by


def read_overpass(f, group, var):
    g = f[group]
    if var not in g:
        return None
    a = g[var][:].astype("float64")
    a[a == FILL] = np.nan
    return a


def process_month(ym, files):
    """返回 (ym, sm_mean, vod_mean)；EASE2 网格 406x964。"""
    shape = (406, 964)
    sm_sum = np.zeros(shape); sm_cnt = np.zeros(shape)
    vod_sum = np.zeros(shape); vod_cnt = np.zeros(shape)
    for fp in sorted(files):
        try:
            with h5py.File(fp, "r") as f:
                # 逐变量：AM、PM 两次过境取平均
                day = {}
                for var in VARS:
                    layers = []
                    for grp in GROUPS:
                        a = read_overpass(f, grp, var)
                        if a is not None:
                            layers.append(a)
                    if layers:
                        day[var] = np.nanmean(np.stack(layers, 0), axis=0)
        except Exception as e:
            print("skip", os.path.basename(fp), e)
            continue
        if "soil_moisture" in day:
            a = day["soil_moisture"]; m = np.isfinite(a)
            sm_sum[m] += a[m]; sm_cnt[m] += 1
        if "vegetation_opacity" in day:
            a = day["vegetation_opacity"]; m = np.isfinite(a)
            vod_sum[m] += a[m]; vod_cnt[m] += 1
    sm_mean = np.where(sm_cnt > 0, sm_sum / np.maximum(sm_cnt, 1), np.nan)
    vod_mean = np.where(vod_cnt > 0, vod_sum / np.maximum(vod_cnt, 1), np.nan)
    return ym, sm_mean.astype("float32"), vod_mean.astype("float32")


def build_grid_coords():
    """从一个完整覆盖的参考文件重建 EASE2 1D 经纬度（行内纬度恒定、列内经度恒定）。"""
    ref = os.path.join(SMAP_DIR, "SMAP_L3_SM_P_20150701_R18290_001.h5")
    if not os.path.exists(ref):
        ref = sorted(glob.glob(os.path.join(SMAP_DIR, "*.h5")))[100]
    with h5py.File(ref, "r") as f:
        g = f["Soil_Moisture_Retrieval_Data_AM"]
        lat = g["latitude"][:].astype("float64"); lat[lat == FILL] = np.nan
        lon = g["longitude"][:].astype("float64"); lon[lon == FILL] = np.nan
    lat1 = np.nanmedian(lat, axis=1)   # 406, 递减
    lon1 = np.nanmedian(lon, axis=0)   # 964, 递增
    return lat1, lon1


if __name__ == "__main__":
    t0 = time.time()
    by = list_files_by_month()
    tasks = [(ym, by.get(ym, [])) for ym in MONTHS]
    nfiles = sum(len(t[1]) for t in tasks)
    print(f"months={len(tasks)}  total daily files={nfiles}")

    lat1, lon1 = build_grid_coords()
    print("grid coords built: lat", lat1.shape, "lon", lon1.shape)

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_month)(ym, files) for ym, files in tasks)
    results.sort(key=lambda r: r[0])

    years = np.array([r[0][0] for r in results])
    monthsn = np.array([r[0][1] for r in results])
    sm = np.stack([r[1] for r in results], 0)    # (n, 406, 964)
    vod = np.stack([r[2] for r in results], 0)

    np.savez_compressed(OUT_NPZ, sm=sm, vod=vod, years=years, months=monthsn,
                        lat1=lat1.astype("float32"), lon1=lon1.astype("float32"))
    valid = np.isfinite(sm).sum(axis=(1, 2))
    print(f"saved {OUT_NPZ}  shape={sm.shape}  time={(time.time()-t0)/60:.1f} min")
    print("per-month valid SM cells (first 5):", valid[:5], "... mean", int(valid.mean()))
