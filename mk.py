import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import pymannkendall as mk
from joblib import Parallel, delayed

# ======================================================
# 统一相对路径
# ======================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1", "mk")
os.makedirs(RESULT, exist_ok=True)

VARIABLES = ['VOD', 'SM', 'PRE', 'VPD', 'ET']
MIN_LEN = 10          # 单像元做趋势检验所需的最少有效月数
N_JOBS = 14
CHUNK = 800

# 注意：仓库提供的是逐月 z-score 标准化（去季节）异常序列，因此趋势分析
# 在异常序列上进行。采用 Hamed-Rao 自相关校正的 Mann-Kendall 检验，slope
# 取 Theil-Sen 斜率。下方会先移除缺测月，因此无缺测记录的 slope 单位为
# z-score/月；有缺测记录的 slope 是每个连续有效观测序号的变化量。


def load_all():
    files = sorted(glob.glob(os.path.join(DATA, "*.nc")))
    dss = []
    for fp in files:
        fn = os.path.basename(fp)
        date = datetime.strptime(fn.split('_')[0] + '-' + fn.split('_')[1], '%Y-%m')
        dss.append(xr.open_dataset(fp).expand_dims({'time': [date]}))
    return xr.concat(dss, dim='time').sortby('time')


def mk_chunk(chunk):
    """chunk: list of (lat, lon, series). 返回结果行列表。"""
    rows = []
    for lat, lon, series in chunk:
        try:
            r = mk.hamed_rao_modification_test(series)
            rows.append([lat, lon, r.trend != 'no trend', r.slope, r.p, r.trend])
        except Exception:
            continue
    return rows


def run_variable(ds_all, var_name):
    da = ds_all[var_name].transpose('lat', 'lon', 'time')
    A = da.values                      # (nlat, nlon, ntime)
    lats = da['lat'].values.astype(float)
    lons = da['lon'].values.astype(float)

    valid_count = np.isfinite(A).sum(axis=2)
    ii, jj = np.where(valid_count >= MIN_LEN)

    # 提取每个有效像元的非缺测时间序列（已按时间升序）
    tasks = []
    for i, j in zip(ii, jj):
        s = A[i, j, :]
        s = s[np.isfinite(s)]
        tasks.append((lats[i], lons[j], s))

    chunks = [tasks[k:k + CHUNK] for k in range(0, len(tasks), CHUNK)]
    results = Parallel(n_jobs=N_JOBS)(delayed(mk_chunk)(c) for c in chunks)
    rows = [r for sub in results for r in sub]

    df = pd.DataFrame(rows, columns=['Latitude', 'Longitude', 'Trend', 'Slope', 'P', 'TrendDir'])
    out = os.path.join(RESULT, f"{var_name.lower()}_mk.csv")
    df.to_csv(out, index=False)
    print(f"[{var_name}] pixels={len(df)}  significant={int(df['Trend'].sum())}  -> {out}")


if __name__ == "__main__":
    print("Loading data ...")
    ds_all = load_all()
    for v in VARIABLES:
        run_variable(ds_all, v)
    print("Done.")
