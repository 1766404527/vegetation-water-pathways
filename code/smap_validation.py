# -*- coding: utf-8 -*-
"""
B5 独立产品交叉验证 —— 生成相关一致性 CSV。
在 SMOS(svvpesv)与 SMAP(smap)重叠期 2015-06 ~ 2023-06（97 个月）上，
逐像元、在两产品共同有效月（>=30）上计算 Pearson 相关：
  - 两套 VOD（SMOS_VOD / SMAP_VOD）分别与 5 个驱动的相关
    驱动 = SMOS_SM、SMAP_SM、PRE、VPD、ET
其中 SMOS_SM / SMAP_SM 两组同时构成"耦合检验"（同源 vs 跨产品配对）。
输出 result1/smap_validation/smap_validation.csv（每对相关一列，x=SMOS_VOD、y=SMAP_VOD）。
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVV = os.path.join(BASE, "data", "svvpesv")
SMAP = os.path.join(BASE, "data", "smap")
OUT = os.path.join(BASE, "result1", "smap_validation")
os.makedirs(OUT, exist_ok=True)

MIN_MONTHS = 30

# 重叠期 2015-06 ~ 2023-06
MONTHS = [(y, m) for y in range(2015, 2024) for m in range(1, 13)
          if (2015, 6) <= (y, m) <= (2023, 6)]

# 7 个代表区（用于可选的区域统计）
REGIONS = [
    ('A', (30, 48), (-123, -105)), ('B', (45, 54), (56, 83)),
    ('C', (-38, -20), (140, 153)), ('D', (-35, -20), (25, 35)),
    ('E', (-18, -9), (16, 23)), ('F', (-5, 16), (98, 120)),
    ('G', (-6, 7), (-78, -59)),
]


def load_smos():
    frames = []
    for y, m in MONTHS:
        fp = os.path.join(SVV, f"{y:04d}_{m:02d}_svvpe_zv.nc")
        if not os.path.exists(fp):
            print("missing SMOS", fp); continue
        df = xr.open_dataset(fp).to_dataframe().reset_index()
        df = df.dropna(subset=['VOD', 'SM', 'VPD', 'ET', 'PRE'])
        df = df.rename(columns={'VOD': 'SMOS_VOD', 'SM': 'SMOS_SM'})
        df['ym'] = y * 100 + m
        frames.append(df[['ym', 'lat', 'lon', 'SMOS_VOD', 'SMOS_SM', 'VPD', 'ET', 'PRE']])
    return pd.concat(frames, ignore_index=True)


def load_smap():
    frames = []
    for y, m in MONTHS:
        fp = os.path.join(SMAP, f"{y:04d}_{m:02d}_smap_zv.nc")
        if not os.path.exists(fp):
            print("missing SMAP", fp); continue
        df = xr.open_dataset(fp).to_dataframe().reset_index()
        df = df.dropna(subset=['SM', 'VOD'])
        df = df.rename(columns={'VOD': 'SMAP_VOD', 'SM': 'SMAP_SM'})
        df['ym'] = y * 100 + m
        frames.append(df[['ym', 'lat', 'lon', 'SMAP_VOD', 'SMAP_SM']])
    return pd.concat(frames, ignore_index=True)


def region_of(lat, lon):
    for name, (la0, la1), (lo0, lo1) in REGIONS:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return ''


if __name__ == "__main__":
    print("loading SMOS ...")
    smos = load_smos()
    print("loading SMAP ...")
    smap = load_smap()
    print(f"SMOS rows={len(smos)}  SMAP rows={len(smap)}")

    merged = pd.merge(smos, smap, on=['ym', 'lat', 'lon'], how='inner')
    print("merged (common pixel-months):", len(merged))

    order = ['SMOS_VOD', 'SMAP_VOD', 'SMOS_SM', 'SMAP_SM', 'VPD', 'PRE', 'ET']
    rows = []
    for (lat, lon), sub in merged.groupby(['lat', 'lon'], sort=True):
        if len(sub) < MIN_MONTHS:
            continue
        M = sub[order].to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            C = np.corrcoef(M, rowvar=False)
        rows.append([
            lat, lon, region_of(lat, lon), len(sub),
            C[0, 2], C[1, 2],   # SMOS_SM 子图: x=SMOSVOD_SMOSSM, y=SMAPVOD_SMOSSM
            C[0, 3], C[1, 3],   # SMAP_SM 子图: x=SMOSVOD_SMAPSM, y=SMAPVOD_SMAPSM
            C[0, 5], C[1, 5],   # PRE 子图
            C[0, 4], C[1, 4],   # VPD 子图
            C[0, 6], C[1, 6],   # ET 子图
            C[2, 3],            # SMOSSM_SMAPSM: SM 产品互相关（SM 互验图）
            C[0, 1],            # VODVOD: VOD 产品互相关（仅供文字说明 DCA VOD 不一致）
        ])

    cols = ['lat', 'lon', 'region', 'n_months',
            'SMOSVOD_SMOSSM', 'SMAPVOD_SMOSSM',
            'SMOSVOD_SMAPSM', 'SMAPVOD_SMAPSM',
            'SMOSVOD_PRE', 'SMAPVOD_PRE',
            'SMOSVOD_VPD', 'SMAPVOD_VPD',
            'SMOSVOD_ET', 'SMAPVOD_ET',
            'SMOSSM_SMAPSM', 'VODVOD']
    out_df = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(OUT, "smap_validation.csv")
    out_df.to_csv(out_path, index=False)
    print(f"saved {out_path}  pixels={len(out_df)}")

    # 速览：每个子图 x-y 的空间相关 r 与同号比例
    pairs = [('SMOSVOD_SMOSSM', 'SMAPVOD_SMOSSM', 'VOD~SMOS_SM (coupling)'),
             ('SMOSVOD_SMAPSM', 'SMAPVOD_SMAPSM', 'VOD~SMAP_SM (coupling)'),
             ('SMOSVOD_PRE', 'SMAPVOD_PRE', 'VOD~PRE'),
             ('SMOSVOD_VPD', 'SMAPVOD_VPD', 'VOD~VPD'),
             ('SMOSVOD_ET', 'SMAPVOD_ET', 'VOD~ET')]
    print("\npanel                    spatial_r  same_sign%")
    for xc, yc, lab in pairs:
        a = out_df[xc].to_numpy(); b = out_df[yc].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        ss = np.mean(np.sign(a[ok]) == np.sign(b[ok])) * 100
        print(f"{lab:26s} {r:6.3f}    {ss:5.1f}")
