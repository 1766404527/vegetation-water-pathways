import os
import glob
import time
import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

import RandomForest as rf_main

# ======================================================
# A1：把原"PLS-SEM"（多元直接系数 + 两两间接系数相加）重写为
#     自洽的 OLS 通径分析（path analysis）。
#     数据已逐月 z-score 标准化，OLS 偏系数即标准化通径系数；
#     所有系数来自同一套结构方程系统，Total = Direct + ΣIndirect 恒等成立。
# A2：直接路径显著性用 OLS 偏系数的解析 p 值；
#     间接/总效应显著性用非参 bootstrap（按月有放回重采样）求 95% CI 与 p。
#
# 结构方程组：
#   VOD ~ SM + VPD + PRE + ET     (4 条直接效应)
#   SM  ~ PRE + ET + VPD          (SM 作为中介)
#   ET  ~ VPD                     (ET 作为中介)
# 间接路径：
#   PRE→SM→VOD = b(SM~PRE)*b(VOD~SM)
#   ET →SM→VOD = b(SM~ET )*b(VOD~SM)
#   VPD→SM→VOD = b(SM~VPD)*b(VOD~SM)
#   VPD→ET→VOD = b(ET~VPD)*b(VOD~ET)
#   VPD→ET→SM→VOD = b(ET~VPD)*b(SM~ET)*b(VOD~SM)
# 总效应：
#   SM  = 直接
#   PRE = 直接 + PRE→SM→VOD
#   ET  = 直接 + ET →SM→VOD
#   VPD = 直接 + VPD→SM→VOD + VPD→ET→VOD + VPD→ET→SM→VOD
# ======================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")
os.makedirs(RESULT, exist_ok=True)

MIN_MONTHS = 30      # 与随机森林一致；OLS 4 元回归需足够样本量
N_BOOT = 300
N_JOBS = int(os.environ.get("SEM_N_JOBS", min(14, os.cpu_count() or 1)))
CHUNK = 200


def _ols_full(y, cols):
    """返回 beta(含常数), se, pval。cols 为不含常数项的自变量列表。"""
    n = len(y)
    Xc = np.column_stack([np.ones(n)] + cols)
    XtX = Xc.T @ Xc
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xc.T @ y)
    resid = y - Xc @ beta
    dof = max(n - Xc.shape[1], 1)
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        tval = np.where(se > 0, beta / se, np.nan)
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    yhat = Xc @ beta
    return beta, se, pval, yhat


def _ols_coef(y, cols):
    """仅返回系数（含常数），用于 bootstrap 重采样，带奇异回退。"""
    n = len(y)
    Xc = np.column_stack([np.ones(n)] + cols)
    try:
        return np.linalg.solve(Xc.T @ Xc, Xc.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Xc, y, rcond=None)[0]


def _batch_ols_coef(y, predictors):
    """Batched OLS for bootstrap arrays.

    Parameters
    ----------
    y : (bootstrap, observation)
    predictors : (bootstrap, observation, predictor)
    """
    b, n = y.shape
    design = np.concatenate([np.ones((b, n, 1)), predictors], axis=2)
    xtx = np.einsum('bni,bnj->bij', design, design, optimize=True)
    xty = np.einsum('bni,bn->bi', design, y, optimize=True)
    try:
        return np.linalg.solve(xtx, xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # A small minority of resamples can be singular when n is low.
        return np.einsum('bij,bj->bi', np.linalg.pinv(xtx), xty, optimize=True)


def _bootstrap_paths(arr, rng):
    """Vectorized nonparametric bootstrap of five indirect and four total effects."""
    n = len(arr)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    sample = arr[idx]
    b_vod = _batch_ols_coef(sample[:, :, 0], sample[:, :, [1, 2, 3, 4]])
    b_sm = _batch_ols_coef(sample[:, :, 1], sample[:, :, [3, 4, 2]])
    b_et = _batch_ols_coef(sample[:, :, 4], sample[:, :, [2]])

    sm_vod, vpd_vod, pre_vod, et_vod = [b_vod[:, i] for i in range(1, 5)]
    pre_sm, et_sm, vpd_sm = [b_sm[:, i] for i in range(1, 4)]
    vpd_et = b_et[:, 1]
    ind_pre = pre_sm * sm_vod
    ind_et = et_sm * sm_vod
    ind_vpd_sm = vpd_sm * sm_vod
    ind_vpd_et = vpd_et * et_vod
    ind_vpd_et_sm = vpd_et * et_sm * sm_vod
    tot_sm = sm_vod
    tot_vpd = vpd_vod + ind_vpd_sm + ind_vpd_et + ind_vpd_et_sm
    tot_pre = pre_vod + ind_pre
    tot_et = et_vod + ind_et
    return np.column_stack([
        ind_pre, ind_et, ind_vpd_sm, ind_vpd_et, ind_vpd_et_sm,
        tot_sm, tot_vpd, tot_pre, tot_et,
    ])


def _paths_from_coef(b_vod, b_sm, b_et):
    sm_vod, vpd_vod, pre_vod, et_vod = b_vod[1], b_vod[2], b_vod[3], b_vod[4]
    pre_sm, et_sm, vpd_sm = b_sm[1], b_sm[2], b_sm[3]
    vpd_et = b_et[1]
    ind_pre = pre_sm * sm_vod
    ind_et = et_sm * sm_vod
    ind_vpd_sm = vpd_sm * sm_vod
    ind_vpd_et = vpd_et * et_vod
    ind_vpd_et_sm = vpd_et * et_sm * sm_vod
    tot_sm = sm_vod
    tot_pre = pre_vod + ind_pre
    tot_et = et_vod + ind_et
    tot_vpd = vpd_vod + ind_vpd_sm + ind_vpd_et + ind_vpd_et_sm
    # 返回顺序：5 间接 + 4 总效应
    return np.array([ind_pre, ind_et, ind_vpd_sm, ind_vpd_et, ind_vpd_et_sm,
                     tot_sm, tot_vpd, tot_pre, tot_et])


def sem_one(lat, lon, arr, seed):
    VOD = arr[:, 0]; SM = arr[:, 1]; VPD = arr[:, 2]; PRE = arr[:, 3]; ET = arr[:, 4]
    n = len(VOD)

    # 点估计（VOD 主方程带 SE → 直接路径解析 p）
    b_vod, se_vod, p_vod, yhat = _ols_full(VOD, [SM, VPD, PRE, ET])
    b_sm = _ols_coef(SM, [PRE, ET, VPD])
    b_et = _ols_coef(ET, [VPD])

    sm_vod, vpd_vod, pre_vod, et_vod = b_vod[1], b_vod[2], b_vod[3], b_vod[4]
    eff = _paths_from_coef(b_vod, b_sm, b_et)
    (ind_pre, ind_et, ind_vpd_sm, ind_vpd_et, ind_vpd_et_sm,
     tot_sm, tot_vpd, tot_pre, tot_et) = eff

    # 拟合优度
    resid = VOD - yhat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = np.sum((VOD - VOD.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan

    # bootstrap：间接(5) + 总效应(4) 的分布
    rng = np.random.default_rng(seed)
    boot = _bootstrap_paths(arr, rng)
    ok = len(boot)

    point = eff  # [5 indirect effects, then totals in SM,VPD,PRE,ET order]
    if ok >= 20:
        frac_pos = np.mean(boot > 0, axis=0)
        p_boot = 2.0 * np.minimum(frac_pos, 1 - frac_pos)
        p_boot = np.clip(p_boot, 0.0, 1.0)
        ci_lo = np.percentile(boot, 2.5, axis=0)
        ci_hi = np.percentile(boot, 97.5, axis=0)
    else:
        p_boot = np.full(9, np.nan)
        ci_lo = np.full(9, np.nan)
        ci_hi = np.full(9, np.nan)

    # p_boot 索引：0..4 间接, 5..8 总(SM,VPD,PRE,ET)
    return [lat, lon, n,
            sm_vod, vpd_vod, pre_vod, et_vod,
            float(p_vod[1]), float(p_vod[2]), float(p_vod[3]), float(p_vod[4]),
            ind_pre, ind_et, ind_vpd_sm, ind_vpd_et, ind_vpd_et_sm,
            float(p_boot[0]), float(p_boot[1]), float(p_boot[2]), float(p_boot[3]), float(p_boot[4]),
            tot_sm, tot_vpd, tot_pre, tot_et,
            float(p_boot[5]), float(p_boot[6]), float(p_boot[7]), float(p_boot[8]),
            float(ci_lo[5]), float(ci_hi[5]), float(ci_lo[6]), float(ci_hi[6]),
            float(ci_lo[7]), float(ci_hi[7]), float(ci_lo[8]), float(ci_hi[8]),
            r2, rmse]


def sem_chunk(indices, cube, lats, lons):
    rows = []
    for j in indices:
        arr = cube[j]
        arr = arr[np.isfinite(arr).all(axis=1)].astype(float)
        rows.append(sem_one(lats[j], lons[j], arr, 1000 + int(j)))
    return rows


COLUMNS = [
    'lat', 'lon', 'n_months',
    'SM Direct', 'VPD Direct', 'PRE Direct', 'ET Direct',
    'p_SM_VOD', 'p_VPD_VOD', 'p_PRE_VOD', 'p_ET_VOD',
    'PRE-SM-VOD Indirect', 'ET-SM-VOD Indirect', 'VPD-SM-VOD Indirect', 'VPD-ET-VOD Indirect',
    'VPD-ET-SM-VOD Indirect',
    'p_ind_PRE_SM_VOD', 'p_ind_ET_SM_VOD', 'p_ind_VPD_SM_VOD', 'p_ind_VPD_ET_VOD',
    'p_ind_VPD_ET_SM_VOD',
    'SM Total', 'VPD Total', 'PRE Total', 'ET Total',
    'p_total_SM', 'p_total_VPD', 'p_total_PRE', 'p_total_ET',
    'SM_Total_CIlo', 'SM_Total_CIhi', 'VPD_Total_CIlo', 'VPD_Total_CIhi',
    'PRE_Total_CIlo', 'PRE_Total_CIhi', 'ET_Total_CIlo', 'ET_Total_CIhi',
    'R2', 'RMSE',
]


if __name__ == "__main__":
    t0 = time.time()
    print("Loading the common complete-case cube ...")
    cube, lats, lons, _dates = rf_main.load_pixel_cube()
    indices = np.arange(cube.shape[0])
    print("pixels (n_months>=%d): %d" % (MIN_MONTHS, len(indices)))

    chunks = [indices[k:k + CHUNK] for k in range(0, len(indices), CHUNK)]
    results = Parallel(n_jobs=N_JOBS, verbose=5, max_nbytes="10M", mmap_mode="r")(
        delayed(sem_chunk)(chunk, cube, lats, lons) for chunk in chunks
    )
    rows = [r for sub in results for r in sub]

    out_df = pd.DataFrame(rows, columns=COLUMNS)
    out_path = os.path.join(RESULT, "PLS_SEM_ETnolag.csv")
    out_df.to_csv(out_path, index=False)
    print("Saved %s  rows=%d  time=%.1fmin" % (out_path, len(out_df), (time.time() - t0) / 60))

    # 健康检查：恒等式 Total = Direct + ΣIndirect
    chk_vpd = (out_df['VPD Total'] - (
        out_df['VPD Direct'] + out_df['VPD-SM-VOD Indirect']
        + out_df['VPD-ET-VOD Indirect'] + out_df['VPD-ET-SM-VOD Indirect']
    )).abs().max()
    chk_pre = (out_df['PRE Total'] - (out_df['PRE Direct'] + out_df['PRE-SM-VOD Indirect'])).abs().max()
    chk_et = (out_df['ET Total'] - (out_df['ET Direct'] + out_df['ET-SM-VOD Indirect'])).abs().max()
    chk_sm = (out_df['SM Total'] - out_df['SM Direct']).abs().max()
    print("identity max|err|: SM=%.2e PRE=%.2e ET=%.2e VPD=%.2e" % (chk_sm, chk_pre, chk_et, chk_vpd))
