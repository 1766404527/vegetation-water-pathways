# -*- coding: utf-8 -*-
"""
B5 独立产品交叉验证图(2 面板):
(a) 耦合检验散点：x = corr(SMOS_VOD, SMOS_SM)（同源反演），
                   y = corr(SMOS_VOD, SMAP_SM)（独立 SMAP SM）。
    点沿 1:1 → 把同源 SM 换成独立 SM 后 VOD–SM 相关不变 → SM 主导非反演耦合伪迹。
(b) SM 互验地图：逐像元 corr(SMOS_SM, SMAP_SM)，干旱区接近 0.8 → 头号驱动被独立产品复现。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
# 字体
FS = 26
matplotlib.rcParams['font.size'] = FS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "result1", "smap_validation", "smap_validation.csv")
REF = os.path.join(BASE, "data", "svvpesv", "2010_06_svvpe_zv.nc")
OUT = os.path.join(BASE, "result1", "smap_validation", "Fig_SMAP_validation.png")

df = pd.read_csv(CSV)

# ===================== (a) 耦合检验散点 =====================
x = df['SMOSVOD_SMOSSM'].to_numpy()
y = df['SMOSVOD_SMAPSM'].to_numpy()
ok = np.isfinite(x) & np.isfinite(y)
x, y = x[ok], y[ok]
r = np.corrcoef(x, y)[0, 1]
ss = np.mean(np.sign(x) == np.sign(y)) * 100
strong = np.abs(x) >= 0.3
ss_strong = np.mean(np.sign(x[strong]) == np.sign(y[strong])) * 100

# ===================== (b) SM 互验图：读网格 =====================
ds = xr.open_dataset(REF)
glat = ds['lat'].values.astype(float)
glon = ds['lon'].values.astype(float)


def centers_to_edges(c):
    c = np.asarray(c, float)
    e = np.empty(c.size + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0] = c[0] - (e[1] - c[0]); e[-1] = c[-1] + (c[-1] - e[-2])
    return e


def nearest_index(centers, v):
    centers = np.asarray(centers, float); v = np.asarray(v, float)
    idx = np.clip(np.searchsorted(centers, v, 'left'), 1, len(centers) - 1)
    return np.where(np.abs(centers[idx] - v) < np.abs(v - centers[idx - 1]), idx, idx - 1)


lat_e, lon_e = centers_to_edges(glat), centers_to_edges(glon)
grid = np.full((glat.size, glon.size), np.nan)
li = nearest_index(glat, df['lat'].values)
oi = nearest_index(glon, df['lon'].values)
vals = df['SMOSSM_SMAPSM'].values
for i in range(len(vals)):
    grid[li[i], oi[i]] = vals[i]
grid = np.ma.masked_invalid(grid)

# ===================== 画图（左右排版） =====================
fig = plt.figure(figsize=(18, 6.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.7], wspace=0.34)

# --- (a) 左：耦合检验散点 ---
axa = fig.add_subplot(gs[0])
axa.text(-0.18, 1.06, 'a)', transform=axa.transAxes, fontsize=FS + 2,
         fontweight='bold', ha='left', va='bottom')
hb = axa.hexbin(x, y, gridsize=45, extent=(-1, 1, -1, 1), mincnt=1,
                cmap='viridis', bins='log')
axa.plot([-1, 1], [-1, 1], 'r--', lw=1.5, zorder=5)
axa.axhline(0, color='gray', lw=0.6); axa.axvline(0, color='gray', lw=0.6)
axa.set_xlim(-1, 1); axa.set_ylim(-1, 1); axa.set_aspect('equal')
# 标签尽量精简，避免大字号下相互挤压
axa.set_xlabel('r(VOD, SMOS SM)\n(same retrieval)', fontsize=FS)
axa.set_ylabel('r(VOD, SMAP SM)\n(independent)', fontsize=FS)
axa.tick_params(labelsize=FS)
# 统计文字框放右下角（该区数据稀疏，不挡点）；多行框略小以免溢出窄面板
axa.text(0.96, 0.05,
         f'r = {r:.2f}\nsame sign {ss:.0f}%\n|r|≥0.3: {ss_strong:.0f}%',
         transform=axa.transAxes, va='bottom', ha='right', fontsize=FS - 4,
         bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
cba = fig.colorbar(hb, ax=axa, fraction=0.046, pad=0.03)
cba.ax.set_title('count\n(log)', fontsize=FS - 4, pad=8)
cba.ax.tick_params(labelsize=FS - 4)

# --- (b) 右：SM 互验地图 ---
axb = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
axb.text(-0.08, 1.08, 'b)', transform=axb.transAxes, fontsize=FS + 2,
         fontweight='bold', ha='left', va='bottom')
axb.set_extent([-180, 180, -60.01, 60.01], crs=ccrs.PlateCarree())
axb.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
axb.add_feature(cfeature.OCEAN, facecolor='white')
axb.add_feature(cfeature.COASTLINE, edgecolor='black')
axb.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
gl = axb.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
gl.xlabel_style = {'size': FS}; gl.ylabel_style = {'size': FS}
pm = axb.pcolormesh(lon_e, lat_e, grid, cmap='viridis', vmin=-0.45, vmax=1,
                    transform=ccrs.PlateCarree(), shading='flat')
cbb = fig.colorbar(pm, ax=axb, orientation='vertical', fraction=0.025, pad=0.03)
cbb.set_label('SMAP–SMOS SM r', fontsize=FS)
cbb.ax.tick_params(labelsize=FS - 4)

plt.savefig(OUT, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved:", OUT)
print(f"(a) coupling: spatial r={r:.2f}, same-sign={ss:.0f}%, same-sign|r|>=0.3={ss_strong:.0f}%")
print(f"(b) SM-SM corr: median={np.nanmedian(vals):.2f}")
