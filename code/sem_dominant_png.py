import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
# 字体：本图为单幅图（不并列、不缩小）→ 用全局字号
FS = 14
matplotlib.rcParams['font.size'] = FS
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib.patches as mpatches
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm

# ======================================================
# 统一相对路径。一次性输出基于"直接效应"(Fig 9a) 与"总效应"(Fig 9b) 的主导图。
# 子图标题与正文的直接路径、总路径主导因子术语一致。
# ======================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "svvpesv")
RESULT = os.path.join(BASE, "result1")

file_path = os.path.join(RESULT, "PLS_SEM_ETnolag.csv")
df = pd.read_csv(file_path)

colors_dict = {
    'SM Importance +': '#f7b44d', 'SM Importance -': '#c28228',
    'PRE Importance +': '#1c7ed6', 'PRE Importance -': '#13538e',
    'VPD Importance +': '#e76a34', 'VPD Importance -': '#b14622',
    'ET Importance +': '#01a67c', 'ET Importance -': '#016e52'
}
legend_order = ['SM Importance +', 'SM Importance -', 'PRE Importance +', 'PRE Importance -',
                'VPD Importance +', 'VPD Importance -', 'ET Importance +', 'ET Importance -']
ordered_colors = [colors_dict[k] for k in legend_order]
color_to_code = {c: i for i, c in enumerate(ordered_colors)}

# nc 网格
ds = xr.open_dataset(os.path.join(DATA, "2010_06_svvpe_zv.nc"))
grid_lats = ds['lat'].values.astype(float)
grid_lons = ds['lon'].values.astype(float)


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


lat_edges = centers_to_edges(grid_lats)
lon_edges = centers_to_edges(grid_lons)


def nearest_index(sorted_centers, values):
    centers = np.asarray(sorted_centers, dtype=float)
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(centers, v, side='left')
    idx = np.clip(idx, 1, len(centers) - 1)
    left = idx - 1
    right = idx
    choose_right = (np.abs(centers[right] - v) < np.abs(v - centers[left]))
    return np.where(choose_right, right, left)


boxes = [
    (-123, -105, 30, 48), (56, 83, 45, 54), (140, 153, -38, -20),
    (25, 35, -35, -20), (16, 23, -18, -9), (98, 120, -5, 16),
    (-78, -59, -6, 7),
]


def draw_box(ax, lon_min, lon_max, lat_min, lat_max):
    ax.plot([lon_min, lon_max], [lat_min, lat_min], color='purple', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([lon_min, lon_max], [lat_max, lat_max], color='purple', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([lon_min, lon_min], [lat_min, lat_max], color='purple', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([lon_max, lon_max], [lat_min, lat_max], color='purple', linewidth=2, transform=ccrs.PlateCarree())


def make_map(mode, title, out_name):
    """mode: 'Direct' or 'Total' -> 用 'SM Direct'/'SM Total' 等列。"""
    cols = {'SM': f'SM {mode}', 'VPD': f'VPD {mode}', 'PRE': f'PRE {mode}', 'ET': f'ET {mode}'}
    factor_order = ['SM', 'VPD', 'PRE', 'ET']
    values = df[[cols[f] for f in factor_order]].to_numpy(float)
    leader = np.argmax(np.abs(values), axis=1)
    leader_value = values[np.arange(len(df)), leader]
    first_code = np.array([0, 4, 2, 6])
    codes = first_code[leader] + (leader_value < 0).astype(np.int16)

    grid_code = np.full((grid_lats.size, grid_lons.size), -1, dtype=np.int16)
    lat_idx = nearest_index(grid_lats, df['lat'].values.astype(float))
    lon_idx = nearest_index(grid_lons, df['lon'].values.astype(float))
    grid_code[lat_idx, lon_idx] = codes

    cmap = ListedColormap(ordered_colors)
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
    grid_masked = np.ma.masked_where(grid_code < 0, grid_code)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -60.01, 60.01], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': FS, 'color': 'black'}
    gl.ylabel_style = {'size': FS, 'color': 'black'}
    ax.pcolormesh(lon_edges, lat_edges, grid_masked, cmap=cmap, norm=norm,
                  transform=ccrs.PlateCarree(), shading='flat')
    for box_i, b in enumerate(boxes):
        draw_box(ax, *b)
        lon0, _lon1, _lat0, lat1 = b
        ax.text(lon0 + 0.8, lat1 - 1.2, chr(ord('A') + box_i), color='purple',
                fontsize=FS, fontweight='bold', ha='left', va='top',
                transform=ccrs.PlateCarree(),
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 0.6})

    # SMOS-IC cells are equal area, so global summaries use equal cell weights.
    area_weights = np.ones(len(df), dtype=float)
    total_weight = area_weights.sum()

    legend_handles = []
    print(f"=== SEM {mode}-effect dominance (%) ===")
    for key in legend_order:
        color = colors_dict[key]
        factor = key.split()[0]
        sign_display = "(+)" if key.split()[-1] == "+" else "(-)"
        code = legend_order.index(key)
        mask = codes == code
        percentage = 100 * area_weights[mask].sum() / total_weight
        print(f"  {factor} {sign_display}: {percentage:.1f}%")
        legend_handles.append(mpatches.Patch(color=color, label=f"{factor} Dominant {sign_display} ({percentage:.1f}%)"))

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.00, 0.31),
              ncol=1, fontsize=FS, framealpha=0.7, edgecolor='black', facecolor='white')
    ax.set_title(title, fontsize=15, pad=20)
    out = os.path.join(RESULT, "sem_s", out_name)
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", out)
    return pd.DataFrame({
        'mode': mode,
        'category': legend_order,
        'pixel_count': [(codes == i).sum() for i in range(8)],
        'pixel_percent': [100 * (codes == i).mean() for i in range(8)],
        'area_weighted_percent': [100 * area_weights[codes == i].sum() / total_weight for i in range(8)],
    })


if __name__ == "__main__":
    direct = make_map('Direct', 'Direct-path dominance', 'sem_dominant_d.png')
    total = make_map('Total', 'Total-path dominance', 'sem_dominant_t.png')
    pd.concat([direct, total], ignore_index=True).to_csv(
        os.path.join(RESULT, 'sem_s', 'SEM_dominance_summary.csv'), index=False
    )
