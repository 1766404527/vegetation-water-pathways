import os
import numpy as np
import pandas as pd
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
import matplotlib.patches as mpatches
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm

# ===================== 统一相对路径 =====================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "result1")
csv_path = os.path.join(RESULT, "PLS_SEM_ETnolag.csv")
lc_nc_path = os.path.join(BASE, "data", "ccy2019.nc")
out_png = os.path.join(RESULT, "f_sem", "forest_vs_nonforest.png")

# ===================== 类型定义（按IGBP常见编码） =====================
FOREST_CODES = [1, 2, 3, 4, 5]
NONFOREST_VEG_CODES = [6, 7, 8, 9, 10, 11, 12, 14]  # 有植被但非森林
# 其它类型（0,13,15,16等）即使CSV有点也不画

# ===================== 颜色（你可自定义） =====================
forest_color = "#2ca25f"
nonforestveg_color = "#fdae6b"

# ===================== 工具函数：中心点 -> 边界（pcolormesh需要） =====================
def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges

# searchsorted 最近邻索引（要求 centers 单调递增）
def nearest_index(sorted_centers, values):
    centers = np.asarray(sorted_centers, dtype=float)
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(centers, v, side='left')
    idx = np.clip(idx, 1, len(centers) - 1)
    left = idx - 1
    right = idx
    choose_right = (np.abs(centers[right] - v) < np.abs(v - centers[left]))
    return np.where(choose_right, right, left)

# ===================== 读取 CSV（点） =====================
df = pd.read_csv(csv_path)

# 如果你的列名不是 lat/lon，请在这里改
pt_lat = df["lat"].to_numpy(dtype=float)
pt_lon = df["lon"].to_numpy(dtype=float)

# 去掉无效点
valid = np.isfinite(pt_lat) & np.isfinite(pt_lon)
pt_lat = pt_lat[valid]
pt_lon = pt_lon[valid]

# ===================== 读取 landcover nc（网格） =====================
ds = xr.open_dataset(lc_nc_path)

# 变量名：优先 data，否则取第一个数据变量
var_name = "data" if "data" in ds.data_vars else list(ds.data_vars)[0]
lc = ds[var_name]

# 坐标名兼容
if "latitude" in lc.coords and "lat" not in lc.coords:
    lc = lc.rename({"latitude": "lat"})
if "longitude" in lc.coords and "lon" not in lc.coords:
    lc = lc.rename({"longitude": "lon"})

grid_lat = lc["lat"].values.astype(float)
grid_lon = lc["lon"].values.astype(float)

# 若经度是 0–360，转换为 -180–180，并按经度排序
lon_wrapped = grid_lon.copy()
sorted_lon = False
if np.nanmax(grid_lon) > 180:
    lon_wrapped = ((grid_lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped = lon_wrapped[sort_idx]
    lc = lc.isel(lon=sort_idx)
    sorted_lon = True

# 若纬度从北到南，翻转到南到北（单调递增）
if grid_lat[0] > grid_lat[-1]:
    grid_lat = grid_lat[::-1]
    lc = lc.isel(lat=slice(None, None, -1))

grid_lat = lc["lat"].values.astype(float)
grid_lon = lon_wrapped.astype(float) if (np.nanmax(grid_lon) > 180) else lc["lon"].values.astype(float)

# CSV点经度也做同样的转换（如果你网格转了）
if np.nanmax(grid_lon) > 180:
    pt_lon = ((pt_lon + 180) % 360) - 180

lc_vals = lc.values

# ===================== 仅在 CSV 点所在像元上赋值：0=非森林有植被, 1=森林, -1=不画 =====================
grid_code = np.full((grid_lat.size, grid_lon.size), -1, dtype=np.int8)

# 过滤落在网格范围外的点（避免越界）
in_domain = (pt_lat >= grid_lat.min()) & (pt_lat <= grid_lat.max()) & (pt_lon >= grid_lon.min()) & (pt_lon <= grid_lon.max())
pt_lat = pt_lat[in_domain]
pt_lon = pt_lon[in_domain]

ilat = nearest_index(grid_lat, pt_lat)
ilon = nearest_index(grid_lon, pt_lon)

# 取这些点对应的地表类型
lc_at_pts = lc_vals[ilat, ilon]

is_forest = np.isin(lc_at_pts, FOREST_CODES)
is_nonforestveg = np.isin(lc_at_pts, NONFOREST_VEG_CODES)

# 填到网格：同一个像元多个点重复赋值无所谓
grid_code[ilat[is_nonforestveg], ilon[is_nonforestveg]] = 0
grid_code[ilat[is_forest], ilon[is_forest]] = 1

# mask 掉 -1（即：CSV有点但不是两类，或没有点的像元）
grid_masked = np.ma.masked_where(grid_code < 0, grid_code)

# pcolormesh edges
lat_edges = centers_to_edges(grid_lat)
lon_edges = centers_to_edges(grid_lon)

# cmap / norm（两类）
cmap = ListedColormap([nonforestveg_color, forest_color])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

# ===================== 画图（风格参考你给的代码） =====================
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

pm = ax.pcolormesh(
    lon_edges, lat_edges, grid_masked,
    cmap=cmap, norm=norm,
    transform=ccrs.PlateCarree(),
    shading='flat'
)

# ===================== 图例（不统计占比，放左下角） =====================
handles = [
    mpatches.Patch(color=forest_color, label="Forest"),
    mpatches.Patch(color=nonforestveg_color, label="Non-Forest"),
]

ax.legend(handles=handles,
          loc='lower left',                 # 左下角
          bbox_to_anchor=(0.02, 0.02),      # 稍微往里一点，避免贴边
          ncol=1,
          fontsize=FS,
          framealpha=0.7,
          edgecolor='black',
          facecolor='white')

ax.set_title("Forest vs Non-forest Vegetated", fontsize=15, pad=20)

plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved:", out_png)
