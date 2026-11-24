import xarray as xr
import numpy as np
import os
from collections import defaultdict

# ====== 以下为本机原始逐月数据文件夹（未随仓库提供）—— 请改为你的实际路径 ======
# 各变量文件夹路径
vod_dir = r"D:\Desktop\vodmonth"
sm_dir = r"D:\Desktop\smmonth"
vpd_dir = r"D:\Desktop\vpdmonth"
et_dir = r"D:\Desktop\etmonth"
pre_dir = r"D:\Desktop\premonth"

# 输出路径
output_dir = r"D:\Desktop\svvpe"
os.makedirs(output_dir, exist_ok=True)

# 收集所有文件路径
def collect_files_by_month(folder):
    file_map = {}
    for file in os.listdir(folder):
        if file.endswith(".nc"):
            month_key = file[:7]  # 提取年月
            file_map[month_key] = os.path.join(folder, file)
    return file_map

vod_files = collect_files_by_month(vod_dir)
sm_files = collect_files_by_month(sm_dir)
vpd_files = collect_files_by_month(vpd_dir)
et_files = collect_files_by_month(et_dir)
pre_files = collect_files_by_month(pre_dir)

# 对所有有完整五种数据的月份进行处理
all_keys = set(vod_files) & set(sm_files) & set(vpd_files) & set(et_files) & set(pre_files)

def bilinear_interp(ds, var_name, target_lat, target_lon):
    return ds[var_name].interp(lat=target_lat, lon=target_lon, method="linear").astype(np.float32)

for month in sorted(all_keys):
    ds_vod = xr.open_dataset(vod_files[month])
    ds_sm = xr.open_dataset(sm_files[month])
    ds_vpd = xr.open_dataset(vpd_files[month])
    ds_et = xr.open_dataset(et_files[month])
    ds_pre = xr.open_dataset(pre_files[month])

    # 插值到 VOD 网格
    target_lat = ds_vod.lat
    target_lon = ds_vod.lon

    sm_interp = bilinear_interp(ds_sm, "SM", target_lat, target_lon)
    vpd_interp = bilinear_interp(ds_vpd, "VPD", target_lat, target_lon)
    et_interp = bilinear_interp(ds_et, "ET", target_lat, target_lon)
    pre_interp = bilinear_interp(ds_pre, "PRE", target_lat, target_lon)

    # 处理 PRE：当 VPD 为 NaN 时，PRE 设为 NaN
    pre_masked = pre_interp.where(~np.isnan(vpd_interp))

    # 纬度裁剪到 -60 ~ 60
    lat_mask = (target_lat >= -60) & (target_lat <= 60)
    clipped_lat = target_lat.where(lat_mask, drop=True)

    vod_clipped = ds_vod.VOD.sel(lat=clipped_lat).astype(np.float32)
    sm_clipped = sm_interp.sel(lat=clipped_lat)
    vpd_clipped = vpd_interp.sel(lat=clipped_lat)
    et_clipped = et_interp.sel(lat=clipped_lat)
    pre_clipped = pre_masked.sel(lat=clipped_lat)

    # 构建合并数据集
    merged = xr.Dataset(
        {
            "VOD": vod_clipped.assign_attrs({"units": "unit", "long_name": "VOD", "_FillValue": np.nan}),
            "SM": sm_clipped.assign_attrs({"units": "m³/m³", "long_name": "SM", "_FillValue": np.nan}),
            "VPD": vpd_clipped.assign_attrs({"units": "hPa", "long_name": "VPD", "_FillValue": np.nan}),
            "ET": et_clipped.assign_attrs({"units": "mm", "long_name": "ET", "_FillValue": np.nan}),
            "PRE": pre_clipped.assign_attrs({"units": "mm", "long_name": "PRE", "_FillValue": np.nan})
        },
        coords={
            "lat": clipped_lat.assign_attrs({"long_name": "lat"}),
            "lon": target_lon.assign_attrs({"long_name": "lon"})
        }
    )

    # 最终保存
    output_path = os.path.join(output_dir, f"{month}_svvpe.nc")
    merged.to_netcdf(output_path)
    print(f"✔ 合并完成：{output_path}")
