from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ======================
# 1. 路径设置（自己改这里）
# ======================
# 输入 nc 文件所在文件夹（每个文件 = 某年某月，名字形如 2010_06_svvpe.nc）
data_dir = Path(r"D:\Desktop\svvpe")

# 输出标准化后的文件夹（可以不存在，代码会自动创建）
out_dir = Path(r"D:\Desktop\svvpes")
out_dir.mkdir(parents=True, exist_ok=True)

# 如果你的文件名模式不一样，可以改这里的通配符
files = sorted(data_dir.glob("*_svvpe.nc"))

print(f"发现 {len(files)} 个文件")

# ======================
# 2. 读入所有文件，并加上 time 维度
# ======================
dss = []
for fp in files:
    ds = xr.open_dataset(fp)

    # 假设文件名是 2010_06_svvpe.nc 这种结构
    stem = fp.stem  # '2010_06_svvpe'
    year_str, month_str, *_ = stem.split("_")
    time = np.datetime64(f"{year_str}-{month_str}-15")  # 这一天随便选

    # 增加 time 维度
    ds = ds.expand_dims(time=[time])
    dss.append(ds)

# 拼成一个大 Dataset：time × lat × lon
ds_all = xr.concat(dss, dim="time")

# ======================
# 3. 选出需要标准化的五个变量
# ======================
vars5 = ["VOD", "SM", "VPD", "ET", "PRE"]
ds5 = ds_all[vars5]

# ======================
# 4. 计算逐月气候态均值 & 标准差（13 年）
# ======================
clim_mean = ds5.groupby("time.month").mean("time")
clim_std  = ds5.groupby("time.month").std("time")

# 防止标准差为 0 导致除 0
clim_std = clim_std.where(clim_std > 0)

# ======================
# 5. 计算逐月 z-score 异常： (X - mean_month) / std_month
# ======================
anom = ds5.groupby("time.month") - clim_mean
z = anom.groupby("time.month") / clim_std   # 这里就是 VOD/SM/VPD/ET/PRE 的 z-score

# ======================
# 6. 按时间拆回每月文件保存到自定义 out_dir
# ======================
for t in z.time.values:
    ds_t = z.sel(time=t)

    # 去掉 groupby 加的 month 坐标（可选）
    if "month" in ds_t.coords:
        ds_t = ds_t.drop_vars("month")

    dt = pd.to_datetime(str(t))
    year = dt.year
    month = dt.month

    out_name = f"{year:04d}_{month:02d}_svvpe_z.nc"
    out_path = out_dir / out_name

    ds_t.to_netcdf(out_path)
    print("保存：", out_path)
