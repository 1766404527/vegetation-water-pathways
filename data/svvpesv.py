from pathlib import Path
import xarray as xr

# 仓库根目录（本文件位于 <repo>/data/ 下）
BASE = Path(__file__).resolve().parent.parent

# ====== 路径设置 ======
# sv_dir：上一步 svvpes 标准化结果（中间产物，未随仓库提供）—— 请改为本机实际路径
sv_dir   = Path(r"D:\Desktop\svvpes")            # svvpes nc 所在文件夹
# 下面两项指向仓库内置文件/文件夹，已用相对路径，无需修改
ccy_path = BASE / "data" / "ccy2019.nc"          # 植被分类文件（仓库内置）
out_dir  = BASE / "data" / "svvpesv"             # 输出文件夹（= 仓库最终输入 data/svvpesv）
# ========================

out_dir.mkdir(parents=True, exist_ok=True)

# 非植被类别：0(水体), 13(城市), 15(冰雪), 16(裸地/稀疏植被)
nonveg_classes = [0, 13, 15, 16]
ccy_varname = "data"

# 1) 读取分类并做植被mask（只做一次，后面复用）
ds_ccy = xr.open_dataset(ccy_path)
veg_mask_base = (~ds_ccy[ccy_varname].isin(nonveg_classes)) & ds_ccy[ccy_varname].notnull()

# 2) 批量处理 sv_dir 下所有 nc（排除 ccy2019 本身）
nc_files = sorted(sv_dir.glob("*.nc"))
print(f"Found {len(nc_files)} nc files in {sv_dir}")

for f in nc_files:
    if f.resolve() == ccy_path.resolve():
        continue

    ds = xr.open_dataset(f)

    # 优先要求网格完全一致；不一致则用最近邻把mask投到该网格
    try:
        ds_aligned, mask_aligned = xr.align(ds, veg_mask_base, join="exact")
    except Exception:
        mask_aligned = veg_mask_base.interp_like(ds, method="nearest")
        ds_aligned = ds

    ds_out = ds_aligned.where(mask_aligned)

    out_file = out_dir / (f.stem + "v.nc")
    ds_out.to_netcdf(out_file)

    print("Saved:", out_file)

print("Done.")
