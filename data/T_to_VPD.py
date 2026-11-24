import xarray as xr
import numpy as np
import os
import glob

# --- 配置部分 ---
# 定义存放输入nc文件的文件夹路径
data_folder = r'D:\Desktop\ERA5-LAND' # <-- 请修改为你的输入文件夹实际路径
# 定义存放输出nc文件的文件夹路径
output_folder = r'D:\Desktop\VPD' # <-- 请修改为你的输出文件夹实际路径

# 定义数据集中温度和露点温度变量的名称
temp_var_name = 't2m' # 假设温度变量名为 t2m
dewpoint_var_name = 'd2m' # 假设露点温度变量名为 d2m
# 定义数据集中的时间维度名称
time_dim_name = 'valid_time' # 假设时间维度名为 valid_time
# 定义可能需要移除的额外坐标变量名称 (如 ensemble member)
coords_to_drop = ['number'] # <-- 如果还有其他不需要的坐标，可以添加到这个列表

# --- Magnus 公式函数 ---
def magnus_formula(temp_celsius):
    """
    使用 Magnus 公式计算饱和蒸气压。
    输入: temp_celsius (xarray DataArray 或 float), 摄氏度
    输出: 饱和蒸气压 (xarray DataArray 或 float), 单位 hPa
    """
    return 6.1094 * np.exp(17.625 * temp_celsius / (temp_celsius + 243.04))

# --- 检查或创建输出文件夹 ---
if not os.path.exists(output_folder):
    print(f"输出文件夹不存在，正在创建: {output_folder}")
    try:
        os.makedirs(output_folder)
        print("输出文件夹创建成功。")
    except OSError as e:
        print(f"创建输出文件夹失败: {e}")
        print("请检查路径是否有权限创建文件夹。程序将退出。")
        exit() # 如果无法创建输出文件夹，程序就没有必要继续了

# --- 扫描输入文件夹并分组文件 ---
print(f"正在扫描输入文件夹: {data_folder}")
monthly_files = {} # 字典用于存储按月份分组的文件: {'YYYYMM': {'d': 'path_to_d.nc', 't': 'path_to_t.nc'}}

file_list = glob.glob(os.path.join(data_folder, '*_[dt].nc'))

if not file_list:
    print(f"在输入文件夹 {data_folder} 中没有找到符合模式 (*_[dt].nc) 的文件。请检查路径和文件名。")
else:
    for full_path in file_list:
        filename = os.path.basename(full_path)
        if len(filename) >= 9:
            yyyymm = filename[:6]
            file_type = filename[7:8]
            if file_type in ['d', 't']:
                if yyyymm not in monthly_files:
                    monthly_files[yyyymm] = {}
                monthly_files[yyyymm][file_type] = full_path

    # 确保每个月份都有 'd' 和 't' 两个文件
    months_to_process = sorted([
        yyyymm for yyyymm, files in monthly_files.items()
        if 'd' in files and 't' in files
    ])

    if not months_to_process:
        print("没有找到成对的 (t 和 d) 月份文件进行处理。请确保文件名格式为YYYYMM_[d/t].nc")
    else:
        print(f"找到以下月份进行处理: {months_to_process}")

        # --- 按月份循环处理 ---
        for yyyymm in months_to_process:
            dewpoint_file = monthly_files[yyyymm]['d']
            temperature_file = monthly_files[yyyymm]['t']
            year = yyyymm[:4]
            month = yyyymm[4:]
            output_filename = f"{year}_{month}_vpd.nc"
            # *** 修改这里，使用 output_folder 构建输出文件路径 ***
            output_file_path = os.path.join(output_folder, output_filename)
            # *******************************************************

            print(f"\n--- 正在处理 {yyyymm} (露点文件: {os.path.basename(dewpoint_file)}, 温度文件: {os.path.basename(temperature_file)}) ---")
            print(f"--- 结果将保存到: {os.path.basename(output_file_path)} ---") # 打印输出文件名

            # 使用独立的 try...except...finally 块处理每个月份
            dewpoint_ds = None
            temperature_ds = None
            try:
                # 1. 打开数据集
                print("正在打开数据集...")
                dewpoint_ds = xr.open_dataset(dewpoint_file)
                temperature_ds = xr.open_dataset(temperature_file)
                print("数据集打开成功。")

                # 2. 提取变量并进行单位转换 (K -> C)
                if temp_var_name not in temperature_ds.variables:
                    raise ValueError(f"温度变量 '{temp_var_name}' 未在文件 '{temperature_file}' 中找到。")
                if dewpoint_var_name not in dewpoint_ds.variables:
                     raise ValueError(f"露点温度变量 '{dewpoint_var_name}' 未在文件 '{dewpoint_file}' 中找到。")
                # 可以添加检查时间维度是否一致的代码，这里简化处理，假设它们是匹配的

                t2m = temperature_ds[temp_var_name]
                d2m = dewpoint_ds[dewpoint_var_name]

                print("正在进行单位转换 (K -> C)...")
                t2m_celsius = t2m - 273.15
                d2m_celsius = d2m - 273.15
                print("单位转换完成。")

                print("正在计算饱和蒸气压 (es) 和实际蒸气压 (ea)...")
                # 3. 计算饱和蒸气压 (es) 和实际蒸气压 (ea)
                es = magnus_formula(t2m_celsius)
                ea = magnus_formula(d2m_celsius)
                print("蒸气压计算完成。")

                print("正在计算 VPD...")
                # 4. 计算 VPD
                vpd = es - ea
                print("VPD 计算完成。")

                print(f"正在计算 {time_dim_name} 维度的月平均值...")
                # 5. 计算 VPD 的月平均值
                if time_dim_name not in vpd.dims:
                    print(f"警告: VPD 数据中未找到时间维度 '{time_dim_name}'。无法计算月平均，跳过保存。")
                    # 可以选择在这里 continue 跳过保存，或根据需求处理
                    continue # 跳过当前月份的处理和保存
                vpd_monthly_mean = vpd.mean(dim=time_dim_name)
                print("月平均 VPD 计算完成。")

                # 6. 删除不需要的坐标变量
                for coord_name in coords_to_drop:
                    if coord_name in vpd_monthly_mean.coords:
                        print(f"检测到并尝试移除坐标: '{coord_name}'...")
                        try:
                            vpd_monthly_mean = vpd_monthly_mean.drop_vars(coord_name)
                            print(f"'{coord_name}' 坐标已移除。")
                        except Exception as drop_e:
                            print(f"移除坐标 '{coord_name}' 时发生错误: {drop_e}")
                            # 不中断流程，继续执行

                # 7. 创建新的 Dataset 用于输出
                output_ds = xr.Dataset({
                    'VPD': vpd_monthly_mean
                })

                # 添加一些元数据属性
                output_ds['VPD'].attrs['units'] = 'hPa'
                output_ds['VPD'].attrs['long_name'] = f"Monthly Mean Vapor Pressure Deficit ({yyyymm})"
                output_ds.attrs['description'] = f'Monthly mean VPD calculated from 2m temperature and dewpoint temperature using Magnus formula for {yyyymm}'
                output_ds.attrs['calculation_date'] = str(np.datetime64('now'))
                output_ds.attrs['calculation_month'] = yyyymm

                # 检查输出文件是否已存在，避免覆盖（可选）
                # if os.path.exists(output_file_path):
                #     print(f"警告: 输出文件已存在，将覆盖: {os.path.basename(output_file_path)}")


                print(f"正在保存结果到文件: {os.path.basename(output_file_path)}")
                # 8. 将结果保存为新的 netCDF 文件
                output_ds.to_netcdf(output_file_path)
                print(f"成功保存月平均 VPD 到文件: {os.path.basename(output_file_path)}")

            except FileNotFoundError as e:
                print(f"处理 {yyyymm} 时错误：输入文件未找到 - {e}")
            except ValueError as e:
                print(f"处理 {yyyymm} 时数据处理错误：{e}")
            except Exception as e:
                print(f"处理 {yyyymm} 时发生未知错误: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # 确保关闭文件句柄
                print(f"正在尝试关闭 {yyyymm} 相关文件句柄...")
                if dewpoint_ds is not None:
                    try:
                        dewpoint_ds.close()
                        # print(f"已关闭文件句柄: {os.path.basename(dewpoint_file)}") # 不频繁打印关闭信息
                    except Exception as e:
                        print(f"关闭文件句柄时发生错误 {os.path.basename(dewpoint_file)}: {e}")

                if temperature_ds is not None:
                    try:
                        temperature_ds.close()
                        # print(f"已关闭文件句柄: {os.path.basename(temperature_file)}") # 不频繁打印关闭信息
                    except Exception as e:
                         print(f"关闭文件句柄时发生错误 {os.path.basename(temperature_file)}: {e}")
                # print(f"{yyyymm} 文件句柄关闭尝试完成。") # 不频繁打印关闭尝试完成信息

print("\n所有符合条件的月份处理完毕。程序结束。")