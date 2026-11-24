import os
import pandas as pd

# ======================================================
# 统一相对路径：合并随机森林重要性与相关分析结果
# ======================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(BASE, "result1")
out_dir = os.path.join(RESULT, "RandomForest Importance")
os.makedirs(out_dir, exist_ok=True)

df_importance = pd.read_csv(os.path.join(RESULT, "Worlds_importancenolag.csv"))
df_correlation = pd.read_csv(os.path.join(RESULT, "Worlds_nolag.csv"))

# 基于 lat/lon 合并（内连接：仅保留通过样本量阈值的随机森林像元）
df_merged = pd.merge(
    df_importance,
    df_correlation,
    on=['lat', 'lon'],
    how='inner',
    validate='one_to_one',
)
if len(df_merged) != len(df_importance):
    raise RuntimeError(
        f"Correlation/RF mask mismatch: RF={len(df_importance)}, merged={len(df_merged)}"
    )

out_path = os.path.join(out_dir, "pearsonr_importancenolag.csv")
df_merged.to_csv(out_path, index=False)

print(f"Merged CSV saved to '{out_path}'  (rows={len(df_merged)})")
