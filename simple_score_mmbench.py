import pandas as pd

# Excel 文件路径
excel_path = "./playground/data/eval/mmbench/answers_upload/llava-v1.5-7b-dualcd-lora-dcd-layer1.xlsx"

print(f"📁 正在读取文件: {excel_path}\n")

# 读取 Excel
df = pd.read_excel(excel_path)

print("表格列名:", list(df.columns))
print()

# 有些 MMBench 文件列名可能带空格，统一清洗一下
df.columns = [str(c).strip().lower() for c in df.columns]

# prediction / answer 转成字符串并标准化
df["prediction"] = (
    df["prediction"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["answer"] = (
    df["answer"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# 去掉 prediction 为空或非法的样本
valid_choices = ["A", "B", "C", "D"]
valid_mask = df["prediction"].isin(valid_choices) & df["answer"].isin(valid_choices)

invalid_num = (~valid_mask).sum()
if invalid_num > 0:
    print(f"⚠️ 跳过 {invalid_num} 条 prediction / answer 非法的数据\n")

df_valid = df[valid_mask].copy()

# 判断是否正确
df_valid["correct"] = df_valid["prediction"] == df_valid["answer"]

# 总准确率
overall_acc = df_valid["correct"].mean() * 100

print("📊 ============ MMBench 最终成绩 ============")
print(f"总样本数           : {len(df_valid)}")
print(f"正确样本数         : {df_valid['correct'].sum()}")
print(f"Overall Accuracy   : {overall_acc:.2f}%")
print("-" * 45)

# 如果有 split 列，则分别统计 dev / test / val
if "split" in df_valid.columns:
    print("📌 各 split 准确率:")
    split_acc = df_valid.groupby("split")["correct"].mean() * 100
    split_cnt = df_valid.groupby("split")["correct"].count()

    for split in split_acc.index:
        print(
            f"  {str(split).ljust(8)} "
            f"{split_acc[split]:6.2f}%   "
            f"({split_cnt[split]} samples)"
        )

    print("-" * 45)

# 如果存在 category 列，则统计分类准确率
cat_col = None
for col in ["l2-category", "category", "l2_category", "category2"]:
    if col in df_valid.columns:
        cat_col = col
        break

if cat_col is not None:
    print(f"📌 分类准确率（按 {cat_col}）:")
    category_acc = df_valid.groupby(cat_col)["correct"].mean() * 100
    category_cnt = df_valid.groupby(cat_col)["correct"].count()

    for cat in category_acc.index:
        print(
            f"  {str(cat).ljust(20)} "
            f"{category_acc[cat]:6.2f}%   "
            f"({category_cnt[cat]} samples)"
        )

print("=" * 45)