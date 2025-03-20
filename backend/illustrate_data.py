import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "dataset"
output_path = "output"
merged_csv_path = dataset_path + "/" + "merged_diabetes.csv"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 获取所有 CSV 文件
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
dataframes = []


# 读取 CSV 文件
def load_datasets():
    for file in csv_files:
        file_path = os.path.join(dataset_path, file)
        df = pd.read_csv(file_path)
        df["source_file"] = file  # 添加文件来源信息
        dataframes.append(df)
    return dataframes


# 加载数据集
dataframes = load_datasets()

# 合并数据集，并用 null 处理缺失值
merged_df = pd.concat(dataframes, ignore_index=True).fillna("null")
merged_df.to_csv(merged_csv_path, index=False)
print(f"Merged dataset saved as {merged_csv_path}")

# 选择第一个数据集进行 EDA 分析（数值型数据）
df1 = dataframes[0]


def plot_histograms(df, output_dir):
    df.hist(bins=20, figsize=(12, 10), edgecolor='black')
    plt.suptitle("Numerical Feature Distributions", fontsize=14)
    plt.savefig(os.path.join(output_dir, "histograms.png"))
    plt.close()
    print("Histogram plots saved.")


def plot_correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=["number"])  # 仅保留数值型列
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    print("Correlation heatmap saved.")


def plot_target_distribution(df, output_dir):
    if "Outcome" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df['Outcome'])
        plt.title("Diabetes Outcome Distribution")
        plt.xlabel("Diabetes (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "outcome_distribution.png"))
        plt.close()
        print("Outcome distribution plot saved.")


if __name__ == '__main__':
    # 生成图表
    plot_histograms(df1, output_path)
    plot_correlation_heatmap(df1, output_path)
    plot_target_distribution(df1, output_path)
    print("illustrate data completed.")
