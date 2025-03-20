import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# 忽略 numpy 的 MaskedArray 警告
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

if __name__ == '__main__':
    # 读取合并后的数据
    dataset_path = "dataset"
    merged_csv_path = dataset_path + "/" + "merged_diabetes.csv"
    cleaned_csv_path = dataset_path + "/" + "cleaned_diabetes.csv"
    df = pd.read_csv(merged_csv_path)

    # 1. 处理 `CLASS` 和 `class`，转换为 `Outcome`
    outcome_columns = ["CLASS", "class", "Outcome"]
    for col in outcome_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().replace({"yes": 1, "no": 0, "positive": 1, "negative": 0})
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 取 `CLASS`、`class`、`Outcome` 三者的最大值作为最终 `Outcome`
    df["Outcome"] = df[outcome_columns].max(axis=1, skipna=True)
    df.drop(columns=[col for col in outcome_columns if col != "Outcome"], inplace=True)

    # 2. 删除无意义列
    drop_columns = ["ID", "No_Pation", "source_file"]
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    # 3. 合并 `Age` 和 `AGE`
    if "Age" in df.columns and "AGE" in df.columns:
        df["Age"] = df[["Age", "AGE"]].max(axis=1, skipna=True)
        df.drop(columns=["AGE"], inplace=True)

    # 4. 转换类别变量为 0/1
    binary_columns = ["Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
                      "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability",
                      "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"]
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().replace({"yes": 1, "no": 0, "male": 1, "female": 0})
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. 异常值处理（使用 Winsorization 代替 IQR 过滤）
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = pd.Series(winsorize(df[col], limits=[0.01, 0.01]))  # 截断前1% 和 后1% 数据，转换回 Pandas Series

    # 6. 处理 NaN 值
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # 只对数值列操作
            if df[col].isna().all():  # 如果整列都是 NaN
                df[col] = 0
            else:
                df[col] = df[col].fillna(df[col].median(skipna=True))  # 用中位数填充
        else:
            df.loc[:, col] = df[col].fillna(0)  # 类别变量填充 0，避免链式赋值警告

    # 7. 进行数值标准化
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if "Outcome" in numeric_cols:
        numeric_cols = numeric_cols.drop("Outcome")  # Outcome 不进行标准化
    df[numeric_cols] = df[numeric_cols].fillna(0)  # 确保无 NaN
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 8. 确保 Outcome 在最后一列
    cols = [col for col in df.columns if col != "Outcome"] + ["Outcome"]
    df = df[cols]

    # 9. 保存清理后的数据
    df.to_csv(cleaned_csv_path, index=False)
    print("clean data completed.")
