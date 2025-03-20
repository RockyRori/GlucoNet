import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    # 定义数据路径和模型存储路径
    dataset_path = "../dataset/cleaned_diabetes.csv"
    model_path = "logistic_regression.pkl"

    # 读取数据
    df = pd.read_csv(dataset_path)

    # 分割特征和目标变量
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, random_state=28)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 保存模型
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")
