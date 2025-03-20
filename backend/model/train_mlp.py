import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    # 定义数据路径和模型存储路径
    dataset_path = "../dataset/cleaned_diabetes.csv"
    model_path = "mlp.pkl"

    # 读取数据
    df = pd.read_csv(dataset_path)

    # 分割特征和目标变量
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # 训练 MLP 神经网络模型
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=28)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print("MLP Neural Network Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 保存模型
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")
