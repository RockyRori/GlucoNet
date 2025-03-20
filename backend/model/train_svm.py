import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

"""
训练时间10分钟，不建议运行！
"""
if __name__ == '__main__':
    # 定义数据路径和模型存储路径
    dataset_path = "../dataset/cleaned_diabetes.csv"
    model_path = "svm.pkl"

    # 读取数据
    df = pd.read_csv(dataset_path)

    # 分割特征和目标变量
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # 进行超参数调优
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(SVC(probability=True, class_weight='balanced', random_state=28), param_grid, cv=5)
    grid.fit(X_train, y_train)

    # 选取最佳模型
    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # 预测
    y_pred = best_model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print("Optimized SVM Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 保存模型
    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)
    print(f"Optimized model saved to {model_path}")
