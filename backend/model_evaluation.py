import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == '__main__':
    # 读取数据
    dataset_path = "dataset"
    merged_csv_path = dataset_path + "/" + "cleaned_diabetes.csv"
    df = pd.read_csv(merged_csv_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 确保输出文件夹存在
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)


    # 加载模型
    def load_model(model_name):
        model_path = f"model/{model_name}"
        with open(model_path, "rb") as file:
            return pickle.load(file)


    models = {
        "Logistic Regression": load_model("logistic_regression.pkl"),
        "Random Forest": load_model("random_forest.pkl"),
        "SVM": load_model("svm.pkl"),
        "XGBoost": load_model("xgboost.pkl"),
        "MLP Neural Network": load_model("mlp.pkl")
    }

    # 评估所有模型
    results = []
    for name, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1})

    df_results = pd.DataFrame(results)
    results_csv_path = os.path.join(output_path, "model_evaluation.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"Evaluation results saved to {results_csv_path}")

    # 绘制可视化
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Accuracy", data=df_results, palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    accuracy_plot_path = os.path.join(output_path, "accuracy_comparison.png")
    plt.savefig(accuracy_plot_path)
    plt.show()
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Recall", data=df_results, palette="magma")
    plt.title("Model Recall Comparison (Important for Diabetes Detection)")
    plt.ylabel("Recall")
    plt.xticks(rotation=30)
    recall_plot_path = os.path.join(output_path, "recall_comparison.png")
    plt.savefig(recall_plot_path)
    plt.show()
    print(f"Recall plot saved to {recall_plot_path}")

    # 输出最终选择模型的原因
    best_model = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
    print("Final Model Selection:")
    print(f"The best model is {best_model['Model']} with an accuracy of {best_model['Accuracy']:.4f}.")
