import joblib
import pandas as pd

if __name__ == '__main__':
    model_path = "model"
    output_path = "output"
    model_pkl_path = model_path + "/" + "xgboost.pkl"
    feature_importance_csv_path = output_path + "/" + "feature_importance.csv"

    # 加载模型
    model = joblib.load(model_pkl_path)

    # 特征名
    features = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Gender", "Urea", "Cr", "HbA1c", "Chol", "TG",
        "HDL", "LDL", "VLDL", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
        "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
    ]

    # 提取并排序
    importance = model.feature_importances_
    df = pd.DataFrame({"Feature": features, "Importance": importance})
    df.sort_values(by="Importance", ascending=False).to_csv(feature_importance_csv_path, index=False)
