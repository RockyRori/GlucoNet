import os
import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import pandas as pd
import numpy as np
from data_model import UserInput, GENDER_MAPPING, BINARY_MAPPING

# 获取当前 `local_app.py` 的文件夹路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 计算 `model/mlp.pkl` 的绝对路径
MODEL_PATH = os.path.join(BASE_DIR, "../model/mlp.pkl")

# 确保路径正确
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# 加载模型
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# 计算数据集路径
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/cleaned_diabetes.csv")
df = pd.read_csv(DATASET_PATH)
default_values = df.median(numeric_only=True).to_dict()

# 预定义医疗建议 10 级医疗建议（从健康到糖尿病高风险）
MEDICAL_ADVICE = {
    1: "Your health is in good condition. Maintain a balanced diet and regular exercise.",
    2: "Low risk. It is recommended to have an annual health check-up and maintain a healthy lifestyle.",
    3: "Slight risk. Reduce sugar intake and eat more fiber-rich foods.",
    4: "Moderate risk. Engage in regular physical activity, manage weight, and monitor blood sugar levels.",
    5: "High risk. Consult a doctor and consider taking a glucose tolerance test.",
    6: "Pre-diabetes stage. Follow a strict diet and take preventive measures as advised by your doctor.",
    7: "Significant risk of diabetes. Closely monitor blood sugar changes and consult your doctor.",
    8: "Diabetes is highly likely. Schedule an appointment with a doctor immediately for a comprehensive examination.",
    9: "Diabetes diagnosis risk is extremely high. Immediate medical intervention is required.",
    10: "Diabetes diagnosed. Follow your doctor’s instructions strictly and undergo regular check-ups."
}


# 创建 GUI 应用
class DiabetesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction")

        # 创建一个主Frame用于存放所有字段
        fields_frame = tk.Frame(root)
        fields_frame.grid(row=0, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")

        self.entries = {}
        # 遍历所有字段并创建子Frame，每行6个
        for i, (field, field_type) in enumerate(UserInput.__annotations__.items()):
            description = UserInput.__fields__[field].description

            # 创建子Frame用于单个字段
            subframe = tk.Frame(fields_frame, relief="groove", borderwidth=1, padx=5, pady=5)
            subframe.grid(row=i // 6, column=i % 6, padx=10, pady=10, sticky="nsew")

            # 添加字段标签
            tk.Label(subframe, text=field).pack(anchor="w")
            # 根据字段类型选择控件
            if field in ["Gender", "Polyuria", "Polydipsia", "SuddenWeightLoss", "Weakness", "Polyphagia",
                         "GenitalThrush", "VisualBlurring", "Itching", "Irritability", "DelayedHealing",
                         "PartialParesis", "MuscleStiffness", "Alopecia", "Obesity"]:
                entry = ttk.Combobox(subframe, values=["Male", "Female"] if field == "Gender" else ["Yes", "No"])
            else:
                entry = tk.Entry(subframe)
            entry.pack(fill="x", expand=True)

            # 添加描述标签
            tk.Label(subframe, text=description, fg="gray").pack(anchor="w")

            self.entries[field] = entry

        # 按钮部分放在单独的一行，可以放在主窗口的下方
        buttons_frame = tk.Frame(root)
        buttons_frame.grid(row=1, column=0, columnspan=6, pady=10)

        self.fill_button = tk.Button(buttons_frame, text="Fill Defaults", command=self.fill_defaults)
        self.fill_button.pack(side="left", padx=10)

        self.predict_button = tk.Button(buttons_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side="left", padx=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=2, column=0, columnspan=6, pady=10)

    def fill_defaults(self):
        for field, entry in self.entries.items():
            if isinstance(entry, ttk.Combobox):
                entry.set("Male" if field == "Gender" else "No")
            elif not entry.get():
                default_value = int(default_values[field]) if field in ["Pregnancies", "Age"] else float(
                    default_values[field])
                entry.insert(0, str(default_value))

    def predict(self):
        input_data = {}
        missing_fields = []

        for field, entry in self.entries.items():
            value = entry.get()
            if not value:
                missing_fields.append(field)
                entry.config(bg="red")
            else:
                if isinstance(entry, ttk.Combobox):
                    input_data[field] = value  # 直接存储字符串，符合 Pydantic 预期
                else:
                    try:
                        input_data[field] = int(value) if field in ["Pregnancies", "Age"] else float(value)
                        entry.config(bg="white")
                    except ValueError:
                        messagebox.showerror("Input Error", f"Invalid value for {field}, please enter a number!")
                        return

        if missing_fields:
            messagebox.showwarning("Missing Data", "Missing Data",
                                   "Please fill in all required fields or click 'Fill Defaults'!")
            return

        # 进行预测
        try:
            user_input = UserInput(**input_data)
            input_df = pd.DataFrame([user_input.to_model_input()], columns=df.columns[:-1])
            prediction = model.predict(input_df)[0]
            prediction_prob = model.predict_proba(input_df)[0][1]  # 预测糖尿病概率
            risk_level = min(int(prediction_prob * 10) + 1, 10)  # 计算风险等级（1-10）
            advice = MEDICAL_ADVICE[risk_level]

            result_text = f"Prediction: {'Diabetes' if prediction == 1 else 'Non-Diabetes'}\nLevel: {risk_level}/10\nProbability: {prediction_prob:.2%}\nMedical Suggestion: {advice}"
            self.result_label.config(text=result_text, fg="red" if risk_level > 5 else "green")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")


# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesPredictionApp(root)
    root.mainloop()
