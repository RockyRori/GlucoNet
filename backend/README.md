# **📌 项目 README（`README.md`）**

## **🎯 项目简介**

GlucoNet是一个 **糖尿病风险预测系统**，包含 **本地 GUI 应用** 和 **FastAPI 后端接口**，基于机器学习模型（MLP）预测糖尿病风险，并提供个性化医疗建议。

---

## **📌 目录结构**

```bash
|---backend
|   clean_data.py
|   illustrate_data.py
|   main.py
|   model_evaluation.py
|   README.md
|   requirements.txt
|
+---dataset
|       cleaned_diabetes.csv
|       diabetes_1.csv
|       diabetes_2.csv
|       diabetes_3.csv
|       merged_diabetes.csv
|
+---model
|       logistic_regression.pkl
|       mlp.pkl
|       random_forest.pkl
|       svm.pkl
|       train_logistic_regression.py
|       train_mlp.py
|       train_random_forest.py
|       train_svm.py
|       train_xgboost.py
|       xgboost.pkl
|
+---output
|       accuracy_comparison.png
|       correlation_heatmap.png
|       histograms.png
|       model_evaluation.csv
|       outcome_distribution.png
|       recall_comparison.png
|
\---program
        data_model.py
        local_app.py
        remote_app.py
```

---

## **🖥️ 1. 本地 GUI 应用**

本地应用 **`local_app.py`** 提供 **图形界面**，允许用户输入数据并获取糖尿病预测结果。  
✅ **31 个特征输入框，包含下拉框选择项**  
✅ **一键填充默认值（基于数据集计算）**  
✅ **点击“预测”按钮，返回风险等级、置信度和医疗建议**

### **📌 运行方式**

```bash
cd backend/program
python local_app.py
```

---

## **🔧 2. 安装依赖**

**确保已安装 Python 3.8+，然后运行以下命令：**

```bash
pip install -r requirements.txt
```

📌 **`requirements.txt` 主要依赖**

- `fastapi` + `uvicorn`（后端 API 运行）
- `pandas` + `numpy`（数据处理）
- `scikit-learn`（机器学习模型）
- `xgboost`（可扩展）
- `tkinter`（GUI 界面）

---

## **🚀 3. 项目特点**

✅ **基于 `MLPClassifier` 的糖尿病预测模型**  
✅ **支持 `GUI 应用` 和 `REST API` 方式访问**  
✅ **31 个特征输入，数据自动填充默认值**  
✅ **CORS 兼容，方便与前端集成**  
✅ **医疗建议分 `10 级`，根据风险等级智能推荐**  


