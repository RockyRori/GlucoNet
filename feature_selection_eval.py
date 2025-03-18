#importing the packages
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
#from sklearn.lifrom sklearn import preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score,confusion_matrix,recall_score,roc_auc_score
#%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_selection import RFE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


#a.	Recursive Feature Elimination (RFE) – Wrapper Method
def best_model_rfe():
    model = LogisticRegression()
    best_feature={}
    

    for n in range (5,17):
        rfe = RFE(model, n_features_to_select=n) 
        X_selected = rfe.fit_transform(X, Y)
        selected_features = X.columns[rfe.support_]
        print("Selected features:", selected_features)
        X_train, X_test, y_train, y_test = train_test_split( X[selected_features], Y, test_size = 0.2, random_state = 100)
        acc=test_model(X_train, X_test, y_train, y_test)
        best_feature[n]={tuple(selected_features):acc}#model2 最高acc的features
    return best_feature

#b.	Feature Importance – Embedded Method

def best_model_emb():
    model = RandomForestClassifier()
    model.fit(X, Y)
    best_feature={}
    

    for n in range (5,17):
       # 获取特征重要性
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        selected_features = feature_importances.sort_values(ascending=False).head(n).index
        print("Selected features:", selected_features)
        X_train, X_test, y_train, y_test = train_test_split( X[selected_features], Y, test_size = 0.2, random_state = 100)
        acc=test_model(X_train, X_test, y_train, y_test)
        best_feature[n]={tuple(selected_features):acc}#model2 最高acc的features
    return best_feature



def eval(final_model_predictions):


    # 计算混淆矩阵
    print(confusion_matrix(y_test, final_model_predictions.predictions))
    print(classification_report(y_test, final_model_predictions.predictions))
    print("----------------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------------")

    # 计算 Accuracy
    accuracy_LR = accuracy_score(final_model_predictions.Actual, final_model_predictions.predictions) * 100
    accuracy_LR = '{:.2f}'.format(accuracy_LR)
    print('Total Accuracy : ', accuracy_LR)

    # 计算 Recall
    recall_LR = recall_score(final_model_predictions.Actual, final_model_predictions.predictions, average='micro')
    print('Recall:', recall_LR)

    # 计算 Precision
    precision_LR = precision_score(final_model_predictions.Actual, final_model_predictions.predictions, average='micro')
    print('Precision:', precision_LR)
    return accuracy_LR

def curve(model):
    # 计算 AUC-ROC
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('AUC-ROC:', auc_roc)
    model_name=model.__class__.__name__

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} (AUC = {auc_roc_LR:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def test_model(X_train, X_test, y_train, y_test):
   
    # Logistic Regression


    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)

    LogisticRegression()

    predictions_LR = logmodel.predict(X_test)

    final_model_predictions_LR = pd.DataFrame({'Actual':y_test, 'predictions':predictions_LR})
    eval(final_model_predictions_LR)
    #curve(logmodel)

    #XBG


    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train, y_train)

    y_pred_XGB = xgb_clf.predict(X_test)

    final_model_predictions_XGB = pd.DataFrame({'Actual': y_test, 'predictions': y_pred_XGB})
    acc_XGB=eval(final_model_predictions_XGB)
    #curve(xgb_clf)
    

    # Neural Network
    model_nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # 二分类任务
    ])
    model_nn.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练神经网络
    model_nn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # 预测并转换为 1D 数组
    y_pred_nn = (model_nn.predict(X_test) > 0.5).astype(int).ravel()

    # 创建 DataFrame
    final_model_predictions_NN = pd.DataFrame({'Actual': y_test, 'predictions': y_pred_nn})

    # 计算准确率
    acc_nn = accuracy_score(y_test, y_pred_nn)
    eval(final_model_predictions_NN)
    return acc_nn
   


def plot_roc_curves(X_train, X_test, y_train, y_test):
    # Logistic Regression
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    y_pred_proba_LR = logmodel.predict_proba(X_test)[:, 1]  # 获取预测概率
    fpr_LR, tpr_LR, _ = roc_curve(y_test, y_pred_proba_LR)
    auc_LR = auc(fpr_LR, tpr_LR)

    # XGBoost
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred_proba_XGB = xgb_clf.predict_proba(X_test)[:, 1]  # 获取预测概率
    fpr_XGB, tpr_XGB, _ = roc_curve(y_test, y_pred_proba_XGB)
    auc_XGB = auc(fpr_XGB, tpr_XGB)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_LR, tpr_LR, color='blue', label=f'Logistic Regression (AUC = {auc_LR:.2f})')
    plt.plot(fpr_XGB, tpr_XGB, color='red', label=f'XGBoost (AUC = {auc_XGB:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # 随机分类器的参考线

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


Diabetes = pd.read_csv('new_diabetes_data.csv')  # loading the dataset
X = Diabetes.drop(['class'], axis=1)  # Independent
Y = Diabetes['class']  # Dependent
#选vairables
selected_features = ['Polyuria',
   'Polydipsia',
   'Gender',
   'Age',
   'sudden weight loss',
   'partial paresis',
   'Alopecia',
   'Irritability',
   'delayed healing',
   'Polyphagia']

X_train, X_test, y_train, y_test = train_test_split( X[selected_features], Y, test_size = 0.2, random_state = 100)

best_model_rfe()
best_model_emb()
test_model(X_train, X_test, y_train, y_test)
plot_roc_curves(X_train, X_test, y_train, y_test)