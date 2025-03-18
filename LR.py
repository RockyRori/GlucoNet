# importing the packages
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
# from sklearn.lifrom sklearn import preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score, roc_auc_score
# %matplotlib inline
from sklearn import model_selection

# Model building Logistic Regression
Diabetes = pd.read_csv('new_diabetes_data.csv')  # loading the dataset
X = Diabetes.drop(['class'], axis=1)  # Independent
Y = Diabetes['class']  # Dependent
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X, Y, test_size=0.3, random_state=100)

logmodel = LogisticRegression()
logmodel.fit(X_train_LR, y_train_LR)

LogisticRegression()

predictions_LR = logmodel.predict(X_test_LR)

final_model_predictions_LR = pd.DataFrame({'Actual': y_test_LR, 'predictions': predictions_LR})

print(confusion_matrix(y_test_LR, predictions_LR))
print(classification_report(y_test_LR, predictions_LR))
print(
    "----------------------------------------------------------------------------------------------------------------------")
print(
    "----------------------------------------------------------------------------------------------------------------------")
accuracy_LR = metrics.accuracy_score(final_model_predictions_LR.Actual, final_model_predictions_LR.predictions) * 100
accuracy_LR = '{:.2f}'.format(accuracy_LR)
print('Total Accuracy : ', accuracy_LR)
recall_LR = metrics.recall_score(final_model_predictions_LR.Actual, final_model_predictions_LR.predictions,
                                 average='micro')
print('recall', recall_LR)
Precision_LR = metrics.precision_score(final_model_predictions_LR.Actual, final_model_predictions_LR.predictions,
                                       average='micro')
print('Precision', Precision_LR)
