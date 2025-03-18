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

from sklearn.ensemble import RandomForestClassifier

Diabetes = pd.read_csv('new_diabetes_data.csv')  # loading the dataset
X = Diabetes.drop(['class'], axis=1)  # Independent
Y = Diabetes['class']  # Dependent
X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X, Y, test_size=0.3, random_state=100)

clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train_RF, y_train_RF)
# use the model to make predictions with the test data
y_pred_RF = clf.predict(X_test_RF)

# combining 2 numpy arrays into one pandas dataframe
final_model_predictions_RF = pd.DataFrame({'Actual': y_test_RF, 'predictions': y_pred_RF})

final_model_predictions_RF.head()

y_pred_prob = clf.predict_proba(X_test_RF)  # 2  columns for probability it is creating

y_pred_prob = clf.predict_proba(X_test_RF)[:,
              1]  # The first index refers to the probability that the data belong to class 0, and the second refers to the probability that the data belong to class 1

final_model_predictions_RF['Predicted_prob'] = y_pred_prob

final_model_predictions_RF.head()


def draw_cm(actual, predicted):
    """
    This function draws a confusion matrix.

    Parameters
    ----------
    actual : array-like
        The actual labels.
    predicted : array-like
        The predicted labels.

    Returns
    -------
    None

    """
    # Calculate the confusion matrix
    cm = metrics.confusion_matrix(actual, predicted)

    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=["Default", "No Default"], yticklabels=["Default", "No Default"])

    # Add labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Show the plot
    plt.show()


draw_cm(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions)

accuracy_RF = metrics.accuracy_score(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions) * 100
accuracy_RF = '{:.2f}'.format(accuracy_RF)
print('Total Accuracy : ', accuracy_RF)
recall_RF = metrics.recall_score(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions)
print('recall :', recall_RF)
precision_RF = metrics.precision_score(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions)
print('Precision :', precision_RF)

cm1 = metrics.confusion_matrix(final_model_predictions_RF.Actual, final_model_predictions_RF.predictions)

sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
print('Sensitivity : ', round(sensitivity, 2))

specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
print('Specificity : ', round(specificity, 2))
