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
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

decision_Tree_Classifier = DecisionTreeClassifier(random_state=0)
Diabetes= pd.read_csv('new_diabetes_data.csv') # loading the dataset
X = Diabetes.drop(['class'],axis=1) # Independent
Y = Diabetes['class'] # Dependent
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X, Y, test_size=0.3, random_state=100)

decision_Tree_Classifier.fit(X_train_DT, y_train_DT)

DecisionTreeClassifier(random_state=0)

# predicting a new value

# test the output by changing values, like 3750
y_pred_DT = decision_Tree_Classifier.predict(X_test_DT)

final_model_predictions_DT = pd.DataFrame({'Actual': y_test_DT, 'predictions': y_pred_DT})

final_model_predictions_DT.head()

print(confusion_matrix(y_test_DT, y_pred_DT))
print(classification_report(y_test_DT, y_pred_DT))

accuracy_DT=( metrics.accuracy_score( final_model_predictions_DT.Actual, final_model_predictions_DT.predictions  ))*100
accuracy_DT='{:.2f}'.format(accuracy_DT)
print( 'Total Accuracy : ',accuracy_DT)
recall_DT=metrics.recall_score(final_model_predictions_DT.Actual, final_model_predictions_DT.predictions,average='micro' )
print('recall',recall_DT)
Precision_DT=metrics.precision_score(final_model_predictions_DT.Actual, final_model_predictions_DT.predictions,average='micro' )
print('Precision',Precision_DT)