

from sklearn import preprocessing
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




Diabetes= pd.read_csv('diabetes_data_upload.csv') # loading the dataset

"""
Preparing the Dataset
Checking for missing/null values.
"""
Diabetes.isnull().sum()
Diabetes.isna().sum()
Diabetes.info()


"""
Data Processing
Encoding the categorical variable to nurmeric values (0's & 1's)
"""

number = preprocessing.LabelEncoder()

dtacpy1 = Diabetes.copy()   # Duplicating the Dataset

#Encoding the categorical variable to nurmeric values (0's & 1's)
for i in dtacpy1:
    dtacpy1[i] = number.fit_transform(dtacpy1[i])

dtacpy1.head()

# Setting target variable
X = dtacpy1.drop(['class'],axis=1) # Independent
Y = dtacpy1['class'] # Dependent

"""
1.Identify Key Features and Risk Factors (4 points)
oClearly identify and justify key features and risk factors based on data analysis and literature review.

We can see from the above graph that "Class" and the following factors have a strong correlation.

The attributes listed below are favorably correlated with the most closely related variables listed first.

    Ployuria
    Polydipsia
    Sudden weight loss
    partial paresis

Negatively correlated variables are not very significant. Meaning that the likelihood of the patient having diabetes is very minimal if you have tested positive for alopecia.

"""
correlation = X.corrwith(Y)

print(correlation)

correlation.plot.bar(title="Correlation with target variable class", grid=True, figsize=(15,5))

#Data Normalization

min_max = MinMaxScaler()
X[['Age']] = min_max.fit_transform(X[['Age']])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify= Y, random_state = 1000)

## checking for the distribution of traget variable in train test split
print('Distribution of traget variable in training dataset')
print(Y_train.value_counts())

print('Distribution of traget variable in test dataset')
print(Y_test.value_counts())





# new DataFrame
new_data = pd.concat([X, Y], axis=1)

# save the new dataset
new_data.to_csv('new_diabetes_data.csv', index=False)





