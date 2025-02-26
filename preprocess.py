
import os
from sklearn import preprocessing
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn import model_selection
save_dir = "pic"
os.makedirs(save_dir, exist_ok=True)

Diabetes= pd.read_csv('diabetes_data_upload.csv') # loading the dataset

"""
Preparing the Dataset
Checking for missing/null values.
"""
Diabetes.isnull().sum()
Diabetes.isna().sum()
Diabetes.info()

gendis= px.histogram(Diabetes, x = 'Gender', color = 'class', title="Postive/Negative count Vs Gender")
#gendis.show()
gendis.write_image(f"{save_dir}/gender_distribution.png")

"""
pltbl= ['Gender', 'class']
cm = sns.light_palette("green", as_cmap=True)
(round(pd.crosstab(Diabetes[pltbl[0]],Diabetes[pltbl[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)

"""
# 2. Crosstab Heatmap (Gender vs Class)
plt.figure(figsize=(5, 3))
cm = sns.light_palette("green", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['Gender'], Diabetes['class'], normalize='columns') * 100, cmap=cm, annot=True, fmt=".2f")
plt.title("Gender vs Class Heatmap")
plt.savefig(f"{save_dir}/gender_vs_class_heatmap.png")
plt.close()

agehist= px.histogram(Diabetes, x='Age', color="class", title="Distribution of Postive cases with Ages")
#agehist.show()
agehist.write_image(f"{save_dir}/age_distribution.png")

genbox = px.box(Diabetes, y="Age", x="class", color="Gender", points="all", title= "Age Vs Positive/Negative")
#genbox.show()
genbox.write_image(f"{save_dir}/age_vs_class_boxplot.png")

clspi = px.pie(Diabetes, values='Age', names='class', title= "Ratio of Positive and Negative cases")
#clspi.show()
clspi.write_image(f"{save_dir}/class_ratio_pie.png")

print("Count of cases:", Diabetes['class'].value_counts())

polyuria=px.histogram(Diabetes, x = 'Polyuria', color = 'class', title="Polyuria")
#polyuria.show()
polyuria.write_image(f"{save_dir}/polyuria_distribution.png")

"""
plttbl_polyuria= ['Polyuria', 'class']
pm = sns.light_palette("orange", as_cmap=True)
(round(pd.crosstab(Diabetes[plttbl_polyuria[0]], Diabetes[plttbl_polyuria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = pm)

"""
# 7. Crosstab Heatmap (Polyuria vs Class)
plt.figure(figsize=(5, 3))
pm = sns.light_palette("orange", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['Polyuria'], Diabetes['class'], normalize='columns') * 100, cmap=pm, annot=True, fmt=".2f")
plt.title("Polyuria vs Class Heatmap")
plt.savefig(f"{save_dir}/polyuria_vs_class_heatmap.png")
plt.close()

polydispia = px.histogram(Diabetes, x = 'Polydipsia', color = 'class', title="Increased consumption of water")
#polydispia.show()
polydispia.write_image(f"{save_dir}/polydipsia_distribution.png")

"""
plttblpolydispia= ['Polydipsia', 'class']
rm = sns.light_palette("green", as_cmap=True)
(round(pd.crosstab(Diabetes[plttblpolydispia[0]], Diabetes[plttblpolydispia[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = rm)

"""

plt.figure(figsize=(5, 3))
rm = sns.light_palette("green", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['Polydipsia'], Diabetes['class'], normalize='columns') * 100, cmap=rm, annot=True, fmt=".2f")
plt.title("Polydipsia vs Class Heatmap")
plt.savefig(f"{save_dir}/polydipsia_vs_class_heatmap.png")
plt.close()

swl = px.histogram(Diabetes, x = 'sudden weight loss', color = 'class', title="Sudden weight loss")
#swl.show()
swl.write_image(f"{save_dir}/sudden_weight_loss_distribution.png")

"""
plttblswl= ['sudden weight loss', 'class']
qm = sns.light_palette("yellow", as_cmap=True)
(round(pd.crosstab(Diabetes[plttblswl[0]], Diabetes[plttblswl[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = qm)

"""

# 10. Crosstab Heatmap (Sudden Weight Loss vs Class)
plt.figure(figsize=(5, 3))
qm = sns.light_palette("yellow", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['sudden weight loss'], Diabetes['class'], normalize='columns') * 100, cmap=qm, annot=True, fmt=".2f")
plt.title("Sudden Weight Loss vs Class Heatmap")
plt.savefig(f"{save_dir}/swl_vs_class_heatmap.png")
plt.close()

"""
wkns = ['weakness', 'class']
sm = sns.light_palette("green", as_cmap=True)
(round(pd.crosstab(Diabetes[wkns [0]],Diabetes[wkns [1]], normalize='columns') * 100,2)).style.background_gradient(cmap = sm)

"""

plt.figure(figsize=(5, 3))
sm = sns.light_palette("green", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['weakness'], Diabetes['class'], normalize='columns') * 100, cmap=sm, annot=True, fmt=".2f")
plt.title("Weakness vs Class Heatmap")
plt.savefig(f"{save_dir}/weakness_heatmap.png")
plt.close()

eating = px.histogram(Diabetes, x = 'Polyphagia', color = 'class', title="Excessive eating")
#eating.show()
eating.write_image(f"{save_dir}/polyphagia_distribution.png")
"""
plt_eating= ['Polyphagia', 'class']
tm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(Diabetes[plt_eating[0]], Diabetes[plt_eating[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = tm)

"""

plt.figure(figsize=(5, 3))
tm = sns.light_palette("red", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['Polyphagia'], Diabetes['class'], normalize='columns') * 100, cmap=tm, annot=True, fmt=".2f")
plt.title("Polyphagia vs Class Heatmap")
plt.savefig(f"{save_dir}/polyphagia_heatmap.png")
plt.close()

gntlthrsh = px.histogram(Diabetes, x = 'Genital thrush',color='class')
#gntlthrsh.show()
gntlthrsh.write_image(f"{save_dir}/genital_thrush_distribution.png")
"""
plt_thrsh= ['Genital thrush', 'class']
um = sns.light_palette("pink", as_cmap=True)
(round(pd.crosstab(Diabetes[plt_thrsh[0]], Diabetes[plt_thrsh[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = um)

"""
plt.figure(figsize=(5, 3))
um = sns.light_palette("pink", as_cmap=True)
sns.heatmap(pd.crosstab(Diabetes['Genital thrush'], Diabetes['class'], normalize='columns') * 100, cmap=um, annot=True, fmt=".2f")
plt.title("Polyphagia vs Class Heatmap")
plt.savefig(f"{save_dir}/polyphagia_heatmap.png")
plt.close()

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





