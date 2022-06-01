# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:58:18 2021

@author: tarek
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() 

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error, r2_score

# Read a dataset
df=pd.read_csv("Drug_10.csv")



# Drop Columns
df.drop('c_sy', axis=1, inplace=True)
df.drop('c_ana', axis=1, inplace=True)
df.drop('c_ad', axis=1, inplace=True)

# Drop Rows containg Null Values
df.dropna(axis=0, inplace=True)


#############################################################################
# Encoding categorical data
# First determine the columns that contaian a categorical data
col=['Drug1','Drug2','Cell line']
le = preprocessing.LabelEncoder()
for c in col:        
       df[c]= pd.DataFrame(le.fit_transform(df[c]))
       le_name=dict(zip(le.classes_,le.transform(le.classes_)))
       print (c)
       print (le_name)
       print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
       
#############################################################################
# Encoding Class names

df['classs']= df['classs'].map({'additive': 0, 'Synerg' : 1,'antag' : 2})
print (df['classs'].value_counts())

#    correlation martix between all columns in Dataframe and CSS column (y variable)
print (df.corr()['CSS'])
# Get a corrMatrix as a variable to get data as a figure
corrMatrix = df.corr()

#############################################################################

# Read X and y variables
X=df[['Drug1','Drug2','Cell line',"S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum','CSS']]
y=df['classs']

# To Avoid ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
X = np.nan_to_num(X)

#############################################################################


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#############################################################################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#############################################################################

print ("========================== Randon Forest Classification ==========================")
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################

print ("========================== DecisionTreeClassifier  ==========================")

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################
print ("========================== Naive Bayes Classification  ==========================")


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################

print ("========================== LogisticRegression Classification  ==========================")

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear',random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################

print ("========================== KNeighborsClassifier Classification  ==========================")

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################

print ("========================== SVC Classification  ==========================")


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))

# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))

# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

#############################################################################


plt.plot(X,y)
df.hist()
plt.show(block=True)
