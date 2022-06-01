# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:10:50 2021

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

print("=================== LinearRegression =======================")

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


#Accuracy score is only for classification problems. For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).
print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))

###############################################################################

print("=================== Random Forest Regressor =======================")
# Fitting Multiple Linear Regression to the Training set
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Accuracy score is only for classification problems. For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).
print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))

###############################################################################
print("=================== Decision Tree Regressor =======================")

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Accuracy score is only for classification problems. For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).
print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))

###############################################################################

print("=================== Bayesian Ridge Regressor =======================")

from sklearn.linear_model import Ridge, BayesianRidge, Lasso
regressor = BayesianRidge()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Accuracy score is only for classification problems. For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).
print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))

###############################################################################
print("=================== KNeighborsRegressor =======================")

from sklearn.neighbors import KNeighborsRegressor

# KNeighborsRegressor  
regressor = KNeighborsRegressor(n_neighbors = 8, weights='uniform',algorithm = 'auto')   
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Test for Overfitting and underfitting 
print('Train Score is : ' , regressor.score(X_train, y_train))
print('Test Score is : ' , regressor.score(X_test, y_test))


#Accuracy score is only for classification problems. For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).
print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print("R^2 = %0.5f" % r2_score(y_test, y_pred))
# The mean squared error
print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))

###############################################################################
