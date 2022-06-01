# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 00:03:03 2021

@author: tarek
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, NuSVC , LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

sns.set() 

import warnings
warnings.filterwarnings('ignore')

#############################################################################

import pandas as pd
df = pd.read_csv(r"Drug_10.csv",sep=',')


# Drop Columns
df.drop('c_sy', axis=1, inplace=True)
df.drop('c_ana', axis=1, inplace=True)
df.drop('c_ad', axis=1, inplace=True)

# Drop Rows containg Null Values
df.dropna(axis=0, inplace=True)

#############################################################################

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Drug1'] = labelencoder.fit_transform(df['Drug1'])
df['Drug2'] = labelencoder.fit_transform(df['Drug2'])
df['Cell line'] = labelencoder.fit_transform(df['Cell line'])

#df['classs'] = labelencoder.fit_transform(df['classs'])
df['classs']= df['classs'].map({'additive': 0, 'Synerg' : 1,'antag' : 2})
print (df['classs'].value_counts())
#############################################################################

X=df[['Drug1','Drug2','Cell line',"S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum','CSS']]
y=df['classs']
#############################################################################

#    correlation martix
data1=df[['Drug1','Drug2','Cell line',"S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum','CSS','classs']]
print (data1.corr()['CSS'])
    
corrMatrix = data1.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
 #############################################################################  
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)
#############################################################################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#############################################################################

from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("================",classifier,"================================")
# Precision Score = TP / (FP + TP)
print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))
# Recall Score = TP / (FN + TP)
print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))
# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
print('Accuracy: ' , accuracy_score(y_test, y_pred))
# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))

cl=[LogisticRegression,LinearSVC,KNeighborsClassifier,RandomForestClassifier,
    RidgeClassifier,Perceptron,ComplementNB,NearestCentroid,MultinomialNB,
    BernoulliNB,SGDClassifier,SVC, DecisionTreeClassifier]

for classifier in cl:
    m=classifier()
    m.fit(X_train,y_train)
    y_pred=m.predict(X_test)
    print("=================",m, "=========================")
    #print(accuracy_score(y_test,predictions))
    # Precision Score = TP / (FP + TP)
    print('Precision: ' , precision_score(y_test, y_pred, pos_label='positive' , average='macro'))
    # Recall Score = TP / (FN + TP)
    print('Recall: ' , recall_score(y_test, y_pred, pos_label='positive' , average='macro'))
    # Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
    print('Accuracy: ' , accuracy_score(y_test, y_pred))
    # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
    print('F1 Score: ' , f1_score(y_test, y_pred, pos_label='positive' , average='macro'))
#############################################################################


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

pca = PCA(n_components=2)

# Maybe some original features were good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
print("Combined space has", X_features.shape[1], "features")

svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
#############################################################################
print("============ Deep Learning 1 ==============================")

# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

"""
# load dataset
dataframe = pandas.read_csv("iris.data", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
""" 
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=9, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
estimator = KerasClassifier(build_fn= baseline_model, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=4, shuffle=True)

results = cross_val_score(estimator, X, y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#############################################################################

print("============ Deep Learning 2 ==============================")

# multi-class classification with Keras
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=9, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

"""
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
"""

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [10, 20, 30]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
    
    