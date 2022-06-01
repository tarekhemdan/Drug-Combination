import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() 

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

import warnings
warnings.filterwarnings('ignore')

def analysis(data,col):
    
    print(data.info())
    print (data.describe())
    
    print (data.isnull().sum())
    print ("shape before delete null values")
    print(data.shape)
    
    for c in col:
        data[c].replace("#N/A",np.nan,inplace=True)
        
    data.dropna(inplace=True)
    print ("shape after delete null values")
    print(data.shape)
    print (data.head(10))
    
def encode(data,col):
    le = preprocessing.LabelEncoder()
    for c in col:
           
       data[c]= pd.DataFrame(le.fit_transform(data[c]))
       le_name=dict(zip(le.classes_,le.transform(le.classes_)))
       print (c)
       print (le_name)
       print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
       
    return data

def split_data(data):


    X = data.drop("CSS",axis=1)
    y = data["CSS"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=100)
    return x_train, x_test, y_train, y_test
    
    
def main_func():      
   
#   read data  
    data= pd.read_csv("Drug_5.csv")
    cc=['Drug1','Drug2','Cell line']
    
    data=encode(data,cc)
    
    col=['Drug1','Drug2','Cell line',"CSS","S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum',"classs"]     
    data1= pd.DataFrame(data,columns=col)    
    
    analysis(data1,col)
    
    data1['classs']= data1['classs'].map({'additive': 0, 'Synerg' : 1,'antag' : 2})
    
    
    print (data1.classs.value_counts())

#    correlation martix
    print (data1.corr()['CSS'])
    
    corrMatrix = data1.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()
   
    col1=["S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum','CSS']
    data11= pd.DataFrame(data1,columns=col1)
    analysis(data11,col1)
    
#     splti data
    
    train_df,test_df=train_test_split(data11,test_size=0.30,random_state=100)
    min_max=preprocessing.MinMaxScaler()


    for c in col1:
        train_df[c]=min_max.fit_transform(np.array(train_df[c]).reshape(-1,1))
     
    
    for c in col1:
        test_df[c]=min_max.fit_transform(np.array(test_df[c]).reshape(-1,1))
    
    test_df1= test_df.drop('CSS',axis=1)
    print (test_df1)

    x_train, x_test, y_train, y_test = split_data(train_df)
    
   
    
    n_neighbors=5
    models=[]
    models.append(('linear', LinearRegression()))
    models.append(('Lasso', Lasso(alpha=1.0)))
    models.append(('RF_Regressor', RandomForestRegressor(n_jobs=-1,random_state=42)))
    models.append(('BayesianRidge', linear_model.BayesianRidge()))
    models.append(('DecisionTreeRegressor', tree.DecisionTreeRegressor(max_depth=3)))
    models.append(('Ridge', linear_model.Ridge()))
    models.append(('KNeighborsRegressor',neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')))
    
    print  (x_test.shape)
    print  (y_test.shape)
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error, r2_score
    
    for name, model in models:
        #          train models
        print("++++++++++++++++++++++", name,"++++++++++++++++++++++")
        model.fit(x_train,y_train)
        y_pred= model.predict(x_test)

           
        print("MAE = %5.3f" % mean_absolute_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print("R^2 = %0.5f" % r2_score(y_test, y_pred))
        # The mean squared error
        print("MSE = %5.3f" % mean_squared_error(y_test, y_pred))
        

        
main_func()  