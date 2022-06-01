import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() 
import warnings
warnings.filterwarnings('ignore')


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from numpy import mean
from sklearn import preprocessing
from sklearn import metrics
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids


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
    
def encodedata(data):
    cc=['Drug1','Drug2','Cell line']
    
    le = preprocessing.LabelEncoder()
    
    for c in cc:
           
       data[c]= pd.DataFrame(le.fit_transform(data[c]))
       le_name=dict(zip(le.classes_,le.transform(le.classes_)))
       print (c)
       print (le_name)
       print (data)
    return data

    
    
def classification(x,y):
    print ("=============== classifiction Starts ===============================")
    kfold=KFold(n_splits=3,random_state=0,shuffle=True)
     
    models = []
    
    models.append(('naive', GaussianNB()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('LR',LogisticRegression()))
    print("************************************************************")
    
    for name, model in models:
          
            print ("............",name)
    
            acc=cross_val_score(model,x,y,scoring= 'accuracy',cv=kfold)
            print("*************************************Acc ***********************")
            
            r=cross_val_score(model,x,y,scoring= make_scorer(recall_score,average='macro'),cv=kfold)
            pr=cross_val_score(model,x,y,scoring= make_scorer(precision_score,average='macro'),cv=kfold)
            f1=cross_val_score(model,x,y,scoring= make_scorer(f1_score,average='macro'),cv=kfold)

     
            msg1="%s:%f   %s:%f    %s:%f    %s:%f " % ('accuracy',(mean(acc)*100), 'precision',(mean(pr)*100),'recall',(mean(r)*100), 'f1',(mean(f1)*100))
            
            print (msg1)
            print ("=============== classifiction Ends ===============================")
            
         
        
def balance_data(x,y,method):
    
    from imblearn.under_sampling import TomekLinks    
    if(method=='tomeklinks'):
        tl = TomekLinks(return_indices=True, ratio='majority')
        print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")  
        x_res, y_res,ind = tl.fit_sample(x,y)
        print (x_res.shape,y_res.shape)
        print ("x_res=",x_res)
        
        
    if(method=='NearMiss'):
        
        nm = NearMiss()
        x_res, y_res=nm.fit_sample(x,y)
        print (x_res.shape,y_res.shape)
        

        
    return x_res, y_res


def main_func():      
   
#   read data  
    data= pd.read_csv("Drug_10.csv")
    
    data=encodedata(data)
    
    col=['Drug1','Drug2','Cell line',"CSS","S_HSA","S_Bliss","S_Loewe","S_ZIP","classs"]
   
    col1=["CSS","S_HSA","S_Bliss","S_Loewe","S_ZIP",'S_sum',"classs"]

    data1= pd.DataFrame(data,columns=col1)
 
    
    print ("++++++++++++++++++++++++++++data_5++++++++++++++++++++++++++++")
    analysis(data1,col1)
    
    data1['classs']= data1['classs'].map({'additive': 0, 'Synerg' : 1,'antag' : 2})
    data1=data1.query('classs !=0')
  

    print (data1.classs.value_counts())


    
# normalize data
    print ("=============== Normalize Starts ===============================")
    data1=data1.values
    x,y=data1[:,:-1],data1[:,-1]
    min_max=preprocessing.MinMaxScaler()
    x_scaled=min_max.fit_transform(x)
    print ("===============x_scaled= ===============================", x_scaled)
    
    
 # make classifiction using imbalanced data   
    classification(x_scaled, y)
    print ("=============== classifiction using imbalanced data Ends ===============================")
    
    
# make classifiction using balanced data
    
    method=['tomeklinks','NearMiss']
    for m in method:
        print ("==============================================")
        print (m)
        x_res, y_res=balance_data(x_scaled,y,m)
        classification(x_res, y_res)
        print ("x_res=",x_res)
        print ("y_res=",y_res)
main_func()