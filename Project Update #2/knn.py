import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#warnings.filterwarnings('ignore')
df_ = pd.read_csv('ProjectUpdate2-Dataset.csv')

type_df = df_.filter(items = ['REPO_TYPE','SECU_FLAG'])
space_df = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'SECU_FLAG'])
time_df = df_.filter(items = ['PRIOR_AGE', 'SECU_FLAG'])

#Space-based
x = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC'])
y = df_.filter(items = ['SECU_FLAG'])
    
x_train, x_test, y_train, y_test = train_test_split(x, y)

#Preprocessing
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

#Results
print("Results Using KNN:") 
cross_score = max(cross_val_score(knn, x_train, y_train, cv=10))
crs = cross_score * 100
print("Cross Validation Score: ")
print(str(round(crs, 3)), '%')
y_pred = knn.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred))  
      