import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
  
#warnings.filterwarnings('ignore')
df_ = pd.read_csv('ProjectUpdate2-Dataset.csv')

type_df = df_.filter(items = ['REPO_TYPE','SECU_FLAG'])
space_df = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'SECU_FLAG'])
time_df = df_.filter(items = ['PRIOR_AGE', 'SECU_FLAG'])

#Space-based
x = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC'])
y = df_.filter(items = ['SECU_FLAG'])
    
x_train, x_test, y_train, y_test = train_test_split(x, y)

#Training
trained_gini = DecisionTreeClassifier(criterion = "gini")
trained_gini.fit(x_train, y_train)
trained_entropy = DecisionTreeClassifier(criterion = "entropy")
trained_entropy.fit(x_train, y_train)

#Results
print("Results Using Gini Index:") 
y_pred_gini = trained_gini.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_gini)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_gini))  
      
print("Results Using Entropy:") 
y_pred_entropy = trained_entropy.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_entropy)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_entropy))