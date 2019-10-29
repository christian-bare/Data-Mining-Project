import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

#warnings.filterwarnings('ignore')
df_ = pd.read_csv('ProjectUpdate2-Dataset.csv')

type_df = df_.filter(items = ['REPO_TYPE','SECU_FLAG'])
space_df = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'SECU_FLAG'])
time_df = df_.filter(items = ['PRIOR_AGE', 'SECU_FLAG'])

#Space-based
x = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC'])
y = df_.filter(items = ['SECU_FLAG'])
    
x_train, x_test, y_train, y_test = train_test_split(x, y)

#Intialize the different methods of Naive Bayes
gau = GaussianNB()
gau.fit(x_train, y_train)

mul = MultinomialNB()
mul.fit(x_train, y_train)

comp = ComplementNB()
comp.fit(x_train, y_train)

bern = BernoulliNB()
bern.fit(x_train, y_train)


#Results
print("Results Using Gaussian Naive Bayes:") 
y_pred_gau = gau.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_gau)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_gau))  
      
#Results
print("Results Using Multinomial Naive Bayes:") 
y_pred_mul = mul.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_mul)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_mul))  
      
#Results
print("Results Using Complement Naive Bayes:") 
y_pred_comp = comp.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_comp)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_comp))  
      
#Results
print("Results Using Bernoulli Naive Bayes:") 
y_pred_bern = bern.predict(x_test) 

print("Accuracy:")
print(accuracy_score(y_test,y_pred_bern)*100) 
      
print("Report :") 
print(classification_report(y_test, y_pred_bern))  
      