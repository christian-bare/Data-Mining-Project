#Data Mining - Project Update #2 (ANN)
#Christian Bare and Zach Kellerman

import warnings
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
df = pd.read_csv('ProjectUpdate2-Dataset.csv')

#This is ANN for the type-based model
type_model = df[['REPO_TYPE']].copy()
type_dep_var = df['SECU_FLAG']

type_model = pd.get_dummies(type_model)

type_train, type_test, type_dep_train, type_dep_test = train_test_split(type_model, type_dep_var)

scaler = StandardScaler()
scaler.fit(type_train)
type_train = scaler.transform(type_train)
type_test = scaler.transform(type_test)

mlp = MLPClassifier()
mlp.fit(type_train, type_dep_train)

predictions = mlp.predict(type_test)
print("This is the classification report for the type-based model:")
print(classification_report(type_dep_test,predictions))

#This is ANN for the size-based model
size_model = df[['ADD_LOC','DEL_LOC','TOT_LOC']].copy()
size_dep_var = df['SECU_FLAG']

size_train, size_test, size_dep_train, size_dep_test = train_test_split(size_model, size_dep_var)

scaler = StandardScaler()
scaler.fit(size_train)
size_train = scaler.transform(size_train)
size_test = scaler.transform(size_test)

mlp = MLPClassifier()
mlp.fit(size_train, size_dep_train)

predictions = mlp.predict(size_test)
print("This is the classification report for the size-based model:")
print(classification_report(size_dep_test,predictions))

#This is ANN for the time-based model
time_model = df[['PRIOR_AGE']].copy()
time_dep_var = df['SECU_FLAG']

time_train, time_test, time_dep_train, time_dep_test = train_test_split(time_model, time_dep_var)

scaler = StandardScaler()
scaler.fit(time_train)
time_train = scaler.transform(time_train)
time_test = scaler.transform(time_test)

mlp = MLPClassifier()
mlp.fit(time_train, time_dep_train)

predictions = mlp.predict(time_test)
print("This is the classification report for the time-based model:")
print(classification_report(time_dep_test,predictions))

#This is ANN for the full model
full_model = df[['ADD_LOC','DEL_LOC','TOT_LOC','PRIOR_AGE','REPO_TYPE']]
full_dep_var = df['SECU_FLAG']

full_model = pd.get_dummies(full_model)

full_train, full_test, full_dep_train, full_dep_test = train_test_split(full_model, full_dep_var)

scaler = StandardScaler()
scaler.fit(full_train)
full_train = scaler.transform(full_train)
full_test = scaler.transform(full_test)

mlp = MLPClassifier()
mlp.fit(full_train, full_dep_train)

predictions = mlp.predict(full_test)
print("This is the classification report for the full model:")
print(classification_report(full_dep_test,predictions))