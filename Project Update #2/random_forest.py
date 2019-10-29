#Data Mining - Project Update #2 (Random Forest)
#Christian Bare and Zach Kellerman
import warnings
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

warnings.filterwarnings('ignore')
df = pd.read_csv('ProjectUpdate2-Dataset.csv')

#This is Random Forest for the type-based model
type_model = df[['REPO_TYPE']].copy()
type_dep_var = np.array(df['SECU_FLAG'])
type_model = pd.get_dummies(type_model)
type_dep_var = pd.get_dummies(type_dep_var)

type_col_names = list(type_model.columns)
type_model = np.array(type_model)

type_train, type_test, type_dep_train, type_dep_test = train_test_split(type_model, type_dep_var)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(type_train, type_dep_train)
predictions = rf.predict(type_test)
print("Accuracy for type-based model:",accuracy_score(type_dep_test, predictions))
print(classification_report(type_dep_test, predictions))

#This is Random Forest for the size-based model
size_model = df[['ADD_LOC','DEL_LOC','TOT_LOC']].copy()
size_dep_var = np.array(df['SECU_FLAG'])
size_dep_var = pd.get_dummies(size_dep_var)

size_col_names = list(size_model.columns)
size_model = np.array(size_model)

size_train, size_test, size_dep_train, size_dep_test = train_test_split(size_model, size_dep_var)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(size_train, size_dep_train)
predictions = rf.predict(size_test)
print("Accuracy for size-based model:",accuracy_score(size_dep_test, predictions))
print(classification_report(size_dep_test, predictions))

#This is Random Forest for the time-based model
time_model = df[['PRIOR_AGE']].copy()
time_dep_var = np.array(df['SECU_FLAG'])
time_dep_var = pd.get_dummies(time_dep_var)

time_col_names = list(time_model.columns)
time_model = np.array(time_model)

time_train, time_test, time_dep_train, time_dep_test = train_test_split(time_model, time_dep_var)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(time_train, time_dep_train)
predictions = rf.predict(time_test)
print("Accuracy for time-based model:",accuracy_score(time_dep_test, predictions))
print(classification_report(time_dep_test, predictions))

#This is Random Forest for the full model
full_model = df[['ADD_LOC','DEL_LOC','TOT_LOC','PRIOR_AGE','REPO_TYPE']].copy()
full_dep_var = np.array(df['SECU_FLAG'])
full_model = pd.get_dummies(full_model)
full_dep_var = pd.get_dummies(full_dep_var)

full_col_names = list(full_model.columns)
full_model = np.array(full_model)

full_train, full_test, full_dep_train, full_dep_test = train_test_split(full_model, full_dep_var)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(full_train, full_dep_train)
predictions = rf.predict(full_test)
print("Accuracy for the full model:",accuracy_score(full_dep_test, predictions))
print(classification_report(full_dep_test, predictions))
