import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#Function to validate user input and determine the model used
def user_input():
    while True:
        data = input("Please choose which model (type-based, space-based, time-based, or full model): ")
        if data.lower() not in ('type-based', 'space-based', 'time-based', 'full model'):
            print("Not an appropriate choice.")
        else:
            break
    print()
    return data

# Function importing Dataset 
def importDataFrame(data): 

    df_ = pd.read_csv('ProjectUpdate2-Dataset.csv')
    
    if data == 'type-based':
        data_df = df_.filter(items = ['REPO_TYPE','SECU_FLAG'])
    if data == 'space-based':
        data_df = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'SECU_FLAG'])
    if data == 'time-based':
        data_df = df_.filter(items = ['PRIOR_AGE', 'SECU_FLAG'])
    if data == 'full model':
        data_df = df_.filter(items = ['ADD_LOC', 'DEL_LOC', 'TOT_LOC', 'PRIOR_AGE', 'REPO_TYPE','SECU_FLAG']) 

    return data_df 

# Function to split the dataset 
def seperate_dataset(df_, data): 
  
    # Seperating the target variable 
    y = df_.filter(items = ['SECU_FLAG'])
    x = df_.drop(['SECU_FLAG'], axis = 1)

    if(data == 'type-based' or 'full model'):
        x = pd.get_dummies(x)
    
    # Spliting the dataset into train and test 
    x_train, x_test, y_train, y_test = train_test_split(x, y) 
      
    return x, y, x_train, x_test, y_train, y_test 

#Function to validate user input and determine the supervised learning method used
def method_input():
    while True:
        data = input("Which supervised learning algorithm would you like to use? (Naive Bayes, KNN, Decision Tree, ANN, or Random Forest): ")
        if data not in ('Naive Bayes', 'KNN', 'Decision Tree', 'ANN', 'Random Forest'):
            print("Not an appropriate choice.")
        else:
            break
    print()
    return data


#Function that applies the Naive Bayes algorithm
#   User has opportunity to decide which version of Naive Bayes
def naive_bayes(x, y, x_train, x_test, y_train, y_test):
    
    #User determines which version of Naive Bayes
    data = naive_input()
    
    #Initialize
    if data == 'Gaussian':
        nav = GaussianNB()
    if data == 'Multinomial':
        nav = MultinomialNB()
    if data == 'Complement':
        nav = ComplementNB()
    if data == 'Bernoulli':
        nav = BernoulliNB()

    #Perform training
    nav.fit(x_train, y_train)

    print("Results using", data,"Naive Bayes:")
    #Cross Validation
    cross_valid(nav, x_train, y_train)

    #Calculate prediction
    y_pred = nav.predict(x_test)

    #Print the results
    results(y_test, y_pred)

#Function that applies the K-Nearest Neighbor algorithm
def knn(x, y, x_train, x_test, y_train, y_test):
    
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    print("Results using KNN:")
    #Cross Validation
    cross_valid(knn, x_train, y_train)

    #Calculate prediction
    y_pred = knn.predict(x_test)

    #Print the results
    results(y_test, y_pred)

#Function that applies the Decision Tree algorithm
def decision_tree(x, y, x_train, x_test, y_train, y_test):
    data = tree_input()
    
    tree = DecisionTreeClassifier(criterion = data.lower())
    tree.fit(x_train, y_train)

    print("Results using Decision Tree:")
    #Cross Validation
    cross_valid(tree, x_train, y_train)

    #Calculate prediction
    y_pred = tree.predict(x_test)

    #Print the results
    results(y_test, y_pred)

#Function that applies the Artificial Neural Network algorithm
def ann(x, y, x_train, x_test, y_train, y_test):
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)

    print("Results using ANN:")
    #Cross Validation
    cross_valid(mlp, x_train, y_train)

    #Calculate prediction
    y_pred = mlp.predict(x_test)

    #Print the results
    results(y_test, y_pred)

#Function that applies the Random Forest algorithm
def random_forest(x, y, x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(x_train, y_train)

    print("Results using Random Forest:")
    #Cross Validation
    cross_valid(rf, x_train, y_train)

    #Calculate prediction
    y_pred = rf.predict(x_test)

    #Print the results
    results(y_test, y_pred)

#Function for user input to decide which Naive Bayes method to use
def naive_input():
    while True:
        data = input("Please choose which Naive Bayes method (Gaussian, Multinomial, Complementary, or Bernoulli method): ")
        if data not in ('Gaussian', 'Multinomial', 'Complementary', 'Bernoulli'):
            print("Not an appropriate choice.")
        else:
            break
    print()
    return data

#Function for user input to decide which Decision Tree method to use
def tree_input():
    while True:
        data = input("Please choose which Decision Tree method (Gini or Entropy): ")
        if data not in ('Gini', 'Entropy'):
            print("Not an appropriate choice.")
        else:
            break
    print()
    return data

#Function that performs 10-fold cross validation
def cross_valid(object, x_train, y_train):
    cross_score = max(cross_val_score(object, x_train, y_train, cv=10))
    crs = cross_score * 100
    print("Cross Validation Score: ")
    print(str(round(crs, 3)), '%')

    return

# Function to calculate accuracy 
def results(y_test, y_pred): 

    print("Accuracy:")
    acc = accuracy_score(y_test, y_pred) * 100 
    print(str(round(acc, 3)), '%')

    print("Report:") 
    print(classification_report(y_test, y_pred)) 

# Main method 
def main(): 
    warnings.filterwarnings('ignore')  
    
    #User input - determines which model we will be using
    data = user_input()
    
    # Building Phase - split into type-based, space-based, and time-based
    data_df = importDataFrame(data)
    x, y, x_train, x_test, y_train, y_test = seperate_dataset(data_df, data)

    #Preprocessing
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    #Call our different supervised learning algorithms and reports the results
    method = method_input()
    if(method == 'Naive Bayes'):
        naive_bayes(x, y, x_train, x_test, y_train, y_test)
    if(method == 'KNN'):
        knn(x, y, x_train, x_test, y_train, y_test)
    if(method == 'Decision Tree'):
        decision_tree(x, y, x_train, x_test, y_train, y_test)
    if(method == 'ANN'):
        ann(x, y, x_train, x_test, y_train, y_test)
    if(method == 'Random Forest'):
        random_forest(x, y, x_train, x_test, y_train, y_test)
     
# Calling main function 
if __name__=="__main__": 
    main() 