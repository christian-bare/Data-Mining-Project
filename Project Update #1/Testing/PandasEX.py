import pandas as pd  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

lis = [(1, 2, 'Student1', 20), (3, 4, 'Student2', 21), (9, 10, 'Student3', 31), (9, 10, 'Student4', 30), (9, 10, 'Student5', 35)]  

df_=pd.DataFrame(lis, columns=['Index1', 'Index2', 'Name', 'Age']) 

print(df_.tail())

special_df = df_[df_['Age'] > 35] 

print('*'*25)  

print(special_df.head()) 

print('*'*25)  

r_, c_ = special_df.shape  

print('Rows:{}, Cols:{}'.format(r_, c_))  

student1_df =df_[df_['Name']=='Student1'] 

print('*'*15)  

print(student1_df.head())  

print('*'*15)  

age_of_students = df_['Age'].tolist() 

print(age_of_students)


#Test the Pandas functions for the first project update

print('*'*15)  
print('*'*15)  

name_df = df_[(~df_['Name'].str.contains('1')==True) &
            (~df_['Name'].str.contains('2')==True)]

print(name_df.head())

print('*'*15)  
print('*'*15) 


print(df_.head())

dict = {}
name = ''
index = 0
for index in df_.iterrows():
    name = df_['Name']
    dict[name] = dict.get(name, 0)+1



#tfidf = TfidfVectorizer()

#response = tfidf.fit_transform(df_)

#feature_names = tfidf.get_feature_names()
#for col in response.nonzero()[1]:
#    print(feature_names[col], ' - ', response[0,col])


