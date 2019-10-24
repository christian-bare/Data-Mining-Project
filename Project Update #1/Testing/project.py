#Data Mining - Project
#Christian Bare and Zach Kellerman

import pandas as pd  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df_ = pd.read_csv('LOCKED_FINAL_CSC4220_5220_DATASET.csv')

#This command filters out any file with a file extension and stores it into a dataframe
fileext_df = df_[(df_['MODIFIED_FILE'].str.contains('\.[a-z]')==True)]

#Here we create a DataFrame which contains the Neutral Security Flagged files
neutral_df = fileext_df[fileext_df['SECU_FLAG']=='NEUTRAL']

#Here we create a DataFrame which contains the Insecure Security Flagged files
insecure_df = fileext_df[fileext_df['SECU_FLAG']=='INSECURE']

#Here we apply TF-IDF to the two groups using the SKlearn

tfidf_neu = TfidfVectorizer()
#tfidf_ins = TfidfVectorizer()

vocabulary = tfidf_neu.fit((neutral_df['MODIFIED_FILE']))

print(tfidf_neu.get_feature_names())

#tfidf_neu.fit()

#response_neu = tfidf_neu.fit_transform(neu_dict)

#neu_feature_names = tfidf_neu.get_feature_names()

#ins_dict = {}

#index =0
#for index in insecure_df.iterrows():
#    ins_dict[index] = df_['MODIFIED_FILE']

#response_neu = tfidf_ins.fit_transform(ins_dict)

#ins_feature_names = tfidf_ins.get_feature_names()


##After several attempts at the SKlearn library, we were unable to implement TfidfVectorizer()

#Seperate implementation
#Here we save each file name as a key and the frequency as the value
#   The for loop iterates over each row, counting the occurances of each file name
#   

#neu_dict = {}
#neu_name = ''
#for index in neutral_df.iterrows():
#    neu_name = df_['Name']
  #  neu_dict[neu_name] = neu_dict.get(neu_name, 0)+1
