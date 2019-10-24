#Data Mining - Project Update #1
#Christian Bare and Zach Kellerman

import pandas as pd  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df_ = pd.read_csv('LOCKED_FINAL_CSC4220_5220_DATASET.csv')

#This command filters out any file with a file extension and stores it into a dataframe
fileext_df = df_[(df_['MODIFIED_FILE'].str.contains('\.[a-z]')==True)]
fileext_df = fileext_df[['MODIFIED_FILE', 'SECU_FLAG']].copy()

#Here we create a DataFrame which contains the Neutral Security Flagged files
neutral_df = fileext_df[fileext_df['SECU_FLAG']=='NEUTRAL']

#Here we create a DataFrame which contains the Insecure Security Flagged files
insecure_df = fileext_df[fileext_df['SECU_FLAG']=='INSECURE']

#Here we apply TF-IDF to the two groups using the SKlearn

tfidf_neu = TfidfVectorizer()
response_neu = tfidf_neu.fit_transform(neutral_df['MODIFIED_FILE'])
neu_feature_names = tfidf_neu.get_feature_names()

tfidf_ins = TfidfVectorizer()
response_ins = tfidf_ins.fit_transform(insecure_df['MODIFIED_FILE'])
ins_feature_names = tfidf_ins.get_feature_names()

#Calculate the tfidf frequency of each neutral term and connect
#The following sorting is based off of code found at: https://stackoverflow.com/questions/45805493/sorting-tfidfvectorizer-output-by-tf-idf-lowest-to-highest-and-vice-versa
sums = response_neu.sum(axis=0)
data = []

for col, term in enumerate(neu_feature_names):
    data.append((term, sums[0,col]))

neu_ranking = pd.DataFrame(data, columns=['neu_term','rank'])
neu_ranking_sorted = neu_ranking.sort_values('rank', ascending=False)

#Print the resulting tfidf matrices
print('')
print("The TF-IDF ranking for the Neutral DataFrame:")
print(neu_ranking_sorted.head(1000))
print('*' * 20)

#Calculate the tfidf frequency of each insecure term and connect
sums = response_ins.sum(axis=0)
data = []
for col, term in enumerate(ins_feature_names):
    data.append((term, sums[0,col]))

ins_ranking = pd.DataFrame(data, columns=['ins_term','rank'])
ins_ranking_sorted = ins_ranking.sort_values('rank', ascending=False)

#Print the resulting tfidf matrices
print('')
print("The TF-IDF ranking for the Insecure DataFrame:")
print(ins_ranking_sorted.head(1000))