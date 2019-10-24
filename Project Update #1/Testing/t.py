import pandas as pd  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

lis = [(1, 2, 'Student1', '20'), (3, 4, 'Student2', '21'), (9, 10, 'Student3', '31'), (9, 10, 'Student4', '30'), (9, 10, 'Student5', '30')]  

df_=pd.DataFrame(lis, columns=['Index1', 'Index2', 'Name', 'Age']) 

print(df_)

print('*'*20)

s = pd.Series(df_.get('Age'))

vocab = s.tolist()

vocab = list(map(str, vocab))

#print(vocab)

print('*'*20)

#TFIDF

nvocab = [20,25,30]

nvocab = list(map(str, nvocab))

#print(nvocab)

print('*'*20)

tvocab = ['this', 'is', 'a', 'list', 'hello', 'list']

print(df_['Age'])

vec = TfidfVectorizer()
t = vec.fit_transform(df_['Age'])
voc = vec.get_feature_names()

print('*'*20)

print(t.shape)

print('*'*20)

print(t.toarray())

print('*'*20)

# sum tfidf frequency of each term through documents
sums = t.sum(axis=0)

# connecting term to its sums frequency
data = []
for col, term in enumerate(voc):
    data.append( (term, sums[0,col] ))

ranking = pd.DataFrame(data, columns=['term','rank'])
print(ranking.sort_values('rank', ascending=False))