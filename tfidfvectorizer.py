# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:41:20 2018

@author: suvanjeet
"""

#%%
text= ["While not mandatory to do at this stage prior to tokenization (you will find that this statement is the norm for the relatively flexible ordering of text data preprocessing tasks), replacing contractions with their expansions can be beneficial at this point, since our word tokenizer will split words like didnt into did and not. Its not impossible to remedy this tokenization at  a later stage, but doing so prior makes it easier and more straightforward."]
text

#%%
from sklearn.feature_extraction.text import CountVectorizer

vector=CountVectorizer(stop_words='english')
#%%
vector.fit(text)
#%%
print(vector.vocabulary_)

#%%

v1=vector.transform(text)
#%%
v1.shape
type(v1)
print(v1)

#%%
from sklearn.feature_extraction.text  import TfidfVectorizer
vector1= TfidfVectorizer(stop_words='english')

#%%
vector1.fit(text)
print(vector1.vocabulary_)

#%%
v2=vector1.transform(text)
print(v2)

#%%
print(vector1.use_idf)

#%%

v2.toarray()

#%%
text= "my name is navdeep kohli data"
text
text1="my name was suvanjeet kohli science"
text1

# refreshing, love,characters
#%%
review=[text,text1]
review
#%%
from sklearn.feature_extraction.text  import TfidfVectorizer
vector1= TfidfVectorizer(stop_words='english')

#%%
vector1.fit(review)
print(vector1.vocabulary_)

#%%
v2=vector1.transform(review)
print(v2)

#%%
print(vector1.idf_)
#%%
vector=CountVectorizer(stop_words='english')
#%%
vector.fit(review)
#%%
print(vector.vocabulary_)

#%%

v1=vector.transform(review)
#%%
v1.shape
type(v1)
print(v1)

#%%
from sklearn import preprocessing
import numpy as np

x=[0.21666,0.21666,0.21666,0.21666,1.33]
x=np.asarray(x)
#%%
x= preprocessing.normalize(x)
x

















