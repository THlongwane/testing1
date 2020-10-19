#!/usr/bin/env python
# coding: utf-8

# In[490]:


import string 
import nltk
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer


#from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


# In[491]:


#Loading data
test = pd.read_csv("D:/Data Science/Classification/Data/test.csv")
train = pd.read_csv("D:/Data Science/Classification/Data/train.csv")


# In[492]:


test.head()


# In[493]:


train.head()


# In[560]:


y = train["sentiment"]
X = train["message"]

porter_stemmer=PorterStemmer()

def my_cool_preprocessor(text):
    text=text.lower() 
    text=re.sub("\\W"," ",text) 
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) 
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

def my_tokenizer(text):
    text=re.sub("(\\W)"," \\1 ",text)
    return re.split("\\s+",text)


# In[561]:


vectorizer = TfidfVectorizer(ngram_range= (1,2),tokenizer=my_tokenizer,min_df= 2,max_df=0.7,analyzer = "word",smooth_idf=False,preprocessor=my_cool_preprocessor, stop_words= "english")
X_vectorized = vectorizer.fit_transform(X)


# In[562]:


X_train,X_val,y_train,y_val = train_test_split(X_vectorized,y,test_size= 0.3,shuffle = True, stratify = y, random_state = 42)


# In[563]:


rfc = XGBClassifier()

#rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_val)



lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
lsvc_pred = lsvc.predict(X_val)


# In[564]:


f1_score(y_val, lsvc_pred, average = "macro")


# In[565]:


testx = test["message"]
test_vect = vectorizer.transform(testx)


# In[566]:


y_pred = lsvc.predict(test_vect)


# In[567]:


test["sentiment"] = y_pred


# In[553]:


test.head()


# In[554]:


test[["tweetid","sentiment"]].to_csv("final.csv", index = False)


# In[ ]:





# In[ ]:




