#!/usr/bin/env python
# coding: utf-8

# # AIR QUALITY INDEX

# In[1]:


import os


# In[2]:


import time


# In[5]:


get_ipython().system('python -m pip install requests')


# In[9]:


import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


df=pd.read_csv(r"C:\Users\lenovo\Downloads\AQI_dataset\city_day.csv")
df


# In[11]:


df.info()


# In[12]:


df.shape


# In[13]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


df.columns


# In[ ]:


df


# In[19]:


# mapping
dist=(df['City'])
distset=set(dist)
dd=list(distset)
dictOfWords={dd[i] : i for i in range(0,len(dd))}
df['City']=df['City'].map(dictOfWords)


# In[21]:


dist=(df['AQI_Bucket'])
distset=set(dist)
dd=list(distset)
dictOfWords={dd[i] : i for i in range(0,len(dd))}
df['AQI_Bucket']=df['AQI_Bucket'].map(dictOfWords)


# In[22]:


df['AQI_Bucket']=df['AQI_Bucket'].fillna(df['AQI_Bucket'].mean())


# In[24]:


df


# In[23]:


df.isnull().sum()


# In[25]:


df=df.drop('Date',1)


# In[26]:


df.columns


# In[27]:


df=df.drop('AQI_Bucket',1)


# In[28]:


import plotly.express as px
fig=px.scatter(df,x='City',y='AQI')
fig.show()


# In[54]:


df.dropna()


# In[55]:


import plotly.express as px
fig8=px.scatter(df,x="SO2",y="AQI")
fig.show()


# In[56]:


import plotly.express as px
fig8=px.scatter(df,x="CO",y="AQI")
fig8.show()


# In[57]:


import plotly.express as px
fig11=px.scatter(df,x='Toluene',y='AQI')
fig11.show()


# In[58]:


features=df[['City','PM2.5','NO','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']]
labels=df['AQI']


# In[59]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,labels,test_size=0.2,random_state=2)


# In[60]:


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[61]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)


# In[62]:


X_train = Xtrain.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


# In[64]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr=RandomForestRegressor(max_depth=2,random_state=0)
regr.fit(Xtrain,Ytrain)
print(regr.predict(Xtest))


# In[68]:


y_pred=regr.predict(Xtest)


# In[69]:


from sklearn.metrics import r2_score


# In[70]:


r2_score(Ytest,y_pred)


# In[ ]:




