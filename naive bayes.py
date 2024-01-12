#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from sklearn import linear_model #sckit Lean module to import


# In[18]:


data=sns.load_dataset("Titanic")


# In[19]:


data


# In[20]:


data.head(5)


# In[21]:


data.columns


# In[22]:


data.drop(['pclass','sibsp', 'parch','class','embarked','adult_male','deck','embark_town','alive','alone','who'],axis=1, inplace=True)


# In[23]:


data


# In[24]:


inputs =data.drop("survived",axis="columns")
target =data.survived
target


# In[25]:


dum=pd.get_dummies(inputs.sex)
dum.head()


# In[26]:


inputs=pd.concat([inputs,dum],axis="columns")
inputs


# In[27]:


inputs.drop(["sex","male"],axis="columns",inplace=True)


# In[28]:


inputs.age.isnull().sum()


# In[29]:


inputs.age=inputs.age.fillna(inputs.age.mean())


# In[30]:


inputs.isnull().sum()


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,train_size=0.3)


# In[32]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()


# In[33]:


model.fit(X_train,y_train)


# In[34]:


model.score(X_test,y_test)


# In[35]:


X_test[:10]


# In[36]:


y_test[:10]


# In[37]:


model.predict_proba(X_test[:10])


# In[38]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[39]:


model.fit(X_train, y_train)


# In[40]:


y_predicted = model.predict(X_test)


# In[41]:


model.predict_proba(X_test)


# In[43]:


print(model.predict([[50,100,1]]))


# In[44]:


#check for model
model.score(X_test,y_test)


# In[ ]:




