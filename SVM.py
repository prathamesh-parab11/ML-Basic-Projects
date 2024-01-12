#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from sklearn import linear_model #sckit Lean module to import


# In[2]:


import os
os.chdir("D:\codenera_csv")
os.getcwd()


# In[3]:


import pandas as pd 
from sklearn.datasets import load_iris
dataset=load_iris()


# In[4]:


#feature_names  belongs to columns
dataset.feature_names


# In[5]:


dataset.target_names


# In[6]:


data=pd.DataFrame(dataset.data, columns=dataset.feature_names)
data


# In[7]:


#checking null values :
data.isnull().sum()


# In[8]:


data['target']=dataset.target
data


# In[9]:


data[data.target==0]


# In[10]:


data["flower_Name"]=data.target.apply(lambda x: dataset.target_names[x])
data


# In[11]:


#create Three Data Frame

data1=data[:50]

data2=data[50:100]

data3=data[100:]


# In[12]:


plt.scatter(data1["sepal length (cm)"],data1['sepal width (cm)'],color="green",marker="+")
plt.scatter(data2["sepal length (cm)"],data2['sepal width (cm)'],color="blue",marker=".")


# In[13]:


plt.scatter(data1['petal length (cm)'],data1['petal width (cm)'],color="green",marker="+")
plt.scatter(data2['petal length (cm)'],data2['petal width (cm)'],color="blue",marker=".")


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


x=data.drop(['target',"flower_Name"],axis='columns')


# In[16]:


y=data.target


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.2)


# In[18]:


X_train


# In[19]:


X_test


# In[20]:


from sklearn.svm import SVC
model=SVC()


# In[21]:


model.fit(X_train,y_train)


# In[22]:


model.score(X_test,y_test)


# In[24]:


model.predict([[3.2,5.0,0.5,0.11]])


# In[ ]:




