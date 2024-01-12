#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[35]:


os.chdir("D:\codenera_csv")


# In[36]:


os.listdir()


# In[37]:


ker=pd.read_csv("kerala.csv")


# In[38]:


ker


# In[39]:


ker.head(10)


# In[40]:


ker.tail(5)


# In[41]:


ker.shape


# In[42]:


ker.dtypes


# In[43]:


ker.columns


# In[44]:


ker = ker.drop(['SUBDIVISION'], axis=1)


# In[45]:


ker.columns


# In[50]:


ker=ker.rename(columns={' ANNUAL RAINFALL':'A_R'})


# In[51]:


ker


# In[52]:


duplicate_rows_df = ker[ker.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[53]:


ker.isnull().sum()


# In[54]:


from sklearn.preprocessing import LabelEncoder


# In[56]:


le = LabelEncoder()
FLOODS_encoded = le.fit_transform(ker['FLOODS'])
FLOODS_encoded


# In[57]:


ker['encoded_FLOODS'] = FLOODS_encoded


# In[58]:


ker


# In[59]:


ker = ker.drop(['FLOODS'], axis=1)


# In[60]:


ker


# In[62]:


plt.figure(figsize=(20,10))
c= ker.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# In[64]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(ker['YEAR'], ker['A_R'])
ax.set_xlabel('YEAR')
ax.set_ylabel('A_R')
plt.show()


# In[78]:


plt.figure(figsize=(20,10))
plt.scatter(ker.YEAR,ker.A_R,color='yellow',marker='*')
plt.gca().set_facecolor('black')
z = np.polyfit(ker.YEAR,ker.A_R, 1)
p = np.poly1d(z)
plt.plot(ker.YEAR,ker.A_R)
plt.xlabel('YEAR')
plt.ylabel('A_R')


# In[79]:


from sklearn.datasets import load_iris
dataset=load_iris()


# In[80]:


dataset.feature_names


# In[81]:


dataset.target_names


# In[82]:


data=pd.DataFrame(dataset.data,columns=dataset.feature_names)


# In[83]:


data


# In[84]:


data.isnull().sum()


# In[85]:


data['target']=dataset.target


# In[86]:


data


# In[87]:


data[data.target==0]


# In[88]:


data["flower_Name"]=data.target.apply(lambda x: dataset.target_names[x])


# In[89]:


data


# In[90]:


data1=data[:50]
data1


# In[91]:


data2=data[50:100]
data2


# In[92]:


data3=data[100:]
data3


# In[94]:


plt.scatter(data1["sepal length (cm)"],data1['sepal width (cm)'],color="red",marker="+")
plt.scatter(data2["sepal length (cm)"],data2['sepal width (cm)'],color="blue",marker="*")


# In[96]:


plt.scatter(data1['petal length (cm)'],data1['petal width (cm)'],color="purple",marker="+")
plt.scatter(data2['petal length (cm)'],data2['petal width (cm)'],color="brown",marker=".")


# In[97]:


from sklearn.model_selection import train_test_split


# In[100]:


x=data.drop(['target','flower_Name'],axis='columns')


# In[101]:


y=data.target


# In[102]:


X_train, X_test, y_train ,y_test = train_test_split(x,y,train_size=0.2)


# In[103]:


X_train


# In[104]:


X_test


# In[105]:


from sklearn.neighbors import KNeighborsClassifier


# In[106]:


knn=KNeighborsClassifier()


# In[107]:


#Our Model is ready 
knn.fit(X_train,y_train)


# In[108]:


#evaluating the model
knn.score(X_test,y_test)


# In[111]:


knn.predict([[2.5,3.5,0.5,2.7]])


# In[112]:


from sklearn.metrics import confusion_matrix


# In[113]:


y_pred = knn.predict(X_test)


# In[116]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[114]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[117]:


#Our Model is ready 
knn.fit(X_test,y_test)


# In[118]:


knn.score(X_train,y_train)


# In[119]:


knn.predict([[2.5,3.5,0.5,2.7]])


# In[120]:


from sklearn.metrics import confusion_matrix


# In[122]:


x_pred = knn.predict(X_train)


# In[123]:


cm = confusion_matrix(y_train, x_pred)
cm


# In[ ]:




