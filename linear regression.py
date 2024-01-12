#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from sklearn import linear_model #sckit Lean module to import 


# In[7]:


os.chdir("D:\codenera_csv")


# In[8]:


os.listdir()


# In[9]:


hp=pd.read_csv("homeprices.csv")


# In[10]:


hp


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(hp.area,hp.price,color='red',marker='+')


# In[27]:


plt.figure(figsize=(5,5))
plt.scatter(hp.area,hp.price,color='red',marker='+')
z = np.polyfit(hp.area,hp.price, 1)
p = np.poly1d(z)
plt.plot(hp.area,hp.price)
plt.xlabel('area')
plt.ylabel('price')


# In[14]:


new_df = hp.drop('price',axis='columns')
new_df


# In[15]:


price = hp.price
price


# In[16]:


#create a model
reg = linear_model.LinearRegression()  # reg Regression Object 
reg.fit(new_df,price)


# In[17]:


print(reg.predict([[12000]]))


# In[18]:


reg.coef_


# In[19]:


reg.intercept_


# In[20]:


12000*135.78767123 + 180616.43835616432


# In[28]:


os.listdir()


# In[29]:


canada=pd.read_csv("canada_per_capita_income.csv")


# In[30]:


canada


# In[33]:


#rename when we needed
canada = canada.rename(columns={"year": "Year", "per capita income (US$)": "PCI" })
canada


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(canada.Year,canada.PCI,color='red',marker='+')


# In[35]:


#removal of duplicates rows
duplicate_rows_df = canada[canada.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[36]:


#find the missing values 
canada.isnull().sum()


# In[37]:


canada.shape


# In[38]:


plt.figure(figsize=(10,5))
c= canada.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# In[39]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(canada['Year'], canada['PCI'])
ax.set_xlabel('Year')
ax.set_ylabel('PCI')
plt.show()


# In[40]:


new_canada = canada.drop('PCI',axis='columns')
new_canada


# In[42]:


PCI = canada.PCI
PCI


# In[49]:


#create a model
regression = linear_model.LinearRegression()  # reg Regression Object 
regression.fit(new_canada,PCI)


# In[50]:


print(regression.predict([[2020]]))


# In[53]:


regression.coef_


# In[54]:


regression.intercept_


# In[55]:


2020*828.46507522 + -1632210.7578554575


# In[56]:


print(regression.predict([[2025]]))


# In[57]:


print(regression.predict([[2038]]))


# # linear regression with multiple values

# In[58]:


hpn=pd.read_csv("homeprices_new.csv")


# In[59]:


hpn


# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(hpn.area,hpn.price,color='red',marker='+')


# In[65]:


#finding median of bedrooms
hpn.bedrooms.median() 


# In[72]:


#replaces median with NAN value
hpn.bedrooms=hpn.bedrooms.fillna(hpn.bedrooms.median())


# In[73]:


#calling linear model function and fit all the values
reg=linear_model.LinearRegression()
reg.fit(hpn.drop("price",axis=1),hpn.price)


# In[79]:


reg.predict([[2000,3,2]])


# In[80]:


reg.coef_


# In[81]:


reg.intercept_


# In[82]:


2000*112.06244194 + 2000*23388.88007794 + 2000*-3231.71790863 + 221323.00186540437


# In[ ]:




