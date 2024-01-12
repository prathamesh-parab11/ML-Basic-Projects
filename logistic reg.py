#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

info = {
    'Gender' : ['male', 'female', 'female', 'male', 'female', 'female'],
    'Position' : ['CEO', 'Cleaner', 'Employee', 'Cleaner', 'CEO', 'Cleaner']
}
df = pd.DataFrame(info)

df


# In[2]:


from sklearn.preprocessing import LabelEncoder


# In[3]:


le = LabelEncoder()
gender_encoded = le.fit_transform(df['Gender'])

gender_encoded


# In[4]:


df['encoded_gender'] = gender_encoded


# In[5]:


df


# In[6]:


le = LabelEncoder()
position_encoded = le.fit_transform(df['Position'])

position_encoded


# In[7]:


df['encoded_position'] = position_encoded


# In[8]:


df


# In[9]:


#drop Dropping irrelevant columns
df1 = df.drop(['Gender', 'Position'], axis=1)


# In[10]:


df1


# In[11]:


import pandas as pd
EmployeeData=pd.DataFrame({'id': [101,102,103,104,105],
                        'Gender': ['M','M','M','F','F'],
                           'Age': [21,25,24,28,25],
                           'Department':['QA','QA','Dev','Dev','UI'],
                           'Rating':['A','B','B','C','B']
                          })
# Priting data
print(EmployeeData)


# In[12]:


# Converting Ordinal Variable Rating to numeric
EmployeeData['Rating'].replace({'A':1, 'B':2, 'C':3}, inplace=True)
print(EmployeeData)


# In[13]:


# Converting binary Nominal Variable Gender to numeric
EmployeeData['Gender'].replace({'M':1, 'F':0}, inplace=True)
print(EmployeeData)


# In[15]:


# Converting multiclass Nominal Variable Department to numeric
# by generating dummy variables
EmployeeData=pd.get_dummies(EmployeeData)
EmployeeData


# In[18]:


import os
os.listdir()


# In[17]:


os.chdir("D:\codenera_csv")


# In[19]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[20]:


ins=pd.read_csv("insurance_data.csv")


# In[21]:


ins


# In[22]:


plt.scatter(ins.age,ins.bought_insurance,marker='+',color='red')
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


# Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(ins[['age']],ins.bought_insurance,train_size=0.8)


# In[26]:


X_test


# In[27]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[28]:


model.fit(X_train, y_train)


# In[29]:


y_predicted = model.predict(X_test)


# In[30]:


model.predict_proba(X_test)


# In[31]:


print(model.predict([[50]]))


# In[32]:


#check for model
model.score(X_test,y_test)


# In[33]:


y_predicted


# In[34]:


X_test


# In[ ]:


#model.coef_ indicates value of m in y=m*x + b equation
model.coef_


# In[ ]:


#model.intercept_ indicates value of b in y=m*x + b equation
model.intercept_


# In[35]:


#Lets defined sigmoid function now and do the math with hand
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y
age = 45
prediction_function(age)


# In[36]:


#0.485 is less than 0.5 which means person with 35 age will not buy insurance
age = 43
prediction_function(age)
#0.485 is more than 0.5 which means person with 43 will buy the insurance


# In[41]:


os.listdir()


# In[38]:


hr1=pd.read_csv("HR_comma_sep.csv")


# In[39]:


hr1


# In[43]:


hp4=pd.read_csv("homeprices (4).csv")


# In[44]:


hp4


# In[45]:


# Converting multiclass Nominal Variable Department to numeric by generating dummy variables
hp4=pd.get_dummies(hp4)
hp4


# In[46]:


#rename when we needed
hp4 = hp4.rename(columns={"town_monroe township": "monroe", "town_robinsville": "robinsville","town_west windsor":'windsor' })
hp4


# In[47]:


#drop Dropping irrelevant columns
hp4n = hp4.drop(['price'], axis=1)


# In[48]:


hp4n


# In[49]:


plt.scatter(hp4.area,hp4.price,marker='+',color='red')
plt.show()


# In[63]:


import os
import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from sklearn import linear_model #sckit Lean module to import 


# In[65]:


#calling linear model function and fit all the values
reg=linear_model.LinearRegression()
reg.fit(hp4.drop("price",axis=1),hp4.price)


# In[66]:


reg.predict([[3000,1,0,0]])


# In[ ]:




