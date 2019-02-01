
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nsepy import get_history,get_index_pe_history
import datetime as dt


# In[2]:


start = dt.datetime(2015,1,1)

end = dt.datetime(2016,1,1)

tcs = get_history(symbol='TCS', start = start, end = end)
tcs.index = pd.to_datetime(tcs.index)
tcs.head(10)


# In[5]:


new_tcs=tcs.reset_index()
new_tcs['Date']=pd.to_datetime(new_tcs['Date'])
new_tcs.head()


# In[6]:


#moving average
def SMA(data, ndays): 
 SMA = pd.Series((data['Close']).rolling(window=ndays).mean(),name = 'SMA')
 data = data.join(SMA) 
 return data


# In[7]:


#moving average for 4 weeks - if there are 5 working days, then n = 20
SMA(new_tcs, 20)


# In[8]:


#moving average for 16 weeks - if there are 5 working days, then n = 80
SMA(new_tcs, 80)


# In[9]:


#moving average for 52 weeks - if there are 5 working days, then n = 260 but here i ll take n=240 due to data size
SMA(new_tcs, 240)


# In[10]:


tcsdf=pd.DataFrame(new_tcs)


# In[11]:


tcsdf.head()


# In[14]:


vol = tcsdf['Volume'].index.values.tolist()
volume_shocks =[]
for i in range(1,len(vol)):
    if(tcsdf['Volume'][i]>(tcsdf['Volume'][i-1]+(.1*tcsdf['Volume'][i-1]))):
        volume_shocks.append(1)
    else:
        volume_shocks.append(0)
tcsdf['Volume_shocks']=pd.DataFrame({'Volume Shocks':volume_shocks})
tcsdf.head()


# In[15]:


closing=tcsdf['Close'].index.tolist()
price_shock=[]
for i in range(0,len(closing)-1):
    if(tcsdf['Close'][i+1]-tcsdf['Close'][i]>(.02*(tcsdf['Close'][i+1]-tcsdf['Close'][i]))):
        price_shock.append(1)
    else:
        price_shock.append(0)
tcsdf['Price_shocks']=pd.DataFrame({'Price_shocks':price_shock})
tcsdf.head()


# In[16]:


#model testing
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# In[17]:


TCS=tcs.drop(columns=['Symbol','Series'])
TCS.head()


# In[18]:


X = TCS.iloc[:, :-1].values
y = TCS.iloc[:, 11].values


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[20]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[21]:


#predict the test set results
y_pred = lr.predict(X_test)
y_pred


# In[22]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - float(len(y)-1)/(len(y)-len(lr.coef_)-1)*(1 - r2)

rmse, r2, adj_r2,  lr.coef_, lr.intercept_


# In[23]:


from sklearn.linear_model import LassoLars
regressor = LassoLars(alpha=0.01)
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[24]:


#Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies


# In[25]:


#ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[26]:


from sklearn.linear_model import Ridge
regressor = Ridge()
regressor.fit(X_train,y_train)


# In[27]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[28]:


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies

