
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nsepy import get_history,get_index_pe_history
import datetime as dt


# In[2]:


start = dt.datetime(2015,1,1)

end = dt.datetime(2016,1,1)

infy = get_history(symbol='INFY', start = start, end = end)
infy.index = pd.to_datetime(infy.index)
infy.head(10)


# In[3]:


new_infy=infy.reset_index()


# In[4]:


new_infy['Date']=pd.to_datetime(new_infy['Date'])
new_infy.head()


# In[11]:


#moving average
def SMA(data, ndays): 
 SMA = pd.Series((data['Close']).rolling(window=ndays).mean(),name = 'SMA')
 data = data.join(SMA) 
 return data


# In[12]:


#moving average for 4 weeks - if there are 5 working days, then n = 20
SMA(new_infy, 20)


# In[17]:


#moving average for 16 weeks - if there are 5 working days, then n = 80
SMA(new_infy, 80)


# In[19]:


#moving average for 52 weeks - if there are 5 working days, then n = 260 but here i ll take n=240 due to data size
SMA(new_infy, 240)


# In[37]:


infydf=pd.DataFrame(new_infy)
infydf.head()


# In[38]:


vol = infydf['Volume'].index.values.tolist()
volume_shocks =[]
for i in range(1,len(vol)):
    if(infydf['Volume'][i]>(infydf['Volume'][i-1]+(.1*infydf['Volume'][i-1]))):
        volume_shocks.append(1)
    else:
        volume_shocks.append(0)
print(volume_shocks)


# In[42]:


infydf['Volume_shocks']=pd.DataFrame({'Volume Shocks':volume_shocks})


# In[44]:


infydf.head(10)


# In[47]:


closing=infydf['Close'].index.tolist()
price_shock=[]
for i in range(0,len(closing)-1):
    if(infydf['Close'][i+1]-infydf['Close'][i]>(.02*(infydf['Close'][i+1]-infydf['Close'][i]))):
        price_shock.append(1)
    else:
        price_shock.append(0)
infydf['Price_shocks']=pd.DataFrame({'Price_shocks':price_shock})
infydf.head(10)


# In[50]:


#model testing
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# In[53]:


Infy=infy.drop(columns=['Symbol','Series'])
Infy.head()


# In[54]:


X = Infy.iloc[:, :-1].values
y = Infy.iloc[:, 11].values


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[57]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[58]:


#predict the test set results
y_pred = lr.predict(X_test)
y_pred


# In[59]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - float(len(y)-1)/(len(y)-len(lr.coef_)-1)*(1 - r2)

rmse, r2, adj_r2,  lr.coef_, lr.intercept_


# In[94]:


from sklearn.linear_model import LassoLars
regressor = LassoLars(alpha=0.01)
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[73]:


#Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies


# In[74]:


#ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[75]:


from sklearn.linear_model import Ridge
regressor = Ridge()
regressor.fit(X_train,y_train)


# In[76]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[78]:


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies

