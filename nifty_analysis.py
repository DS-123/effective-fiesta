
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from nsepy import get_history,get_index_pe_history
import datetime as dt


# In[ ]:


nifty_df=pd.read_csv('D:\\Nifty_it_index.csv')
nifty_df.head()


# In[ ]:


#moving average
def SMA(data, ndays): 
 SMA = pd.Series((data['Close']).rolling(window=ndays).mean(),name = 'SMA')
 data = data.join(SMA) 
 return data


# In[ ]:


#moving average for 4 weeks - if there are 5 working days, then n = 20
SMA(nifty_df, 20)


# In[ ]:


#moving average for 16 weeks - if there are 5 working days, then n = 80
SMA(nifty_df, 80)


# In[ ]:


#moving average for 52 weeks - if there are 5 working days, then n = 260 but here i ll take n=240 due to data size
SMA(nifty_df, 240)


# In[ ]:


vol = nifty_df['Volume'].index.values.tolist()
volume_shocks =[]
for i in range(1,len(vol)):
    if(nifty_df['Volume'][i]>(nifty_df['Volume'][i-1]+(.1*nifty_df['Volume'][i-1]))):
        volume_shocks.append(1)
    else:
        volume_shocks.append(0)
nifty_df['Volume_shocks']=pd.DataFrame({'Volume Shocks':volume_shocks})
nifty_df.head()


# In[ ]:


closing=nifty_df['Close'].index.tolist()
price_shock=[]
for i in range(0,len(closing)-1):
    if(nifty_df['Close'][i+1]-nifty_df['Close'][i]>(.02*(nifty_df['Close'][i+1]-nifty_df['Close'][i]))):
        price_shock.append(1)
    else:
        price_shock.append(0)
nifty_df['Price_shocks']=pd.DataFrame({'Price_shocks':price_shock})
nifty_df.head()


# In[ ]:


#model testing
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# In[ ]:


X = nifty_df.iloc[:, :-1].values
y = nifty_df.iloc[:, 5].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


#predict the test set results
y_pred = lr.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - float(len(y)-1)/(len(y)-len(lr.coef_)-1)*(1 - r2)

rmse, r2, adj_r2,  lr.coef_, lr.intercept_


# In[ ]:


from sklearn.linear_model import LassoLars
regressor = LassoLars(alpha=0.01)
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[ ]:


#Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies


# In[ ]:


#ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


from sklearn.linear_model import Ridge
regressor = Ridge()
regressor.fit(X_train,y_train)


# In[ ]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred


# In[ ]:


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
accuracies


# In[ ]:


def price_shock_wo_vol_shock(stock):
    
    stock["not_vol_shock"]  = ((stock["Volume_shocks"].astype(bool))).astype(int)
    stock["price_shock_w/0_vol_shock"] = stock["not_vol_shock"] & stock["Price_shocks"]
    
    return stock


# In[ ]:


price_shock_wo_vol_shock(nifty_df)


# In[ ]:


import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.io import show, output_notebook
from bokeh.palettes import Blues9
from bokeh.palettes import RdBu3
from bokeh.models import ColumnDataSource, CategoricalColorMapper, ContinuousColorMapper
from bokeh.palettes import Spectral11


# In[ ]:


def bokeh_plot(stock):
    data = dict(stock=stock['Close'], Date=stock.index)
    
    p = figure(plot_width=800, plot_height=250)
    p.line(stock.index, stock['Close'], color='blue', alpha=0.5)
    
    #show price shock w/o vol shock
    
    p.circle(stock.index, stock.Close*stock["price_shock_w/0_vol_shock"], size=4, legend='price shock without vol shock')
    show(p)


# In[ ]:


bokeh_plot(nifty_df)

