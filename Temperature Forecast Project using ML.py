#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler # Scalers 

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


# In[5]:


df = pd.read_csv('temp.csv')
df.head()


# In[6]:


df.shape
pd.set_option('display.max_columns',30)


# In[11]:


data = df.copy()


# In[13]:


data.info()


# In[19]:


data.isna().sum()


# In[20]:


data['Date'][0].split('-')


# In[22]:


for i in data['Date']:
    try :
        split_obj = i.split('-')
        list_of_year.append(split_obj[2])
        list_of_month.append(split_obj[1])    
        list_of_day.append(split_obj[0]) 
    except AttributeError:
        list_of_year.append(np.nan)
        list_of_month.append(np.nan)
        list_of_day.append(np.nan)


# In[24]:


data['year'] = list_of_year
data['month'] = list_of_month
data['day'] = list_of_day


# In[25]:


data['year'] = pd.to_numeric(data['year']) 
data['month'] = pd.to_numeric(data['month']) 
data['day'] = pd.to_numeric(data['day'])


# In[26]:


data.drop('Date', axis=1, inplace=True)


# In[27]:


data.head()


# In[28]:


def plot_line(para):
    df = data.groupby(by=para).mean().reset_index()
    
    fig = px.line(df, x=df[para], y=['Present_Tmax', 'Present_Tmin'])
    fig.update_layout(template='plotly_dark')
    return fig.show()


# In[29]:


plot_line('day')


# In[30]:


data.columns


# In[31]:


data.head()


# In[32]:


data.drop('station',axis=1, inplace=True)


# In[33]:


def treat_nan(df):
    for i in df.columns:
        df[i].fillna(df[i].mean(),inplace=True)


# In[34]:


treat_nan(data)


# In[35]:


data.head()


# In[36]:


X = data.drop(['Next_Tmax', 'Next_Tmin'], axis=1)
y1 = data['Next_Tmax']
y2 = data['Next_Tmin']


# In[37]:


X.head()


# In[38]:


def normalizer(x_train, x_test):
    scaler = Normalizer()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test


# In[39]:


def minmax(x_train, x_test):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test


# In[40]:


def stdscaler(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test


# In[41]:


def best_model(X, y, scaler, algo):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    X_train, X_test = scaler(X_train, X_test)

    model = algo()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(predictions, y_test)
    mse = mean_squared_error(predictions, y_test)

    print(f'The MAE is {mean_absolute_error(predictions, y_test)}')
    print(f'The MSE is {mean_squared_error(predictions, y_test)}')

    fig = px.scatter(x=predictions, y=y_test, template='plotly_dark', title=f'Actual Productivity vs Predictions')
    fig.update_traces(hovertemplate='Predicted Value : %{x} <br> Actual Value: %{y}')
    fig.update_layout(hoverlabel=dict(
        font_size = 20,
        bgcolor = 'white', 
        font_family = 'Helvetica'
    ))
    fig.update_xaxes(title='Predicted Values', showgrid=False)
    fig.update_yaxes(title='Actual Values', showgrid=False)

    return predictions, y_test, mse, mae, fig.show()


# In[65]:


list_of_target = [y1, y2]

list_of_algos = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, SVR]

list_of_MAE = []
list_of_MSE = []

for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y1, normalizer, i)
    list_of_MSE.append(mse)
    list_of_MAE.append(mae)


# In[43]:


msemae_y1 = pd.DataFrame()
msemae_y1['Algos'] =  ['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'SVR']
msemae_y1['MSE_normalizer'] = list_of_MSE
msemae_y1['MAE_normalizer'] = list_of_MAE 


# In[44]:


msemae_y1.head()


# In[45]:


list_of_MAE1 = []
list_of_MSE1 = []


# In[64]:


for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y1, stdscaler, i)
    list_of_MSE1.append(mse)
    list_of_MAE1.append(mae)


# In[47]:


msemae_y1['MSE_stdscaler'] = list_of_MSE1
msemae_y1['MAE_stdscaler'] = list_of_MAE1 


# In[48]:


list_of_MAE2 = []
list_of_MSE2 = []

for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y1, minmax, i)
    list_of_MSE2.append(mse)
    list_of_MAE2.append(mae)


# In[49]:


msemae_y1['MSE_minmax'] = list_of_MSE2
msemae_y1['MAE_minmax'] = list_of_MAE2 


# In[50]:


list_of_algos = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, SVR]


# In[51]:


list_of_MAE3 = []
list_of_MSE3 = []


# In[52]:


for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y2, normalizer, i)
    list_of_MSE3.append(mse)
    list_of_MAE3.append(mae)


# In[53]:


msemae_y2 = pd.DataFrame()
msemae_y2['Algos'] =  ['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'SVR']
msemae_y2['MSE_normalizer'] = list_of_MSE3
msemae_y2['MAE_normalizer'] = list_of_MAE3


# In[54]:


list_of_MAE4 = []
list_of_MSE4 = []


# In[55]:


for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y2, stdscaler, i)
    list_of_MSE4.append(mse)
    list_of_MAE4.append(mae)


# In[56]:


msemae_y2['MSE_stdscaler'] = list_of_MSE4
msemae_y2['MAE_stdscaler'] = list_of_MAE4


# In[57]:


list_of_MAE5 = []
list_of_MSE5 = []


# In[58]:


for i in list_of_algos:
    print(f"{i}")
    pred, y_test, mse, mae, plot = best_model(X, y2, minmax, i)
    list_of_MSE5.append(mse)
    list_of_MAE5.append(mae)


# In[59]:


msemae_y2['MSE_minmax'] = list_of_MSE5
msemae_y2['MAE_minmax'] = list_of_MAE5


# In[60]:


msemae_y2.head()


# In[61]:


msemae_y1.head()


# In[62]:


fig = px.bar(msemae_y2,  x='Algos', y=['MSE_normalizer', 'MAE_normalizer', 'MSE_stdscaler', 'MAE_stdscaler', 'MSE_minmax', 'MAE_minmax'], barmode='group')
fig.update_layout(title='Representation of MAE and MSE values of different Algorithms on the Next Min Temperature', template='plotly_dark', hoverlabel=dict(
    font_size=20,
    font_family='Arial'
))
fig.update_traces(hovertemplate='%{x} : %{y}')
fig.show()


# In[63]:


fig = px.bar(msemae_y1,  x='Algos', y=['MSE_normalizer', 'MAE_normalizer', 'MSE_stdscaler', 'MAE_stdscaler', 'MSE_minmax', 'MAE_minmax'], barmode='group')
fig.update_layout(title='Representation of MAE and MSE values of different Algorithms on the Next Max Temperature', template='plotly_dark', hoverlabel=dict(
    font_size=20,
    font_family='Arial'
))
fig.update_traces(hovertemplate='%{x} : %{y}')
fig.show()


# In[ ]:




