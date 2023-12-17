#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas --upgrade')


# In[4]:


import pandas as pd


# In[22]:


df = pd.read_csv('global_power_plant_database.csv')
df


# In[24]:


df.columns


# In[25]:


df.info()


# In[26]:


selected_column = [
    'country',
    'country_long',
    'name',
    'capacity_mw',
    'primary_fuel',
    'other_fuel1',
    'other_fuel2',
    'other_fuel3',
    'commissioning_year',
    'year_of_capacity_data',
    'generation_gwh_2013',
    'generation_gwh_2014',
    'generation_gwh_2015',
    'generation_gwh_2016',
    'generation_gwh_2017',
    'estimated_generation_gwh'
]


# In[27]:


len(selected_column)


# In[14]:


df.sample(4)


# In[28]:


df.describe()


# # Fule use in power plant
# # Primary Fule

# In[30]:


main_primary_fuel = df.primary_fuel.value_counts()
main_primary_fuel


# In[32]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (12, 8)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[44]:


countries_plant = df.country_long.value_counts().head(30)
countries_plant


# In[45]:


sns.barplot(x = countries_plant.index, y = countries_plant)
plt.xticks(rotation = 90)
plt.title('Country Designation')
plt.ylabel('Number of Power Plant')
plt.xlabel('Countries');


# In[46]:


main_primary_fuel = df.primary_fuel.value_counts() * 100 / df.primary_fuel.count()
main_primary_fuel


# In[47]:


sns.barplot(x = main_primary_fuel, y = main_primary_fuel.index)
plt.title('Main primary fuel')
plt.xlabel('Count (Percentages)');
plt.ylabel('Different type of power plant depands on the type of fuel uses');


# # Capacity of generating power

# In[48]:


countries_capacity = df.groupby('country_long')[['capacity_mw']].sum().sort_values('capacity_mw', ascending = False).head(30)
countries_capacity


# In[49]:


sns.barplot(x = countries_capacity.index, y = countries_capacity.capacity_mw)
plt.xticks(rotation = 90)
plt.title('Countries with capacity');


# In[ ]:




