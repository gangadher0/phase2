#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


loan_dataset = pd.read_csv('Loan Application dataset.csv')


# In[4]:


type(loan_dataset)


# In[6]:


loan_dataset.head(5)


# In[7]:


loan_dataset.shape


# In[8]:


loan_dataset.describe()


# In[12]:


loan_dataset.isnull().sum()


# In[10]:


loan_dataset = loan_dataset.dropna()


# In[11]:


loan_dataset.isnull().sum()


# In[15]:


loan_dataset.replace({"Loan_Status":{'N':0, 'Y':1}},inplace=True)


# In[16]:


loan_dataset.head()


# In[17]:


loan_dataset['Dependents'].value_counts()


# In[18]:


loan_dataset = loan_dataset.replace(to_replace='3+', value=4)


# In[19]:


loan_dataset['Dependents'].value_counts()


# In[20]:


sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


# In[21]:


sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)


# In[22]:


loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[23]:


loan_dataset.head()


# In[25]:


X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']


# In[26]:


print(X)
print(Y)


# In[27]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[28]:


print(X.shape,X_train.shape, X_test.shape)


# In[31]:


classifier = svm.SVC(kernel='linear')


# In[32]:


classifier.fit(X_train,Y_train)


# In[33]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[34]:


print('Accuracy on training data : ', training_data_accuracy)


# In[35]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[36]:


print('Accuracy on test data : ', test_data_accuracy)


# # Accuracy on test data :  0.8333333333333334
