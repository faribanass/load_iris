#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np 
from sklearn import tree 
from sklearn import datasets 
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score


# In[3]:


iris = load_iris() 
X = iris.data 
y = iris.target 
X_train, X_test,y_train, y_test = train_test_split(X, y, random_state=20,test_size=0.20) 


# In[4]:


tree = DecisionTreeClassifier() 
tree.fit(X_train, y_train) 
y_pred = tree.predict(X_test) 


# In[9]:


print("Accuracy:", accuracy_score(y_test,y_pred)) 
print("Precision:", precision_score(y_test, y_pred, average="weighted")) 
print('Recall:', recall_score(y_test, y_pred, average="weighted")) 


# In[10]:


print('F1 score:', f1_score(y_test, y_pred,average="weighted")) 


# In[30]:


import numpy as np 
from sklearn .metrics import roc_auc_score 


# In[29]:


y_true = [1, 0, 0, 1] 
y_pred = [1, 0, 0.9, 0.2] 
auc = np.round(roc_auc_score(y_true, 
							y_pred), 3) 
print("Auc", (auc)) 

