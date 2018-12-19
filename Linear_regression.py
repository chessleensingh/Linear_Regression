#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


#datasets
X1= [1,2,3,4,5,6,7,8,9,10]
y = [0.9,2.1,3.2,3.8,4.95,6.05,6.98,8,9.10,9.95]
X_test1 = [12,13]
y_test = [12,13]
X = np.asarray(X1)
X = X.reshape(-1,1)
X_test = np.asarray(X_test1)
X_test = X_test.reshape(-1,1)


# In[5]:


#visualize
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X, y, c=X1, cmap='plasma')
ax1.plot(X1,X1,'g--')


# In[7]:


#model
clf = LinearRegression()
clf.fit(X, y)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)


# In[10]:


#plot the results
X_plot= [1,2,3,4,5,6,7,8,9,10,11,12,13]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X, y, c= X1, cmap = 'plasma')
ax1.plot(X_plot,X_plot, 'g--')
ax1.scatter(X_test, y_pred, c='r')


# In[11]:


print(accuracy)


# In[12]:


print(y_pred, y_test)


# In[ ]:




