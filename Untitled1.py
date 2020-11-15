#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# # 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('PCA_practice_dataset.csv', header=None)


# In[3]:


data = df.to_numpy()


# In[4]:


data.shape


# In[5]:


data = np.matrix(data)


# In[7]:


data.shape


# In[72]:


mean_val = np.mean(data, axis=0) #axis=0 => column wise


# In[73]:


mean_val.shape


# In[8]:


d = np.matrix(data.T)


# In[9]:


d.shape


# In[10]:


cov = np.cov(d)


# In[11]:


cov.shape


# In[12]:


eig_val, eig_vec = np.linalg.eig(cov)


# In[14]:


eigen_vec_ls = []
for i in range(eig_vec.shape[1]):
    eig1 = data@eig_vec[:,i]
    eig1 = eig1/eig_val[i]
    eigen_vec_ls.append(np.ravel(eig1))


# In[16]:


len(eigen_vec_ls[0])


# In[17]:


threshold_list = np.arange(0.90, 0.98, 0.01)


# In[18]:


threshold_list


# In[19]:


sort_idx = np.argsort(eig_val) ## indices for eigenvalues which are in ascending order
sort_idx = sort_idx[::-1]

eig_val_sum = np.sum(eig_val)

principal_eig_vec = []
principal_eig_val = []

for t in threshold_list:
    temp_sum = 0
    temp_val = []
    temp_vec = []
    i=0
    while(temp_sum<t*eig_val_sum):
        temp_vec.append(eigen_vec_ls[sort_idx[i]])
        temp_val.append(eig_val[sort_idx[i]])
        temp_sum += eig_val[sort_idx[i]]
        i += 1
    print("Number of components is {}".format(i) + " With threshold : {}".format(t))
    principal_eig_vec.append(temp_vec)
    principal_eig_val.append(temp_val)


# In[21]:


len(principal_eig_vec[0])


# In[178]:


#fig, ax = plt.subplots()
plt.figure(figsize=(10, 11))
for i in range(len(principal_eig_vec)):
    plt.subplot(4, 2, i+1)
    plt.plot(principal_eig_val[i])

plt.tight_layout()


# In[ ]:









