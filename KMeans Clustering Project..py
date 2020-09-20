#!/usr/bin/env python
# coding: utf-8

# # A Project on K-Means Clustering for Customer Segmentation.

# ### Table of contents
# 
# - Introdution
# - Setting up K-Means
# - Creating the Visual Plot
# Customer Segmentation with K-Means
# Pre-processing
# Modeling
# Insights
# 

# ## Introduction
# 
# There are many models for **clustering** out there. In this project, I will be presenting the model that is considered one of the simplest models amongst them. Despite its simplicity, the **K-means** is vastly used for clustering in many data science applications, especially useful if we need to quickly discover insights from **unlabeled data**.
# 
# Some real-world applications of k-means:
# - Customer segmentation
# - Understand what the visitors of a website are trying to accomplish
# - Pattern recognition
# - Machine learning
# - Data compression
# 
# In this project, I have represented the algorithms for how to use k-Means for **customer segmentation** using **k-means**.
# 
# 

# **Required Libraries**

# In[1]:


import random 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Dataset
# 
# We have the following customer dataset, and we need to apply customer segmentation on this historical data. Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining those customers. Another group might include customers from non-profit organizations.

# In[45]:


data= pd.read_csv('C:/Users/hp/Downloads/Cust_Segmentation.csv')
data.head()


# ### Data Cleaning

# Dropping address and Customer id

# In[46]:


df= data.drop(['Customer Id','Address'], axis=1)
df.head()


# #### Dealing with null values

# In[47]:


df.isnull().sum()


# In[48]:


df=df.dropna()
df.isnull().sum()


# In[49]:


df.describe()


# We can have a look into the statistical summary of variables present in the dataset.

# In[50]:


df.shape


# **Transforming the data to scale and compute smoothly**.

# In[51]:


from sklearn.preprocessing import StandardScaler
x= df.values[:,0:]
#x= np.nan_to_num(x)
t_df = StandardScaler().fit_transform(x)     #transformed Data


# In[52]:


t_df[0:5]


# Above is the transformed data and is now ready for modeling.

# ### Modeling k-means clustering

# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically.
# Now we can create a profile for each group, considering the common characteristics of each cluster. 
# For example, the 3 clusters can be:
# 
# - AFFLUENT, EDUCATED AND OLD AGED
# - MIDDLE AGED AND MIDDLE INCOME
# - YOUNG AND LOW INCOME

# In[53]:


k=3
kmeans= KMeans(init='k-means++', n_clusters= k, n_init=10)
kmeans.fit(x)
labels=kmeans.labels_


# In[54]:


df['clus']= labels    # adding labels to the dataset


# In[55]:


df.head()


# We can easily check the centroid values by averaging the variables in each cluster.

# In[56]:


df.groupby('clus').mean()


# Now, lets look at the distribution of customers based on their age and income:

# In[57]:


plt.scatter(t_df[:, 0], t_df[:, 3], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


# The three segments has been  categorized on basis of their income as low, middle, and high. 

# 3D visualization along with 'education' feature.

# In[58]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=45, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(t_df[:, 1], t_df[:, 0], t_df[:, 3], c= labels.astype(np.float))


# ### Evaluation
# As we applied KMeans clustering algorithm to the scaled data **t-df**, we shall now evaluate the performance of the clusterig model. Generally, we have following two methods to evaluate the performance :
# 
# 1. Elbow Method
# 2. Silhouette Coefficient

# #### 1. Elbow Method

# To perform the elbow method, we run several k-means, increment k with each iteration, and record the SSE (Sum of Squarred Errors). 

# In[59]:


sse=[]
for k in range(1,11):
    km=KMeans(init='k-means++', n_clusters=k, n_init=10)
    km.fit(t_df)
    sse.append(km.inertia_)
    


# Plotting **Number of clusters** vs **SSE**

# In[60]:


plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# Here, we have elbow at cluster **2** and **4**. To choose the optimal one out of both, we will try to evaluate from **silhouette criteria**.
# 

# #### 2. Silhouette Coefficient

# The silhouette coefficient is a measure of cluster cohesion and separation. It quantifies how well a data point fits into its assigned cluster based on two factors:
# 
# - How close the data point is to other points in the cluster
# - How far away the data point is from points in other clusters
# 
# Silhouette coefficient values range between -1 and 1. Larger numbers indicate that samples are closer to their clusters than they are to other clusters.

# In[61]:


sil_coef= []
for k in range(2,11):
    km2=KMeans(init= 'k-means++', n_clusters=k, n_init=12)
    km2.fit(t_df)
    score= silhouette_score(t_df, km2.labels_)
    sil_coef.append(score)
    
print(sil_coef)


# Plotting the average silhouette scores for each k shows that the best choice for k is 3 since it has the maximum score:

# In[62]:


plt.style.use('fivethirtyeight')
plt.plot(range(2,11),sil_coef)
plt.xticks(range(2,11))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Here also, we find 2 to be the optmal number of clusters as it has the highest score. Further we fit the final model with number of clusters 2.

# ### Optimal Clustering: 2-Means cluster 

# In[63]:


kmeans1= KMeans(n_clusters=2)
kmeans1.fit(t_df)


# In[64]:


kmeans1.inertia_    # optimal SSE


# In[65]:


labels=kmeans1.labels_    # Optimal Labels


# In[66]:


kmeans1.cluster_centers_     # optimal Centroids


# Let's explore the model through visuals from **Age** , **income** and **education**.

# In[67]:


fig = plt.figure(figsize=(8,4))

plt.scatter(t_df[:, 0], t_df[:, 3], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

fig = plt.figure(figsize=(8,5))
plt.subplot(2,1,2)
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=134)
plt.cla()
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('education')
ax.scatter(t_df[:, 0], t_df[:, 1], t_df[:, 3], c= labels.astype(np.float))


# ### Conclusion

# Finally, after evaluating by Elbow and Silcouette Coefficent, we interpret that optimal number of clusters to be used here, is 2. We can further analyse this dataset as per business requirements considering this cluster modeling in account.

# 
# Thanks,
# 
# **Arun Virha**

# In[ ]:




