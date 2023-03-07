#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:07:59 2022

@author: kenzasqalli
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# TASK 0:
    # Keeping only variables we are interested in:
        X = df[['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']]
    # Droping observatios with null values:
        X = X.dropna()
    
# TASK 1:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # Agglomerative Clustering, number of clusters = 2
    # Using sklearn
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')  
    cluster.fit_predict(X_std) 
    ## Printing how many observations are in each cluster:
    labels_array = cluster.labels_
    labels_list = labels_array.tolist()
    c1 = labels_list.count(0)
    c2 = labels_list.count(1)
    print(c1) #71
    print(c2) #3


# TASK 2:
   kmeans = KMeans(n_clusters=2)
   model = kmeans.fit(X_std)
   labels = model.predict(X_std)
   labels_list2 = labels.tolist()
   c3 = labels_list2.count(0)
   c4 = labels_list2.count(1)
   print(c3) # 23
   print(c4) # 51
   

## TASK 3
centroids = model.cluster_centers_
print(pd.DataFrame({"Attribute": X.columns, "Cluster 1" : centroids[0], "Cluster 2" : centroids[1]}))


