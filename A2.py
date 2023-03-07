#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:59:09 2022

@author: kenzasqalli
"""

import pandas as pd


# TASK 2: creating the DataFrame
data = {'y': ['Black', 'Blue', 'Blue'], 'x1': [1, 0, -1], 'x2': [1, 0, -1]}
df = pd.DataFrame(data)

# TASK 3:
    
from sklearn.neighbors import KNeighborsClassifier

X = df.iloc[:,1:]
y = df['y']

knn = KNeighborsClassifier(n_neighbors=2)
model1 = knn.fit(X,y)

# TASK 4:
new_obs = [[0.1,0.1]]
model1.predict(new_obs)

# TASK 5:
model1.predict_proba(new_obs)

# TASK 6:
knn1 = KNeighborsClassifier(n_neighbors=2, weights = "distance").fit(X,y)
model2 = knn1.fit(X,y)
model2.predict(new_obs)
