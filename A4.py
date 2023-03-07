#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:27:53 2022

@author: kenzasqalli
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import timeit
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


## TASK 0:
    # Dropping 'UserID' column:
       df = df.drop('UserID', axis = 1)
    
    # Target variable: 'Personal Loan':
        y = df['Personal Loan']
    
    # Predictors:
        X = df.iloc[:,1:]

## TASK 1:
    # Standariation:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        X_std = pd.DataFrame(X_std, columns=X.columns)

    # LASSO:
        L = Lasso(alpha = 0.05)
        model = L.fit(X_std,y)
        model.coef_
        X_lasso = X_std[['Income', 'Education', 'CD Account']]
        
        pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['Predictor', 'Coefficient'])

## TASK 2:
    #PCA:
        pca = PCA(n_components=3)
        pca.fit(X_std)
        pca.explained_variance_ratio_
        X_new = pca.transform(X_std)
        


## TASK 3: Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

## TASK 4 & 5: K-NN model:

    knn = KNeighborsClassifier(n_neighbors = 3)
    # Model 1:
        start1 = timeit.default_timer()
        model1 = knn.fit(X_train, y_train)
        stop1 = timeit.default_timer()
        
        y_test_pred = model1.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_test_pred)
        p1 = precision_score(y_test, y_test_pred)
        r1 = recall_score(y_test, y_test_pred)
        t1 = stop1 - start1

    # Model 2:
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_lasso, y, test_size=0.3, random_state=5) 
        start2 = timeit.default_timer()
        model2 = knn.fit(X_train_2, y_train_2)        
        stop2 = timeit.default_timer()
        
        y_test_pred_2 = model2.predict(X_test_2)
        accuracy_knn_2 = accuracy_score(y_test_2, y_test_pred_2)
        p2 = precision_score(y_test_2, y_test_pred_2)
        r2 = recall_score(y_test_2, y_test_pred_2)
        t2 = stop2 - start2
        
       

    # Model 3:
        X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_new, y, test_size=0.3, random_state=5) 
        start3 = timeit.default_timer()
        model3 = knn.fit(X_train_3, y_train_3)        
        stop3 = timeit.default_timer()
        
        y_test_pred_3 = model3.predict(X_test_3)
        accuracy_knn_3 = accuracy_score(y_test_3, y_test_pred_3) 
        p3 = precision_score(y_test_3, y_test_pred_3)
        r3 = recall_score(y_test_3, y_test_pred_3)
        t3 = stop3 - start3
        
     
## TASK 6:
    # Model 2:
rf = RandomForestClassifier(random_state=0)
model = rf.fit(X_lasso, y)
result = permutation_importance(rf, X_lasso, y, n_repeats=1, random_state=0)
perm_sorted_idx = result.importances_mean.argsort()
tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5
fig, (ax1) = plt.subplots(1, 1, figsize=(6, 8))
ax1.barh(tree_indices, rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(X_lasso.columns[tree_importance_sorted_idx])



    # Model 3:
X_new = pd.DataFrame(X_new)
rf = RandomForestClassifier(random_state=0)
model = rf.fit(X_new, y)
result = permutation_importance(rf, X_new, y, n_repeats=1, random_state=0)
perm_sorted_idx = result.importances_mean.argsort()
tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5
fig, (ax1) = plt.subplots(1, 1, figsize=(6, 8))
ax1.barh(tree_indices, rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(X_new.columns[tree_importance_sorted_idx])


























