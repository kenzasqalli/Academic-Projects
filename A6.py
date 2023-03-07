#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:59:51 2022

@author: kenzasqalli
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
## ASSIGNMENT 6:

# TASK 1:
    # Target variable: 'raw_material':
        y = df['raw_material']
    # Predictor:
        X=df[['time']]
    # Splitting the data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# TASK 2: Ridge Regression Alpha = 1
ridge = Ridge(alpha = 1)
ridge.fit(X_train, y_train) # Fit a ridge regression on the training data
pred = ridge.predict(X_test)# Use this model to predict the test data
print(pd.Series(ridge.coef_, index = X.columns))
print(mean_squared_error(y_test, pred))

# TASK 3: isolation forest model to remove anomalies
# Building isolation forest model
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(contamination=.1)

pred_ = iforest.fit_predict(df)
score = iforest.decision_function(df)

# TASK 4:
    # Extracting anomalies
    from numpy import where
    anomaly_index = where(pred_==-1)
    anomaly_values = df.iloc[anomaly_index]

    # Creating a new dataframe without the anomalies:
        df_new = df.merge(anomaly_values, how='left', indicator=True)
        df_new = df_new[df_new['_merge'] == 'left_only']

# TASK 5: Running a Ridge regression without outliers: 
        y_new = df_new['raw_material']
        X_new = df_new[['time']]
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_new, y_new, test_size=0.3, random_state=5)

ridge = Ridge(alpha = 1)
ridge.fit(X_train_n, y_train_n) # Fit a ridge regression on the training data
pred_n = ridge.predict(X_test_n)# Use this model to predict the test data
print(pd.Series(ridge.coef_, index = X_new.columns))
print(mean_squared_error(y_test_n, pred_n))
