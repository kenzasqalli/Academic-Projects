#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:04:29 2022

@author: kenzasqalli
"""


import pandas as pd
import numpy as np

# TASK 1:
    ## Dropping observations that have one or more missing values:
        df = df.dropna()
    ## Dummifying:
        df= pd.get_dummies(df, columns=['Manuf', 'Type'])
    ## Dropping unnecessary perdictor:
        df= df.drop('Name', axis=1)

# TASK 2:
    #Construct variables
    Y_ = 'Rating_Binary'
    #Creating Predictor Variable
    y = df[Y_]
    #Creating Target Variable
    X = df[df.columns.drop(Y_)]  

# TASK 3:
    from sklearn.preprocessing import StandardScaler
    import scipy.stats as stats
    # Standarizing the data using z-score:
        df_standarized = df.select_dtypes(include='number').apply(stats.zscore)

# TASK 4:
   
parameters = {'hidden_layer_sizes':np.arange(1, 22)} 
clf = GridSearchCV(MLPClassifier(), parameters)


clf.fit(X, y)
print(clf.score(X, y))
print(clf.best_params_)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

#Building ANN with optimal number of hidden layers
mlp = MLPClassifier(hidden_layer_sizes=(16),max_iter=1000, random_state=0)
model = mlp.fit(X_train,y_train)


## Make prediction and evaluate the performance
y_test_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)
