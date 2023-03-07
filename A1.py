#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:57:00 2022

@author: kenzasqalli
"""

# Importing all needed modules:
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

### TASK 0:
   
    # Import data:
    TC = pd.read_csv("ToyotaCorolla.csv")
   
    # Define variables:
    X = TC.iloc[:,3:]
    Y = TC['Price']
   
    # Standarized data:
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()    
    scaled_X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(scaled_X, columns=X.columns)

    # Separate the data:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = 0.35, random_state = 662)

### TASK 1:
   
    # Run linear regression based on the training data
    lm = LinearRegression()
    model = lm.fit(X_train,Y_train)
   
    # Generate the prediction value from the test data
    Y_test_pred = model.predict(X_test)
   
    # Calculate the MSE
    from sklearn.metrics import mean_squared_error
    lm_mse = mean_squared_error(Y_test, Y_test_pred)
    
    
### TASK 2:
    
    # Run ridge regression:
    ridge1 = Ridge(alpha=1)
    model1 = ridge1.fit(X_train,Y_train)
    
    # Generate the prediction value from the test data:
    Y_test_pred1 = model1.predict(X_test)

    # Calculate the MSE
    ridge_mse = mean_squared_error(Y_test, Y_test_pred1)    
    
    
### TASK 3:
    
    # Run Lasso
    from sklearn.linear_model import Lasso
    lasso1 = Lasso(alpha=1)
    model2 = lasso1.fit(X_train,Y_train)
    
    # Generate the prediction value from the test data
    Y_test_pred2 = model2.predict(X_test)

    # Calculate the MSE
    lasso_mse = mean_squared_error(Y_test, Y_test_pred2)
    
    
### TASK 4:
    ## Alpha = 10:
        # Ridge:
            ridge2 = Ridge(alpha=10)
            model3 = ridge2.fit(X_train,Y_train)
            Y_test_pred3 = model3.predict(X_test)
            ridge_mse2 = mean_squared_error(Y_test, Y_test_pred3)    
    
        # LASSO:
            lasso2 = Lasso(alpha=10)
            model4 = lasso2.fit(X_train,Y_train)
            Y_test_pred4 = model4.predict(X_test)
            ridge_mse3 = mean_squared_error(Y_test, Y_test_pred4) 
            
       ## Alpha = 100:
           # Ridge:
               ridge3 = Ridge(alpha=100)
               model5 = ridge3.fit(X_train,Y_train)
               Y_test_pred5 = model5.predict(X_test)
               ridge_mse4 = mean_squared_error(Y_test, Y_test_pred5)    
       
           # LASSO:
               lasso3 = Lasso(alpha=100)
               model6 = lasso3.fit(X_train,Y_train)
               Y_test_pred6 = model6.predict(X_test)
               ridge_mse5 = mean_squared_error(Y_test, Y_test_pred6) 
           
           
        ## Alpha = 1000:
            # Ridge:
                ridge4 = Ridge(alpha=1000)
                model7 = ridge4.fit(X_train,Y_train)
                Y_test_pred7 = model7.predict(X_test)
                ridge_mse6 = mean_squared_error(Y_test, Y_test_pred7)    
        
            # LASSO:
                lasso4 = Lasso(alpha=1000)
                model8 = lasso4.fit(X_train,Y_train)
                Y_test_pred8 = model8.predict(X_test)
                ridge_mse7 = mean_squared_error(Y_test, Y_test_pred8) 
            
        ## Alpha = 10000:
            # Ridge:
                ridge5 = Ridge(alpha=10000)
                model9 = ridge5.fit(X_train,Y_train)
                Y_test_pred9 = model9.predict(X_test)
                ridge_mse8 = mean_squared_error(Y_test, Y_test_pred9)    
            
            # LASSO:
                lasso5 = Lasso(alpha=10000)
                model10 = lasso5.fit(X_train,Y_train)
                Y_test_pred10 = model10.predict(X_test)
                ridge_mse9 = mean_squared_error(Y_test, Y_test_pred10) 
                
               
            # coefficients:
                #Ridge:
                    model9.coef_
                #LASSO:
                    model10.coef_
           
      
           
           
           
           
    
    
    
    
    