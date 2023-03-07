#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:04:23 2022

@author: kenzasqalli
"""


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

df = pd.read_excel('/Users/kenzasqalli/Desktop/MMA/FALL 2022/INSY 662 - Data mining and visualization/Individual project/Kickstarter.xlsx')

# 1. DEVELOP A CLASSIFICATION MODEL:
## GOAL: Predict if a project will be successfull or not at the moment of submission

# Cleaning the Data:
    
## To make our analysis easier, we want to reset the indexes of each projects from 0 to 15473
## We want the  current index to be deleted entirely and the numeric index to replace it: dop = True
df = df.reset_index(drop=True)

# Dropping NA values:
df.isna().any()
df = df.dropna()

## Dropping invalid & non-significant variables:
df = df.drop(['name','pledged','deadline','state_changed_at','created_at','launched_at',
                              'staff_pick','backers_count','static_usd_rate','usd_pledged','spotlight',
                              'state_changed_at_weekday','state_changed_at_month','state_changed_at_day',
                              'state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days', 
                              'disable_communication'], axis=1)
df = df.drop('country', axis=1) #The currency gives a larger info, and country and currency are correlated


## We are only interested in data which state = failed or state = successful
df['state'].unique()
## We see that come rows take values 'canceled' or 'suspended', and we want to delete these ones
df = df[df.state != 'canceled']
df = df[df.state != 'suspended']
## Now, our data only has projects that were either succesfull, or projects that failed

## Set successful = 1 and failed = 0
dumdum = {'successful':1, 'failed':0}
df['dum_state'] = df['state'].apply(lambda x: dumdum[x])

## Dummifying the rest of the variables:
df = pd.get_dummies(df, columns=(['currency', 'category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']))

# Setting Target and Independent Variables:
Y = df['dum_state']
X = df.drop(['dum_state', 'state'], axis=1)

## Get rid of anomalities with Isolation Forest:   
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(n_estimators=100,contamination=0.02)
label=iforest.fit_predict(X)

## Then re-selecting Variables:
from numpy import where
non_anom=where(label==1)
X=X.iloc[non_anom]
Y=Y.iloc[non_anom] 

# Standariation:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_std, columns=X.columns)
    
     
# LASSO to choose important variables:
from sklearn.linear_model import Lasso
l = Lasso(alpha=0.01)
model=l.fit(X_std,Y)
model.coef_
features=pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
predictors=features[features['coefficient']!=0]

## Keeping only non-zero features:
    X = X[['goal', 'name_len', 'name_len_clean', 'blurb_len', 'deadline_month', 'deadline_yr',
           'created_at_yr', 'launched_at_month', 'launched_at_hr', 'launch_to_deadline_days',
           'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_USD', 'category_Academic',
           'category_Apps', 'category_Blues', 'category_Experimental', 'category_Festivals',
           'category_Flight', 'category_Gadgets', 'category_Immersive', 'category_Makerspaces',
           'category_Musical', 'category_Places', 'category_Plays', 'category_Robots',
           'category_Shorts', 'category_Software', 'category_Sound', 'category_Spaces',
           'category_Thrillers', 'category_Web', 'category_Webseries', 'deadline_weekday_Tuesday',
           'launched_at_weekday_Friday', 'launched_at_weekday_Tuesday']]


# THE MODEL:
# Splitting the Data:
X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=5)    

# Random Forest

random_forest = RandomForestRegressor(random_state=0, max_features=10 ,n_estimators=100)
model = random_forest.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
mse_forest = mean_squared_error(y_test,y_test_pred)
accuracy_forest = accuracy_score(y_test, y_test_pred.round())
print(mse_forest)
print(accuracy_forest)

# GBT

gbt = GradientBoostingClassifier(random_state=0, n_estimators=100)
model2 = gbt.fit(X_train,y_train)

y_test_pred2 = model2.predict(X_test)
mse_gbt = mean_squared_error(y_test, y_test_pred2)
accuracy_gbt = accuracy_score(y_test, y_test_pred2)
print(mse_gbt)
print(accuracy_gbt)
model2.get_params()  

## Random Forest is better because it has a lower MSE






## Grading ##

# Import Grading Data
kickstarter_grading_df = pandas.read_excel("Kickstarter-Grading-Sample.xlsx")

# Pre-Process Grading Data
kickstarter_grading_df = kickstarter_grading_df.reset_index(drop=True)

# Dropping NA values:
kickstarter_grading_df.isna().any()
kickstarter_grading_df = kickstarter_grading_df.dropna()

## Dropping invalid & non-significant variables:
kickstarter_grading_df = kickstarter_grading_df.drop(['name','pledged','deadline','state_changed_at','created_at','launched_at',
                              'staff_pick','backers_count','static_usd_rate','usd_pledged','spotlight',
                              'state_changed_at_weekday','state_changed_at_month','state_changed_at_day',
                              'state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days', 
                              'disable_communication'], axis=1)
kickstarter_grading_df = kickstarter_grading_df.drop('country', axis=1) #The currency gives a larger info, and country and currency are correlated


## We are only interested in data which state = failed or state = successful
kickstarter_grading_df['state'].unique()
## We see that come rows take values 'canceled' or 'suspended', and we want to delete these ones
kickstarter_grading_df = kickstarter_grading_df[kickstarter_grading_df.state != 'canceled']
kickstarter_grading_df = kickstarter_grading_df[kickstarter_grading_df.state != 'suspended']
## Now, our data only has projects that were either succesfull, or projects that failed

## Set successful = 1 and failed = 0
dumdum = {'successful':1, 'failed':0}
kickstarter_grading_df['dum_state'] = kickstarter_grading_df['state'].apply(lambda x: dumdum[x])

## Dummifying the rest of the variables:
kickstarter_grading_df = pd.get_dummies(kickstarter_grading_df, columns=(['currency', 'category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']))

# Setting Target and Independent Variables:
y_grading = kickstarter_grading_df['dum_state']
X_grading = kickstarter_grading_df.drop(['dum_state', 'state'], axis=1)

## Get rid of anomalities with Isolation Forest:   
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(n_estimators=100,contamination=0.02)
label=iforest.fit_predict(X)

## Then re-selecting Variables:
from numpy import where
non_anom=where(label==1)
X_grading=X_grading.iloc[non_anom]
y_grading=y_grading.iloc[non_anom] 

# Standariation:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_grading)
    X_std = pd.DataFrame(X_std, columns=X_grading.columns)
    
     
# LASSO to choose important variables:
from sklearn.linear_model import Lasso
l = Lasso(alpha=0.01)
model=l.fit(X_std,y_grading)
model.coef_
features=pd.DataFrame(list(zip(X_grading.columns,model.coef_)), columns = ['predictor','coefficient'])
predictors=features[features['coefficient']!=0]

## Keeping only non-zero features:
    X_grading = X_grading[['goal', 'name_len', 'name_len_clean', 'blurb_len', 'deadline_month', 'deadline_yr',
           'created_at_yr', 'launched_at_month', 'launched_at_hr', 'launch_to_deadline_days',
           'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_USD', 'category_Academic',
           'category_Apps', 'category_Blues', 'category_Experimental', 'category_Festivals',
           'category_Flight', 'category_Gadgets', 'category_Immersive', 'category_Makerspaces',
           'category_Musical', 'category_Places', 'category_Plays', 'category_Robots',
           'category_Shorts', 'category_Software', 'category_Sound', 'category_Spaces',
           'category_Thrillers', 'category_Web', 'category_Webseries', 'deadline_weekday_Tuesday',
           'launched_at_weekday_Friday', 'launched_at_weekday_Tuesday']]

# THE MODEL:
# Splitting the Data:
X_train, X_test, y_train,y_test = train_test_split(X_grading,y_grading,test_size=0.3,random_state=5)    

# Random Forest

random_forest = RandomForestRegressor(random_state=0, max_features=10 ,n_estimators=100)
model = random_forest.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
mse_forest = mean_squared_error(y_test,y_test_pred)
accuracy_forest = accuracy_score(y_test, y_test_pred.round())
print(mse_forest)
print(accuracy_forest)

# Apply the model previously trained to the grading data
## Random Forest:
y_grading_pred = model.predict(X_grading)

# Calculate the accuracy score
accuracy_score(y_grading, y_grading_pred)


  
    
# 2. DEVELOP A CLUSTERING MODEL   

# Finding optimal K with Elbow method
from sklearn.cluster import KMeans
withinss = []
for i in range (2,11):    
    kmeans = KMeans(n_clusters=i)
    m = kmeans.fit(X)
    withinss.append(m.inertia_)
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7,8,9,10],withinss)

### We are either going to use 4 or 5 clusters from the above plot results
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Finding optimal K with silhouette method
from sklearn.metrics import silhouette_score
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i)
    m = kmeans.fit(X_std)
    labels = m.labels_
    print(i,':',silhouette_score(X_std,labels))

# Calculate F-score
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(X_std, labels)
print(score)

# Calculate p-value
from scipy.stats import f
df1 = 3 # df1 = k-1
df2 = 13431 # df2 = n-k
pvalue = 1-f.cdf(score, df1, df2)
print(pvalue)
## p-value is sufficiently small, so we reject H0 and conclude that
## Ha is true: there are 4 clusters in the data

# Optimal clusetering : K = 4
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model3 = kmeans.fit(X_std)
labels = model3.labels_
model3.cluster_centers_
centers = pd.DataFrame(model3.cluster_centers_, columns = X.columns)




    
    
    
    
    
    
    