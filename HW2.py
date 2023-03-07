#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:52:18 2023

@author: kenzasqalli
"""
# Q1

import gurobipy as gp
import pandas as pd

#Import data set
df = pd.read_csv('advertising.csv')

# Remove the added index column
df = df.drop(columns=['Unnamed: 0'])

# Create a Gurobi model
m = gp.Model()

# Define the independent variables B0, B1, B2, B3 as variables in the model
B0 = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='B0')
B1 = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='B1')
B2 = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='B2')
B3 = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='B3')

# Define the dependent variables F_i
F = {}
for i in range(df.shape[0]):
    F[i] = m.addVar(lb=0, ub=gp.GRB.INFINITY, name='F'+str(i))

# Add the constraints for each observation
for i in range(df.shape[0]):
    y_i = df.iloc[i]['Sales']
    TV = df.iloc[i]['TV']
    Radio = df.iloc[i]['Radio']
    Newspaper = df.iloc[i]['Newspaper']
    m.addConstr(F[i] >= y_i - (B0 + B1*TV + B2*Radio + B3*Newspaper))
    m.addConstr(F[i] >= -(y_i - (B0 + B1*TV + B2*Radio + B3*Newspaper)))

# Minimize the sum of F_i
m.setObjective(sum(F.values()), gp.GRB.MINIMIZE)

# Optimize the model
m.optimize()

# Print the parameter estimates
print('B0:', B0.x)
print('B1:', B1.x)
print('B2:', B2.x)
print('B3:', B3.x)


# Q2
# Create the blobs
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=2, 
                  random_state=0, cluster_std=0.40)

df_blobs = pd.DataFrame({'TV': X[:,0], 'Radio': X[:,1], 'Sales': y})

import matplotlib.pyplot as plt

# Visualize the data
plt.scatter(df_blobs['TV'], df_blobs['Radio'], c=df_blobs['Sales'], cmap='viridis')
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('Scatter plot of TV and Radio with Sales as target variable')
plt.show()

# Finding a solution to the optimization problem
import gurobipy as gp

model = gp.Model()
N = df_blobs.shape[0]

# create variables
Alpha = model.addVars(N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Alpha")
Beta = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Beta")
Epsilon = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Epsilon")

# set objective
model.setObjective(Epsilon, gp.GRB.MAXIMIZE)

# add constraints
for i in range(N):
    if df_blobs.loc[i, 'Sales'] == 1:
        model.addConstr(Alpha[i] * df_blobs.loc[i, 'TV'] + Beta >= Epsilon)
    else:
        model.addConstr(Alpha[i] * df_blobs.loc[i, 'TV'] + Beta + Epsilon <= 0)

# optimize model
model.optimize()



