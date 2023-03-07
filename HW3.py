#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:28:31 2023

@author: kenzasqalli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# load the data
sales_data = pd.read_csv("sales.csv")
sales_data = sales_data.drop(columns=['Unnamed: 1'])
sales_data = sales_data.drop(columns=['Unnamed: 2'])

# Problem 1:

# define negative log-likelihood function
def neg_log_likelihood(params, data):
    mu, sigma = params
    pdf = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((data - mu) / sigma)**2)
    return -np.log(pdf).sum()

# initialize starting values for mu and sigma
params_0 = [sales_data.mean(), sales_data.std()]

# minimize the negative log-likelihood function
result = minimize(neg_log_likelihood, params_0, args=(sales_data,), method="BFGS")
mu, sigma = result.x

# plot the data and the fitted normal distribution
x = np.linspace(sales_data.min(), sales_data.max(), 1000)
pdf = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
plt.hist(sales_data, bins=50, density=True, alpha=0.5, label="Observed")
plt.plot(x, pdf, label="Fitted Normal")
plt.legend()
plt.show()

# Problem 2
# Define the two functions
def f(x, y):
    return (x - 5)**2 + 2*(y + 3)**2 + x*y

def g(x, y):
    return (1 - (y - 3))**2 + 10*((x + 4) - (y - 3)**2)**2

# Define the gradient of each objective function
def gradient_f(x, y):
    grad_x = 2 * (x - 5) + y
    grad_y = 4 * (y + 3) + x
    return np.array([grad_x, grad_y])

def gradient_g(x, y):
    grad_x = 20 * ((x + 4) - (y - 3)**2)
    grad_y = 2 * (y - 3) - 40 * ((x + 4) - (y - 3)**2) * (y - 3)
    return np.array([grad_x, grad_y])

# Define the gradient descent algorithm with a constant learning rate
def gradient_descent(x0, y0, grad_fn, learning_rate, num_iters):
    x = x0
    y = y0
    history = []
    for i in range(num_iters):
        grad = grad_fn(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        history.append((x, y))
    return history

# Define the gradient descent algorithm with an exponential decay learning rate
def gradient_descent_exp_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters):
    x = x0
    y = y0
    history = []
    for i in range(num_iters):
        learning_rate = initial_learning_rate * np.exp(-decay_rate * i)
        grad = grad_fn(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        history.append((x, y))
    return history

# Define the gradient descent algorithm with an inverse decay learning rate
def gradient_descent_inv_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters):
    x = x0
    y = y0
    history = []
    for i in range(num_iters):
        learning_rate = initial_learning_rate / (1 + decay_rate * i)
        grad = grad_fn(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        history.append((x, y))
    return history

# Minimize the objective function f
x0 = 0
y0 = 2
learning_rate = 0.05
num_iters = 100
history_f = gradient_descent(x0, y0, gradient_f, learning_rate, num_iters)
x_f, y_f = np.array(history_f).T

# Minimize the objective function g
x0 = 0
y0 = 2
learning_rate = 0.0015
num_iters = 100
history_g = gradient_descent(x0, y0, gradient_g, learning_rate, num_iters)
x_g, y_g = np.array(history_g).T


# Plot the value of the objective function after each gradient descent step for f
values_f = [f(x, y) for x, y in history_f]
plt.plot(values_f, label='Constant learning rate, f')

# Plot the value of the objective function after each gradient descent step for g
values_g = [g(x, y) for x, y in history_g]
plt.plot(values_g, label='Constant learning rate, g')

#Get the final value of the objective function for f
final_value_f = values_f[-1]
print("The final value of the objective function for f is:", final_value_f)

#Get the final value of the objective function for g
final_value_g = values_g[-1]
print("The final value of the objective function for g is:", final_value_g)




# Comparing the performance of gradient_descent_exp_decay and 
# gradient_descent_inv_decay with a constant learning rate

## Function f

# Minimize the objective function f with a constant learning rate
x0 = 0
y0 = 2
learning_rate = 0.05
num_iters = 100
history_f_const = gradient_descent(x0, y0, gradient_f, learning_rate, num_iters)
values_f_const = [f(x, y) for x, y in history_f_const]

# Minimize the objective function f with an exponential decay learning rate
x0 = 0
y0 = 2
initial_learning_rate = 0.05
decay_rate = 0.01
history_f_exp = gradient_descent_exp_decay(x0, y0, gradient_f, initial_learning_rate, decay_rate, num_iters)
values_f_exp = [f(x, y) for x, y in history_f_exp]

# Minimize the objective function f with an inverse decay learning rate
x0 = 0
y0 = 2
initial_learning_rate = 0.05
decay_rate = 0.01
history_f_inv = gradient_descent_inv_decay(x0, y0, gradient_f, initial_learning_rate, decay_rate, num_iters)
values_f_inv = [f(x, y) for x, y in history_f_inv]

# Plot the value of the objective function after each gradient descent step for f
plt.plot(values_f_const, label='Constant learning rate, f')
plt.plot(values_f_exp, label='Exponential decay learning rate, f')
plt.plot(values_f_inv, label='Inverse decay learning rate, f')
plt.legend()
plt.show()

## Function g

# Minimize the objective function g with a constant learning rate
x0 = 0
y0 = 2
learning_rate = 0.0015
num_iters = 100
history_g_const = gradient_descent(x0, y0, gradient_g, learning_rate, num_iters)
values_g_const = [g(x, y) for x, y in history_g_const]

# Minimize the objective function g with an exponential decay learning rate
x0 = 0
y0 = 2
initial_learning_rate = 0.0015
decay_rate = 0.01
history_g_exp = gradient_descent_exp_decay(x0, y0, gradient_g, initial_learning_rate, decay_rate, num_iters)
values_g_exp = [g(x, y) for x, y in history_g_exp]

# Minimize the objective function f with an inverse decay learning rate
x0 = 0
y0 = 2
initial_learning_rate = 0.0015
decay_rate = 0.01
history_g_inv = gradient_descent_inv_decay(x0, y0, gradient_g, initial_learning_rate, decay_rate, num_iters)
values_g_inv = [g(x, y) for x, y in history_g_inv]

# Plot the value of the objective function after each gradient descent step for f
plt.plot(values_g_const, label='Constant learning rate, g')
plt.plot(values_g_exp, label='Exponential decay learning rate, g')
plt.plot(values_g_inv, label='Inverse decay learning rate, g')
plt.legend()
plt.show()

# Tunning the parameters
# Function f:
def tune_decay_rate_f(x0, y0, grad_fn, initial_learning_rate, decay_rate_range, num_iters):
    best_decay_rate = None
    best_value = float('inf')
    for decay_rate in decay_rate_range:
        history_f_exp = gradient_descent_exp_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters)
        values_f_exp = [f(x, y) for x, y in history_f_exp]
        history_f_inv = gradient_descent_inv_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters)
        values_f_inv = [f(x, y) for x, y in history_f_inv]
        if min(values_f_exp + values_f_inv) < best_value:
            best_decay_rate = decay_rate
            best_value = min(values_f_exp + values_f_inv)
    return best_decay_rate

# Tune the decay rate parameter
x0 = 0
y0 = 2
initial_learning_rate = 0.05
decay_rate_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
best_decay_rate_f = tune_decay_rate_f(x0, y0, gradient_f, initial_learning_rate, decay_rate_range, num_iters)

# Function g:
def tune_decay_rate_g(x0, y0, grad_fn, initial_learning_rate, decay_rate_range, num_iters):
    best_decay_rate = None
    best_value = float('inf')
    for decay_rate in decay_rate_range:
        history_g_exp = gradient_descent_exp_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters)
        values_g_exp = [g(x, y) for x, y in history_g_exp]
        history_g_inv = gradient_descent_inv_decay(x0, y0, grad_fn, initial_learning_rate, decay_rate, num_iters)
        values_g_inv = [g(x, y) for x, y in history_g_inv]
        if min(values_g_exp + values_g_inv) < best_value:
            best_decay_rate = decay_rate
            best_value = min(values_g_exp + values_g_inv)
    return best_decay_rate

# Tune the decay rate parameter
x0 = 0
y0 = 2
initial_learning_rate = 0.05
decay_rate_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
best_decay_rate_g = tune_decay_rate_g(x0, y0, gradient_f, initial_learning_rate, decay_rate_range, num_iters)



# Number of iterations
def gradient_descent(x0, y0, grad_fn, learning_rate, num_iters, epsilon):
    x = x0
    y = y0
    history = []
    for i in range(num_iters):
        grad = grad_fn(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        history.append((x, y))
        if np.linalg.norm(grad) < epsilon:
            return history
    return history

x0 = 0
y0 = 2
learning_rate_f = 0.05
learning_rate_g = 0.0015
num_iters = 2000
epsilon = 1e-4
history_f = gradient_descent(x0, y0, gradient_f, learning_rate_f, num_iters, epsilon)
history_g = gradient_descent(x0, y0, gradient_g, learning_rate_g, num_iters, epsilon)
least_num_iters = min(len(history_f), len(history_g))




