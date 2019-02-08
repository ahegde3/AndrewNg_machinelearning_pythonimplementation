#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:40:56 2019

@author: ahegde3
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta

data=pd.read_csv('ex1data1.txt',header=None)
x=data.iloc[:,0]
y=data.iloc[:,1]
m=len(y)
data.head()

plt.scatter(x,y)
plt.xlabel("population")
plt.ylabel("profit")
plt.show()

x = x[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
x = np.hstack((ones, x))

J = computeCost(x, y, theta)
print(J)

theta = gradientDescent(x, y, theta, alpha, iterations)
print(theta)


plt.scatter(x[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(x[:,1], np.dot(x, theta))
plt.show()