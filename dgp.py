#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:13:43 2019

@author: harshparikh
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

    
fig = plt.figure(figsize=(8.75,7))
plt.scatter(X1, Y, alpha=0.5)
plt.scatter(X2, Y, alpha=0.5)
plt.ylabel('Y'); plt.xlabel('X[.]');
plt.legend(['X1','X2'])
fig.savefig('dgp_standard.png')

