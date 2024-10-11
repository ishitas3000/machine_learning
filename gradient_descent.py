# COST FUNCTION ********** COST FUNCTION ********** COST FUNCTION ********** COST FUNCTION **********

import numpy as np
x_trainSet = np.array ([0,1,2,3,4,5])
y_trainSet = np.array ([3,4,5,6,7,8])

x_fitLine = np.array([])
y_fitLine = np.array([])

def cost_func (x, y, w, b):
    
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        x_fitLine = np.append([x[i]])
        x_fitLine = np.append([f_wb])
        cost = ((f_wb) - y[i]) ** 2
        cost_sum += cost
        j = (1 / (2 * m)) * cost_sum
    
    print(x_fitLine)
    print(y_fitLine)
    print(f"The cost value of 'J' is equal to: {j}")

cost_func(x_trainSet, y_trainSet, 1, 4)


# GRADIENT DESCENT ********** GRADIENT DESCENT ********** GRADIENT DESCENT ********** GRADIENT DESCENT **********

import math, copy

def gradient_func (x, y, w, b):

    m = x.shape[0]

    d_jw = 0
    d_jb = 0

    disc_j = x.shape[1,4]
    disc_w = x.shape[5,9]
    disc_b = x.shape[10,14]