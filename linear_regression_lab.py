import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
matplotlib inline

# load the dataset
x_train, y_train = load_data()

# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

#------------------------------------------------------------------------------------

# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    cost_sum = 0 
    

    for i in range(m):
        f_wb = w * x[i] + b 
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    
    total_cost = (1 / (2 * m)) * cost_sum
    ### END CODE HERE ### 

    return total_cost


# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)

#------------------------------------------------------------------------------------

# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
   """
   Computes the gradient for linear regression 
   Args:
     x (ndarray): Shape (m,) Input to the model (Population of cities) 
     y (ndarray): Shape (m,) Label (Actual profits for the cities)
     w, b (scalar): Parameters of the model  
   Returns
     dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
     dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
   """

   # Number of training examples
   m = x.shape[0]

   # You need to return the following variables correctly
   dj_dw = 0
   dj_db = 0

   ### START CODE HERE ### 
   # Loop over examples
   for i in range(m):  
       # Your code here to get prediction f_wb for the ith example
       f_wb = w * x[i] + b

       # Your code here to get the gradient for w from the ith example 
       dj_dw_i = (f_wb - y[i]) * x[i] 

       # Your code here to get the gradient for b from the ith example 
       dj_db_i = f_wb - y[i]

       # Update dj_db : In Python, a += 1  is the same as a = a + 1
       dj_db += dj_db_i

       # Update dj_dw
       dj_dw += dj_dw_i

   # Divide both dj_dw and dj_db by m
   dj_dw = dj_dw / m
   dj_db = dj_db / m

   ### END CODE HERE ### 
   return dj_dw, dj_db
    


# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)
