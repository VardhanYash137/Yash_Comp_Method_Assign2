#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def rk4_adaptive(f, t0, y0, t_end, h0, tol):
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0
    h = h0
    
    while t < t_end:
        y1 = rk4_step(f, t, y, h)
        y2 = rk4_step(f, t + h, y1, h)
        
        y3 = rk4_step(f, t, y, 2*h)
        
        # Use the difference between two approximations to estimate the error
        error = (np.abs(y3 - y2))
        rho=(30*tol*h/error)
        
        # If error is within tolerance, accept the step
        if rho > 1:
            t = t + h
            y = y2
            t_values.append(t)
            y_values.append(y)
        
        # Adjust step size based on the error
        h = h*((rho)**0.25)
    
    return np.array(t_values), np.array(y_values)

# Example usage:
def f(t, y):
    return (y**2+y)/t  # Example differential equation dy/dt = (y^2+y)/t

t0 = 1
y0 = -2
t_end = 3
h0 = 0.1
tol = 1e-4

t_arr, y_arr = rk4_adaptive(f, t0, y0, t_end, h0, tol)
print("t values:", t_arr)
print("y values:", y_arr)
plt.scatter(t_arr, y_arr)
plt.plot(t_arr, y_arr)
plt.xlabel("y")
plt.ylabel("y(t)")
plt.grid()
plt.title("y'=(y^2+y)/t")
plt.show()


# In[ ]:




