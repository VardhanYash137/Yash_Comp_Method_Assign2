#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

def ode_func1(x,y):
    dy=np.zeros((2,len(x)))
    dy[0]=y[1]
    dy[1]=-np.exp(-2*y[0])
    return dy

def boundary_conditions1(ya,yb):
    return np.array([ya[0],yb[0]-np.log(2)])

x_values1=np.linspace(1,2,100)
y_guess1 = np.zeros((2, len(x_values1)))
sol1 = solve_bvp(ode_func1, boundary_conditions1, x_values1, y_guess1)


# In[17]:


def ode_func2(x, y):
    dydx = np.zeros((2, len(x)))
    dydx[0] = y[1]  # dy/dx = y'
    dydx[1] = (y[1]*np.cos(x))-(y[0]*np.log(y[0]))  # d^2y/dx^2 = -y
    return dydx

def boundary_conditions2(ya, yb):
    return np.array([ya[0]-1, yb[0] - np.exp(1)])  # y(a) = 1, y(b) = 1


x_values2 = np.linspace(0.001,np.pi/2, 100)
y_guess2 = np.ones((2, len(x_values2)))
sol2 = solve_bvp(ode_func2, boundary_conditions2, x_values2, y_guess2)


# In[18]:


def ode_func3(x, y):
    dydx = np.zeros((2, len(x)))
    dydx[0] = y[1]  # dy/dx = y'
    dydx[1] = -( (2*(y[1]**3)) + ((y[0]**2)*y[1]) )/ np.cos(x)  # d^2y/dx^2 = -y
    return dydx

def boundary_conditions3(ya, yb):
    return np.array([ya[0]-(2**(-0.25)), yb[0] - ((12**0.25)/2)])  # y(a) = 1, y(b) = 1


x_values3 = np.linspace(np.pi/4,np.pi/3, 100)
y_guess3 = np.ones((2, len(x_values3)))
sol3 = solve_bvp(ode_func3, boundary_conditions3, x_values3, y_guess3)


# In[19]:


def ode_func4(x, y):
    dydx = np.zeros((2, len(x)))
    dydx[0] = y[1]  # dy/dx = y'
    dydx[1] = 0.5-((y[1]**2)/2)-((y[0]*np.sin(x))/2)  # d^2y/dx^2 = -y
    return dydx

def boundary_conditions4(ya, yb):
    return np.array([ya[0]-2, yb[0] - 2])  # y(a) = 1, y(b) = 1


x_values4 = np.linspace(0,np.pi/2, 100)
y_guess4 = np.ones((2, len(x_values4)))
sol4 = solve_bvp(ode_func4, boundary_conditions4, x_values4, y_guess4)


# In[20]:


# Create a figure and an array of subplots
fig, axes = plt.subplots(2, 2)

# Plot something on each subplot
axes[0, 0].plot(sol1.x, sol1.y[0])
axes[0, 0].grid()
axes[0, 0].set_title("y''=-e^(-2y)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y(x)")

axes[0, 1].plot(sol2.x, sol2.y[0])
axes[0, 1].grid()
axes[0, 1].set_title("y''=y'cos(x)-ylny")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y(x)")

axes[1, 0].plot(sol3.x, sol3.y[0])
axes[1, 0].grid()
axes[1, 0].set_title("y''=-(2(y')^3+(y^2)y')sec(x)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y(x)")

axes[1, 1].plot(sol4.x, sol4.y[0])
axes[1, 1].grid()
axes[1, 1].set_title("y''=0.5-((y'^2)/2)-(ysin(x)/2)")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y(x)")
# Adjust layout to prevent overlap of titles
plt.tight_layout()
# Display the plot
plt.savefig("Plot9")
plt.show()


# In[ ]:




