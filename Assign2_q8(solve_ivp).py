#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func1(t,y):
    return (t*np.exp(3*t))-(2*y)

def true_sol1(t):
    return 1/25*(np.exp(-2*t))*(1-np.exp(5*t)+5*(np.exp(5*t)*t))


t_arr1=np.linspace(0,1,20)
y_arr1=solve_ivp(func1,(0,1),[0],t_eval=t_arr1).y[0]
y_true1=true_sol1(t_arr1)


# In[14]:


def func2(t,y):
    return 1-((t-y)**2)

def true_sol2(t):
    return (1-(3*t)+(t**2))/(t-3)

t_arr2=np.linspace(2,2.9,20)
y_arr2=solve_ivp(func2,(2,3),[1],t_eval=t_arr2).y[0]
y_true2=true_sol2(t_arr2)


# In[15]:


def func3(t,y):
    return 1+(y/t)

def true_sol3(t):
    return 2*t+(t*np.log(t))

t_arr3=np.linspace(1,2,20)
y_arr3=solve_ivp(func3,(1,2),[2],t_eval=t_arr3).y[0]
y_true3=true_sol3(t_arr3)


# In[16]:


def func4(t,y):
    return np.cos(2*t)+np.sin(3*t)

def true_sol4(t):
    return (1/6)*(8-2*np.cos(3*t)+3*np.sin(2*t))

t_arr4=np.linspace(0,1,20)
y_arr4=solve_ivp(func4,(0,1),[1],t_eval=t_arr4).y[0]
y_true4=true_sol4(t_arr4)


# In[34]:


# Create a figure and an array of subplots
fig, axes = plt.subplots(2, 2)

# Plot something on each subplot
axes[0, 0].scatter(t_arr1,y_arr1,label="solve_ivp soln")
axes[0, 0].plot(t_arr1,y_true1,label="true soln")
axes[0, 0].grid()
axes[0, 0].set_title("y'=te^(3t)-2y")
axes[0, 0].set_xlabel("t")
axes[0, 0].set_ylabel("y(t)")
axes[0, 0].legend()

axes[0, 1].scatter(t_arr2,y_arr2,label="solve_ivp soln")
axes[0, 1].plot(t_arr2,y_true2,label="true soln")
axes[0, 1].grid()
axes[0, 1].set_title("y'=1-(t-y)^2")
axes[0, 1].set_xlabel("t")
axes[0, 1].set_ylabel("y(t)")
axes[0, 1].legend()

axes[1, 0].scatter(t_arr3,y_arr3,label="solve_ivp soln")
axes[1, 0].plot(t_arr3,y_true3,label="true soln")
axes[1, 0].grid()
axes[1, 0].set_title("y'-1+(y/t)")
axes[1, 0].set_xlabel("t")
axes[1, 0].set_ylabel("y(t)")
axes[1, 0].legend()


axes[1, 1].scatter(t_arr4,y_arr4,label="solve_ivp soln")
axes[1, 1].plot(t_arr4,y_true4,label="true soln")
axes[1, 1].grid()
axes[1, 1].set_title("y'=cos(2t)+sin(3t)")
axes[1, 1].set_xlabel("t")
axes[1, 1].set_ylabel("y(t)")
axes[1, 1].legend()

# Adjust layout to prevent overlap of titles
plt.tight_layout()
# Display the plot
#plt.savefig("Plot8")
plt.show()

