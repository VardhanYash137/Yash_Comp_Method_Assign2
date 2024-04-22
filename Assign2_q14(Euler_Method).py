#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dydt(t,y):
    y_prime_arr=np.zeros(len(y))
    y_prime_arr[0]=y[1]
    y_prime_arr[1]=(t*np.log(t))+(2*y[1]/t)-(2*y[0]/t**2)
    return y_prime_arr

def euler_method(f,t_arr,y0):
    y_arr=np.zeros((len(t_arr),len(y0)))
    h=t_arr[1]-t_arr[0]
    y_arr[0]=y0
    for i in range(len(t_arr)-1):
        y_arr[i+1]=y_arr[i]+(h*f(t_arr[i],y_arr[i]))
    return y_arr


def true_sol(t):
    return (7*t)/4 + (t**3)/2 * np.log(t) - (3/4)*t**3
                    
a=1
b=2
y0=[1,0]
h=0.001
t_arr=np.arange(a,b+h,h)
                    
y_arr=euler_method(dydt,t_arr,y0)[:,0]
true_sol=true_sol(t_arr)
plt.plot(t_arr,true_sol,label="true soln")
plt.plot(t_arr,y_arr,"--",label="Forward Euler Method")

plt.legend()
plt.grid()
plt.savefig("Plot14")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("(t^2)y''-2ty'+2y=(t^3)ln(t)")
#plt.savefig("Plot14")
plt.show()


# In[ ]:




