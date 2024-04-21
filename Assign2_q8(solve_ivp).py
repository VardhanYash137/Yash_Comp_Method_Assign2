#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func1(t,y):
    return (t*np.exp(3*t))-(2*y)

def true_sol1(t):
    return 1/25*(np.exp(-2*t))*(1-np.exp(5*t)+5*(np.exp(5*t)*t))


t0=0
t1=1
t_arr=np.linspace(t0,t1,20)
y0=[0]

y_arr=solve_ivp(func1,(t0,t1),y0,t_eval=t_arr).y[0]
y_true=true_sol1(t_arr)

plt.scatter(t_arr,y_arr,label="solve_ivp")
plt.plot(t_arr,y_true,"--",label="true soln")
plt.grid()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
#plt.savefig("Plot8a")
plt.show()




# In[2]:


def func2(t,y):
    return 1-((t-y)**2)

def true_sol2(t):
    return (1-(3*t)+(t**2))/(t-3)
    

t0=2
t1=2.9
t_arr=np.linspace(t0,t1,21)
y0=[1]

y_arr=solve_ivp(func2,(t0,t1),y0,t_eval=t_arr).y[0]
y_true=true_sol2(t_arr)

plt.scatter(t_arr,y_arr,label="solve_ivp")
plt.plot(t_arr,y_true,"--",label="true soln")
plt.grid()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
#plt.savefig("Plot8b")
plt.show()


# In[3]:


def func3(t,y):
    return 1+(y/t)

def true_sol3(t):
    return 2*t+(t*np.log(t))

t0=1
t1=2
t_arr=np.linspace(t0,t1,21)
y0=[2]

y_arr=solve_ivp(func3,(t0,t1),y0,t_eval=t_arr).y[0]
y_true=true_sol3(t_arr)

plt.scatter(t_arr,y_arr,label="solve_ivp")
plt.plot(t_arr,y_true,"--",label="true soln")
plt.grid()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
#plt.savefig("Plot8c")
plt.show()



# In[4]:


def func4(t,y):
    return np.cos(2*t)+np.sin(3*t)

def true_sol4(t):
    return (1/6)*(8-2*np.cos(3*t)+3*np.sin(2*t))

t0=0
t1=1
t_arr=np.linspace(t0,t1,21)
y0=[1]

y_arr=solve_ivp(func4,(t0,t1),y0,t_eval=t_arr).y[0]
y_true=true_sol4(t_arr)

plt.scatter(t_arr,y_arr,label="solve_ivp")
plt.plot(t_arr,y_true,"--",label="true soln")
plt.grid()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
#plt.savefig("Plot8d")
plt.show()


# In[ ]:




