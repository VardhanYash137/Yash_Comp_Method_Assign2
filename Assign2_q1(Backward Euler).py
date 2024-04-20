#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Problem-1

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def func1(x,y):
    return -9*y

def euler_method_implicit(f,t_arr,y0): 
    '''
    f     : differential eqn y' function
    t_arr : mesh points 
    y0    : Initial Condition   
    '''
    
    def func_implicit1(y_next,args):  #implicit euler fn y(i+1)-y(i)-h*f(t(i+1),y(i+1))=0
        t=args[0]
        y_prev=args[1]
        return y_next-y_prev-(h*f(t,y_next))
    
    def secant_method(func, x0, x1,args, tol=1e-6, max_iter=100):
        iter_count = 0
        root_list=[]
        root_list.append(x0)
        root_list.append(x1)
        while True:
            # Compute the next approximation
            x_next = x1 - func(x1,args) * (x1 - x0) / (func(x1,args) - func(x0,args))
            root_list.append(x_next)
            # Check convergence
            if abs(x_next - x1) < tol:
                return x_next,root_list
            # Update values for the next iteration
            x0, x1 = x1, x_next
            iter_count += 1
            if iter_count >= max_iter:
                print("Maximum iterations reached without convergence.")
                return x_next,root_list  # Return the last approximation if max_iter is reached

    
    y_arr=np.zeros(len(t_arr))  #initialise
    h=t_arr[1]-t_arr[0]         
    y_arr[0]=y0
    for i in range(len(t_arr)-1):
        #using secant method to solve implicit euler eqn
        root=secant_method(func_implicit1,y0,y0+1,args=(t_arr[i+1],y_arr[i]))[0]
        y_arr[i+1]=root
    return y_arr

def analytic_soln(x):
    return np.exp((-9*x)+1)

h_arr=[0.2,0.1,0.05,0.025]
for h in h_arr:
    a=0              #initial point
    b=1              #final point
    y0=np.exp(1)     #initial condition
    x_arr=np.arange(a,b+h,h)
    y_arr=euler_method_implicit(func1,x_arr,y0)
    plt.plot(x_arr,y_arr)
    plt.scatter(x_arr,y_arr,marker=".",label="h="+str(h))
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("y'=-9y (Backward Integration using Euler Method)")
plt.grid()
plt.plot(x_arr,analytic_soln(x_arr),label="analytic")
plt.legend()
#plt.savefig("C:\\Users\\vyash\\Yash\\Tifr_coursework\\TifrSem2\\Computational Physics\\Assignments\\Assignment-2\\Plots")
plt.show()


# In[11]:


#Problem-2

def func2(x,y):
    return -20*((y-x)**2)+(2*x)

h_arr=[0.01,0.02,0.05]  #h=0.094 onwards give wrong result
for h in h_arr:
    a=0
    b=1
    y0=1/3
    x_arr=np.arange(a,b+h,h)
    y_arr=euler_method_implicit(func2,x_arr,y0)
    plt.plot(x_arr,y_arr)
    plt.scatter(x_arr,y_arr,marker=".",label="h="+str(h))
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("y'=-20(y-x)^2+2x (Implicit Euler Method)")
plt.grid()
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




