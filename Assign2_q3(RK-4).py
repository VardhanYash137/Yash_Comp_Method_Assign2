#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def ode_func(x,y_vec):
    ans_arr=np.zeros(2)
    ans_arr[0]=y_vec[1]
    ans_arr[1]=(2*y_vec[1])-y_vec[0]+(x*np.exp(x))-x
    return ans_arr

def ode_solve_RK4(f,t_arr,y_arr0):
    y_mat=np.zeros((len(t_arr),len(y_arr0)))
    h=t_arr[1]-t_arr[0]
    y_mat[0,:]=y_arr0
    for i in range(len(t_arr)-1):
        k1=h*f(t_arr[i],y_mat[i,:])
        k2=h*f(t_arr[i]+(h/2),y_mat[i,:]+k1/2)
        k3=h*f(t_arr[i]+(h/2),y_mat[i,:]+k2/2)
        k4=h*f(t_arr[i]+h,y_mat[i,:]+k3)
        y_mat[i+1,:]=y_mat[i,:]+(1/6)*(k1+(2*k2)+(2*k3)+k4)
    return y_mat

def euler_method(f,t_arr,y0):
    y_arr=np.zeros((len(t_arr),2))
    h=t_arr[1]-t_arr[0]
    y_arr[0]=y0
    for i in range(len(t_arr)-1):
        y_arr[i+1]=y_arr[i]+(h*f(t_arr[i],y_arr[i]))
    return y_arr

h_arr=[0.02,0.05,0.1]
for h in h_arr:
    a=0
    b=1
    y_arr0=[0,0]
    x_arr=np.arange(a,b+h,h)
    y_arr=ode_solve_RK4(ode_func,x_arr,y_arr0)[:,0]
    plt.plot(x_arr,y_arr)
    plt.scatter(x_arr,y_arr,marker=".",label="h="+str(h))
    
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("y''-2y'+2=x(e^x)-x using RK4 Method")
plt.legend()
plt.grid()
#plt.savefig("plot3")
plt.show()


# In[ ]:




