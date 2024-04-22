#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

def diff_eqn(t,u):
    u_prime_arr=np.zeros(len(u))
    u_prime_arr[0]=u[0]+(2*u[1])-(2*u[2])+np.exp(-t)
    u_prime_arr[1]=u[1]+u[2]-(2*np.exp(-t))
    u_prime_arr[2]=u[0]+(2*u[1])+np.exp(-t)
    return u_prime_arr

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

t0=0
t_end=1
t_arr=np.linspace(t0,t_end,100)

u_arr0=[3,-1,1]
u_mat=ode_solve_RK4(diff_eqn,t_arr,u_arr0)

plt.plot(t_arr,u_mat[:,0],label="u1")
plt.plot(t_arr,u_mat[:,1],label="u2")
plt.plot(t_arr,u_mat[:,2],label="u3")
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Coupled System using RK4")
#plt.savefig("Plot12")
plt.show()


# In[ ]:





# In[ ]:




