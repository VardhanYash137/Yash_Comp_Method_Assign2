#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#defining constants
g=10

# Define the ordinary differential equations
def ode_fun(t,y):
    dy = y[1]
    d2y = -g
    return [dy, d2y]

def exact_soln(t_arr,args):
    t0=args[0]
    x0=args[1]
    t1=args[2]
    x1=args[3]
    u0=(x1+(0.5*g*(t1**2)))/t1
    return x0+(u0*t_arr)-(0.5*g*(t_arr**2))

def secant_method_shooting(func, x0, x1,args,t_arr, tol=1e-6, max_iter=100):
    iter_count = 0
    
    #list for storing values of y derivative
    root_list=[]
    root_list.append(x0)
    root_list.append(x1)
    
    #defining matrix for storing soln of ivp for all values of y derivative
    y_mat=np.zeros((max_iter,len(t_arr)))
    y_mat[0,:]=func(x0,args,key=1)
    y_mat[1,:]=func(x1,args,key=1)
    
    while True:        
        # Compute the next approximation
        x_next = x1 - func(x1,args) * (x1 - x0) / (func(x1,args) - func(x0,args))
        y_mat[iter_count+2,:]=func(x_next,args,key=1)
        root_list.append(x_next)
        
        # Check convergence
        if abs(x_next - x1) < tol:
            return x_next,y_mat,root_list
        
        # Update values for the next iteration
        x0, x1 = x1, x_next
        
        iter_count += 1
        if iter_count >= max_iter:
            print("Maximum iterations reached without convergence.")
            return x_next,root_list  # Return the last approximation if max_iter is reached

def shooting_method(ode_fun,args,t_arr,x0,x1,max_iterations=10):
    
    y_mat=np.zeros((max_iterations,len(t_arr)))
    
    def deviation_func(yderiv,args,key=None):
        t0=args[0]
        x0=args[1]
        t1=args[2]
        x1=args[3]
        y0 = [x0, yderiv]
        t_span=(t0,t1)
        sol = solve_ivp(ode_fun,t_span, y0 ,t_eval=t_arr)
        if key!=None:
            return sol.y[0]
        return float(sol.y[0][-1]-x1)
   
    t_span=(args[0],args[2])
    y_prime0,y_mat,root_list=secant_method_shooting(deviation_func,x0,x1,args,t_arr,max_iter=max_iterations)
    y_mat1=y_mat[0:len(root_list)]
    sol=solve_ivp(ode_fun,(t_arr[0],t_arr[-1]),[args[1],y_prime0],t_eval=t_arr)
    return(y_mat1,root_list)
    
    


#defining boundary values
t0=0
t1=10
x0=0
x1=0
args=[t0,x0,t1,x1]
t_arr=np.linspace(t0,t1,100,float)


y_mat,root_list=shooting_method(ode_fun,args,t_arr,10,20)
exact_soln_arr=exact_soln(t_arr,args)
y_arr_x=solve_ivp(ode_fun,(t_arr[0],t_arr[-1]),[args[1],30],t_eval=t_arr).y[0]

for i in range(len(root_list)):
    plt.plot(t_arr,y_mat[i,:],label="x'(0)="+str(root_list[i]))
plt.plot(t_arr,y_arr_x,label="x'(0)="+str(30))
plt.plot(t_arr,exact_soln_arr,"--",label="exact soln")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid()
plt.title("BVP using shooting method")
#plt.savefig("Plot6a")
plt.show()


# In[2]:


#using argmin

#defining boundary values
t0=0
t1=10
x0=0
x1=0
args=[t0,x0,t1,x1]
t_arr=np.linspace(t0,t1,100,float)

beta_min=0
beta_max=100
beta_arr=np.linspace(beta_min,beta_max,101)
deviation_arr=np.zeros(len(beta_arr))

count=0
for beta in beta_arr:
    deviation_arr[count]=((solve_ivp(ode_fun,(t0,t1),(x0,beta),t_eval=t_arr).y[0][-1])-x1)
    count=count+1
    
correct_index=np.argmin(np.abs(deviation_arr))
y_cor=solve_ivp(ode_fun,(t0,t1),(x0,beta_arr[correct_index]),t_eval=t_arr).y[0]
plt.plot(t_arr,y_cor)
plt.grid()
plt.title("Shooting method using np.argmin")
plt.xlabel("t")
plt.ylabel("x(t)")
#plt.savefig("Plot6b")
plt.show()


# In[ ]:




