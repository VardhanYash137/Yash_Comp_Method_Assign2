#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def dxdt(x,t):
    return (1/(x**2+t**2))

def dtheta(th,x):
    return (1/(x**2+(np.tan(th)**2)))*(1/(np.cos(th)**2))

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

th0=0
th_end=((np.pi/2)-0.001)
th_arr=np.linspace(th0,th_end,100)

y0=1

x_arr=ode_solve_RK4(dtheta,th_arr,[y0])[:,0]


t_arr=np.tan(th_arr)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2)

# Plot data on the first subplot
ax1.plot(th_arr, x_arr, color='r')
ax1.set_ylabel('x(theta)')
ax1.grid()
ax1.set_xlabel("theta")
ax1.set_title("dx/d(theta)=sec^2(theta)/(x^2+tan^2(theta))")


# Plot data on the second subplot
ax2.plot(t_arr, x_arr, color='b')
ax2.set_ylabel('x(t)')
ax2.grid()
ax2.set_xlabel("t")
ax4.set_title("x'=1/(x^2+t^2)")

# Adjust layout
plt.tight_layout()
#plt.savefig("Plot9a")
# Show plot
plt.show()


t_val=3.5*(10**6)
theta_val=np.arctan(t_val)
print(theta_val)
th_arr1=np.linspace(th0,theta_val,100)
t_arr1=np.tan(th_arr1)
x_arr1=ode_solve_RK4(dtheta,th_arr1,[y0])[:,0]
print("Value at t=",t_val,": ",x_arr[-1])


# Create subplots
fig, (ax3, ax4) = plt.subplots(2)

# Plot data on the first subplot
ax3.plot(th_arr1, x_arr1, color='r')
ax3.set_ylabel('x(theta)')
ax3.grid()
ax3.set_xlabel("theta")
ax3.set_title("dx/d(theta)=sec^2(theta)/(x^2+tan^2(theta))")


# Plot data on the second subplot
ax4.plot(t_arr1, x_arr1, color='b')
ax4.set_ylabel('x(t)')
ax4.grid()
ax4.set_xlabel("t")
ax4.set_title("x'=1/(x^2+t^2)")

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()




# In[ ]:




