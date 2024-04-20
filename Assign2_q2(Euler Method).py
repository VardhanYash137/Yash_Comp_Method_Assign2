#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dydt(t,y):
    return (y/t)-((y/t)**2)

def euler_method(f,t_arr,y0):
    y_arr=np.zeros((len(t_arr),len(y0)))
    h=t_arr[1]-t_arr[0]
    y_arr[0]=y0
    for i in range(len(t_arr)-1):
        y_arr[i+1]=y_arr[i]+(h*f(t_arr[i],y_arr[i]))
    return y_arr

def true_sol(t):
    return t/(1+np.log(t))

a=1
b=2
y0=[1]
h=0.1
t_arr=np.arange(a,b+h,h)

y_arr=euler_method(dydt,t_arr,y0)[:,0]
true_sol=np.vectorize(true_sol)
y_true=true_sol(t_arr)
abs_err=np.abs(y_arr-y_true)
rel_err=np.divide(abs_err,y_true) 
ans_mat=np.column_stack((t_arr,y_arr,y_true,abs_err,rel_err))
df1=pd.DataFrame(ans_mat,columns=["t","y_euler","y_true","abs err","rel err"])
print(df1)

# Plot DataFrame as a table
plt.figure(figsize=(8, 6))
table = plt.table(cellText=df1.values, colLabels=df1.columns, loc='center')

# Hide axes
ax = plt.gca()
ax.axis('off')

# Save the table as an image
#plt.savefig('table_image.png', bbox_inches='tight', pad_inches=0.05)
plt.show()

plt.plot(t_arr,y_arr)
plt.scatter(t_arr,y_arr,marker=".",label="Euler Method")
plt.plot(t_arr,y_true)
plt.scatter(t_arr,y_true,marker=".",label="true soln")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid()
plt.legend()
plt.title("Solving y'=(y/t)-((y/t)^2) using Euler Method")
#plt.savefig("plot2")
plt.show()


# In[ ]:




