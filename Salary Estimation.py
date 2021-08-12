# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 08:17:04 2021

@author: FARZAN
"""
#import the libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
#%%
#Importing the data from desktop folder
dataset= pd.read_csv("Salary_DataSet.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#%%
#Training and Testing Data (Dividing into 2 parts)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.5, random_state=0)
#%%
#Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
#%%
#for predict the test values
y_pre=reg.predict(x_test)
accuracy= reg.score(x_test,y_test)
print(accuracy*100)
yp=reg.predict(x_train)
#%%
#Visualisation of Training Data

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train, yp, color="red")
plt.title("Linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salaries of Employee")
plt.show()
#%%
#Visualisation of Testing Data
plt.scatter(x_test,y_test,color="black")
plt.plot(x_test, y_pre, color="red")
plt.title("Linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Salaries of Employee")
plt.show()



