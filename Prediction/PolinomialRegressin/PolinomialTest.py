# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:14:48 2024

@author: tahag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
datas = pd.read_csv("maaslar.csv")

#dataframe dilimleme slice
x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

#numpy array dönüşümü
X = x.values
Y = y.values

# Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)
 

#Polynomial Regression
#non linear model
from sklearn.preprocessing import PolynomialFeatures
 #4. derece test
pol_reg = PolynomialFeatures(degree=4)
x_poly = pol_reg.fit_transform(X)
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly, y)


#visualising
plt.scatter(X,Y,color="red")
plt.plot(x, lin_reg2.predict(pol_reg.fit_transform(X)),color="blue")

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color = 'blue')
plt.show()