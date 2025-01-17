# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:44:25 2025

@author: tahag
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

datas = pd.read_csv('musteriler.csv')

X = datas.iloc[:,3:].values

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,init='k-means++')
km.fit(X)

print(km.cluster_centers_)
result=[]
for i in range(1,11):
    kmean = KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmean.fit(X)
    result.append(kmean.inertia_)
    
    
plt.plot(range(1,11), result)
plt.show()

#hiyerarşik kümeleme

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, linkage='manhattan')

Y_predict=ac.fit_predict(X)
print(Y_predict)
