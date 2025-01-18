# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:46:57 2025

@author: tahag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv('sepet.csv')
t = []
for i in range(0,7501):
    t.append([str(datas.values[i,j]) for j in range(0,20)])

from apyori import apriori
rules = apriori(t,min_support=0.01,min_confidence = 0.2,min_lift=3,min_length=2)
print(list(rules))
