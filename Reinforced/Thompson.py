
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:34:30 2025

@author: tahag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

"""

 RANDOM SELECTION 
N = 10000
d=10


 
result = 0
selected = []


for n in range(0,N):
    ad = random.randrange(d)
    selected.append(ad)
    reward = datas.values[n,ad] 
    result = result + reward


plt.hist(selected)
plt.show()
"""

#UCB

N = 10000
d = 10
#Ri(n)
rewards = [0] * d #başta tüm ödüller 0 
#Ni(n)
summ =0
choosen = [0]
ones = [0]*d
zeros = [0]*d


for n in range(1,N):
    ad=0
    max_th=0
    for i in range(0,d):
        randbeta = random.betavariate(ones[i]+1, zeros[i]+1)
        if randbeta > max_th:
            max_th=randbeta
            ad=i
    
    choosen.append(ad)
    
    rew = datas.values[n,ad]
    
    if rew ==1:
        ones[ad] = ones[ad]+1
    else:
        zeros[ad] = zeros[ad]+1
    
    rewards[ad] = rewards[ad] + rew
    summ = summ+rew  
sad    
print('Summation for rewards')
print(summ)
plt.hist(choosen)
plt.show()