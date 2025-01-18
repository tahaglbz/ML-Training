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
clicked = [0] * d
choosen = [0]
for n in range(1,N):
    ad=0
    max_ucb=0
    for i in range(0,d):
        if clicked[i] > 0:
            average = rewards[i] / clicked[i]
            delta = math.sqrt(3/2* math.log(n/clicked[i]))
            ucb = average + delta
        else:
            ucb=N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad =  i
    
    choosen.append(ad)
    clicked[ad] = clicked[ad]+1
    rew = datas.values[n,ad]
    rewards[ad] = rewards[ad] + rew
    summ = summ+rew  
    
print('Summation for rewards')
print(summ)
plt.hist(choosen)
plt.show()