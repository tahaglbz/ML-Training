#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

# Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Veri Yükleme
veriler = pd.read_csv('satislar.csv')

# Verilerin ayrıştırılması
aylar = veriler[['Aylar']]
print("Aylar:")
print(aylar)

satislar = veriler[['Satislar']]
print("\nSatışlar:")
print(satislar)

# Verilerin eğitim ve test setlerine bölünmesi
from sklearn.model_selection import train_test_split  # Güncel modül burası!
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

# Model inşası (Linear Regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Tahminler
tahmin = lr.predict(x_test)

# Eğitim verilerini sıralama (grafik için)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

# Grafik Çizimi
plt.plot(x_train, y_train, label="Gerçek Satışlar")
plt.plot(x_test, lr.predict(x_test), label="Tahmin Edilen Satışlar", linestyle="--")
plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.legend()
plt.show()
