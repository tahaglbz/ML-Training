# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

# 1. Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Veri Ön İşleme
# 2.1. Veri Yükleme
veriler = pd.read_csv('veriler.csv')
print(veriler)

# Kategorik -> Sayısal Dönüşüm (Ülke sütunu)
ulke = veriler.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# Kategorik -> Sayısal Dönüşüm (Cinsiyet sütunu)
c = veriler.iloc[:, -1:].values
print(c)

c[:, -1] = le.fit_transform(veriler.iloc[:, -1])
print(c)

c = ohe.fit_transform(c).toarray()
print(c)

# Yaş verilerini çıkarma (Boy, Kilo, Yaş sütunları)
Yas = veriler.iloc[:, 1:4].values
print(Yas)

# Numpy dizilerini DataFrame'e dönüştürme
sonuc = pd.DataFrame(data=ulke, index=range(veriler.shape[0]), columns=['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(veriler.shape[0]), columns=['boy', 'kilo', 'yas'])
print(sonuc2)

sonuc3 = pd.DataFrame(data=c[:, :1], index=range(veriler.shape[0]), columns=['cinsiyet'])
print(sonuc3)

# DataFrame birleştirme işlemi
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

# Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# Model oluşturma
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Tahmin
y_pred = regressor.predict(x_test)

# Boyut uyuşmazlığı düzeltildi
boy = s2.iloc[:, 3:4].values  # Sadece 'boy' sütununu alıyoruz
print(boy)

# Giriş değişkenleri (X) ve hedef değişken (Y) tanımlandı
sol = s2.iloc[:, :3]  # İlk üç sütun (fr, tr, us)
sag = s2.iloc[:, 4:]  # 'cinsiyet' sütunu
veri = pd.concat([sol, sag], axis=1)  # Giriş verileri (X)

# Verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

# Model oluşturma
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Tahmin
y_pred = regressor.predict(x_test)

r2 = LinearRegression()
r2.fit(x_train, y_train)

# Tahmin
y_pred = r2.predict(x_test)



import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(boy,X_l).fit()

print(model.summary())

#üsttekinden daha düşük P sonucu verdiği için daha başarılı
X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(boy,X_l).fit()

print(model.summary())




