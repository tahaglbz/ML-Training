import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  # Doğru modülü import ediyoruz
from sklearn.linear_model import LinearRegression

# Veriyi yükle
datas = pd.read_csv('odev_tenis.csv')

# Tüm sütunları LabelEncoder ile encode etme
datas2 = datas.apply(LabelEncoder().fit_transform)

# İlk sütun (örneğin "Outlook") için OneHotEncoder uygulama
c = datas.iloc[:, :1]  # İlk sütunu seçiyoruz
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])  # 0. sütunu encode ediyoruz
    ],
    remainder='passthrough'
)

c = ct.fit_transform(c)

# Yeni oluşturulan sütunları DataFrame'e dönüştürme
weather = pd.DataFrame(data=c, index=range(len(datas)), columns=['o', 'r', 's'])

# Orijinal verinin kalan sütunlarını ekleme
lastdatas = pd.concat([weather, datas.iloc[:, 1:3].reset_index(drop=True)], axis=1)

# Son iki sütunu ve diğer verileri birleştiriyoruz
lastdatas = pd.concat([datas2.iloc[:, -2:], lastdatas], axis=1)

# Veriyi eğitim ve test olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(
    lastdatas.iloc[:, :-1],  # Bağımsız değişkenler
    lastdatas.iloc[:, -1:],  # Bağımlı değişkenler
    test_size=0.33,          # Test veri oranı
    random_state=0           # Rastgelelik için sabit tohum
)

# Lineer regresyon modeli oluşturma
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Tahmin yapma
y_pred = regressor.predict(x_test)

# Sonuçları yazdırma
print("Gerçek Değerler:", y_test.values.flatten())
print("Tahmin Edilen Değerler:", y_pred.flatten())
