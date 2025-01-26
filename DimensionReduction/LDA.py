# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:58:21 2025

@author: tahag
"""
#PCA = PRINCIPAL  COMPONENT ANALYSIS


import pandas as pd

datas = pd.read_csv('Wine.csv')
X = datas.iloc[:,0:13].values
y = datas.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)


#pca dönüşümünden önce gelen lr
from sklearn.linear_model import LogisticRegression
# PCA dönüşümünden önce gelen LR
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)  

# PCA dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train) 
y_pred2 = classifier2.predict(X_test2) 


from sklearn.metrics import confusion_matrix

#actual / PCA olmadan çıkan sonuç
cm = confusion_matrix(y_test, y_pred)
print('pcasiz')
print(cm)
#actual / PCA sonrası çıkan sonuç
cm2 = confusion_matrix(y_test, y_pred2)
print('pca ile')
print(cm2)
#PCA sonrası /PCA öncesi
cm3 = confusion_matrix(y_pred, y_pred2)
print('pcali ve pcasiz')
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train,y_train)

X_train_lda = lda.transform(X_test)

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

y_pred_lda = classifier_lda.predict(X_test_lda)

print('lda ve orjinal')

cm4= confusion_matrix(y_pred, y_pred_lda)
print(cm4)








 

