# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import nltk


comments = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip',sep=',')


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')

from nltk.corpus import stopwords

comp = []

for i in range(716):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])  # regular expressions kullanarak imla işaretlerinin yerine boşluk koyduk böylelikle kelimleri ayırt edebiliriz
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))] #this that gibi direkt olarak anlam ifade etmeyen kelimeleri çıkarma işlemi
    comment = ' '.join(comment)
    comp.append(comment)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(comp).toarray() #independent variable
y =  comments.iloc[:,1].values #dependent variable


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20 ,random_state=0)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)