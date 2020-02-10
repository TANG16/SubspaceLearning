# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:34:27 2018
PCA sklearn 人脸识别
@author: wyk
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from loadORL import loadImg

# 加载数据
split_num = 5
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)

# 数据标准化处理
sc = StandardScaler()
X_train_std = sc.fit_transform(train_imgs)
X_test_std = sc.fit_transform(test_imgs)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# PCA降维
pca = PCA(n_components=100)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# KNN 分类
#KNN = KNeighborsClassifier(n_neighbors=1)
#KNN.fit(X_train_pca, y_train)
#accuracy = KNN.score(X_test_pca, y_test)
#print(accuracy)

accs = []
for i in range(3,70):
    pca = PCA(n_components=i)
    pca.fit(X_train_std)
    X_train_pca = pca.transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train_pca, y_train)
    accuracy = KNN.score(X_test_pca, y_test)
    accs.append(accuracy)

plt.plot(accs)
df = pd.DataFrame(accs, columns=['PCA'])
df.to_csv('./PCA_' + str(split_num) + '.csv', index=False)



