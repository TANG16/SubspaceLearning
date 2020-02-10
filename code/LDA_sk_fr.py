# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:02:21 2018
sklearn实现 LDA人脸识别
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

#先进行 PCA处理，以免维数过高
pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


accs = []
for i in range(3,70):
    lda = LinearDiscriminantAnalysis(n_components=i)
    lda.fit(X_train_pca, y_train)
    X_train_lda = lda.transform(X_train_pca)
    X_test_lda = lda.transform(X_test_pca)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train_lda, y_train)
    accuracy = KNN.score(X_test_lda, y_test)
    accs.append(accuracy)
    
plt.plot(accs)
df = pd.DataFrame(accs, columns=['LDA_sk'])
df.to_csv('./LDA_sk_' + str(split_num) + '.csv', index=False)









