# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 08:49:55 2018
LDA ORL人脸识别
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

# 类均值
def class_mean(X_train_std, y_train, class_num):
    mean_vectors = []
    for cl in range(1, class_num+1):
        mean_vectors.append(np.mean(X_train_std[y_train==cl], axis=0))
#    print(mean_vectors)
    return mean_vectors

# 类内离散度
def within_class(X_train_std, y_train, class_num):
    m = X_train_std.shape[1]
    S_W = np.zeros((m, m))
    mean_vectors = class_mean(X_train_std, y_train, class_num)
    for cl, mv in zip(range(1, class_num+1), mean_vectors):
        class_sc_mat = np.zeros((m, m))
        for row in X_train_std[y_train == cl]:
            row, mv = row.reshape(m, 1), mv.reshape(m, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat
#    print(S_W)
    return S_W

# 类间离散度
def between_class(X_train_std, y_train, class_num):
    m = X_train_std.shape[1]
    all_mean = np.mean(X_train_std, axis=0)
    S_B = np.zeros((m, m))
    mean_vectors = class_mean(X_train_std, y_train, class_num)
    for cl, mean_vec in enumerate(mean_vectors):
        n = X_train_std[y_train == cl+1,:].shape[0]
        mean_vec = mean_vec.reshape(m, 1)
        all_mean = all_mean.reshape(m, 1)
        S_B += n * (mean_vec - all_mean).dot((mean_vec - all_mean).T)
#    print(S_B)
    return S_B

# MMC 降维
def lda(X_train_std, y_train, X_test_std, y_test, class_num=40, n_component=39):
    m = X_train_std.shape[1]
    S_W = within_class(X_train_std, y_train, class_num)
    S_B = between_class(X_train_std, y_train, class_num)
#    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    U, sigma, VT = np.linalg.svd(np.linalg.inv(S_W).dot(S_B))
    eig_vals = sigma
    eig_vecs = U
#    print('eig_vals[0]:%s, eig_vecs[0]:%s' %(eig_vals[0],eig_vecs[:,i]))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    vecs = [eig_pairs[i][1].reshape(m, 1) for i in range(n_component)]
    W = np.hstack(vecs)
    return W


# 加载数据
split_num = 3
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)

# 数据标准化处理
sc = StandardScaler()
X_train_std = sc.fit_transform(train_imgs)
X_test_std = sc.transform(test_imgs)
#X_train_std = train_imgs
#X_test_std = test_imgs
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 降维
#先进行 PCA处理，以免维数过高
pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# KNN 分类
#KNN = KNeighborsClassifier(n_neighbors=1)
#KNN.fit(X_train_lda, y_train)
#accuracy = KNN.score(X_test_lda, y_test)
#print(accuracy)

accs = []
for i in range(3,40):
    w = lda(X_train_pca, y_train, X_test_pca, y_test, n_component=i)
    X_train_lda = X_train_pca.dot(w)
    X_test_lda = X_test_pca.dot(w)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train_lda, y_train)
    accuracy = KNN.score(X_test_lda, y_test)
    accs.append(accuracy)
    
plt.plot(accs)
df = pd.DataFrame(accs, columns=['LDA'])
df.to_csv('./LDA_' + str(split_num) + '.csv', index=False)




