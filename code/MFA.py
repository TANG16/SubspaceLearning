# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:29:57 2018
MFA ORL人脸识别
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
def within_laplace(X_train_std, y_train, class_num, k):
    n = X_train_std.shape[0]
    W = np.zeros((n, n))
#    计算权重矩阵W
    for i in range(n):
        dist = np.arange(0,n)
        for j in range(n):
            dist[j] = (X_train_std[i]-X_train_std[j]).dot((X_train_std[i]-X_train_std[j]).T)
        index = np.argsort(dist)
        count = 0
        j = 0
        while (count < k):
            if y_train[index[j]] == y_train[i]:    
                W[i][index[j]] = 1
                count += 1
            j += 1
#    计算Laplace矩阵
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        D[i][i] = np.sum(W[i])
    L = D - W
#    print(W)
    return L

# 类间离散度
def between_laplace(X_train_std, y_train, class_num, k):
    n = X_train_std.shape[0]
    W = np.zeros((n, n))
#    计算权重矩阵W
    for i in range(n):
        dist = np.arange(0,n)
        for j in range(n):
            dist[j] = (X_train_std[i]-X_train_std[j]).dot((X_train_std[i]-X_train_std[j]).T)
        index = np.argsort(dist)
        count = 0
        j = 0
        while (count < k):
            if y_train[index[j]] != y_train[i]:    
                W[i][index[j]] = 1
                count += 1
            j += 1
#    计算Laplace矩阵
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        D[i][i] = np.sum(W[i])
    L = D - W
#    print(W)
    return L

# MFA 降维
def mfa(X_train_std, y_train, X_test_std, y_test, class_num=40, n_component=39):
    m = X_train_std.shape[1]
    L_W = within_laplace(X_train_std, y_train, class_num, k=split_num-1)
    L_B = between_laplace(X_train_std, y_train, class_num, k=split_num*split_num+2)
    S_W = (X_train_std.T).dot(L_W).dot(X_train_std)
    S_B = (X_train_std.T).dot(L_B).dot(X_train_std)
#    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eig_vals, eig_vecs = np.linalg.eig(S_B-S_W)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    vecs = [eig_pairs[i][1].reshape(m, 1) for i in range(n_component)]
    W = np.hstack(vecs)
    return W


# 加载数据
split_num = 4
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)

# 数据标准化处理
sc = StandardScaler()
X_train_std = sc.fit_transform(train_imgs)
X_test_std = sc.fit_transform(test_imgs)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 降维
#先进行 PCA处理，以免维数过高
pca = PCA(n_components=80)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#w = mfa(X_train_pca, y_train, X_test_pca, y_test, n_component=50)
#X_train_lda = X_train_pca.dot(w)
#X_test_lda = X_test_pca.dot(w)

# KNN 分类
#KNN = KNeighborsClassifier(n_neighbors=1)
#KNN.fit(X_train_lda, y_train)
#accuracy = KNN.score(X_test_lda, y_test)
#print(accuracy)

accs = []
for i in range(3,70):
    w = mfa(X_train_pca, y_train, X_test_pca, y_test, n_component=i)
    X_train_mfa = X_train_pca.dot(w)
    X_test_mfa = X_test_pca.dot(w)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train_mfa, y_train)
    accuracy = KNN.score(X_test_mfa, y_test)
    accs.append(accuracy)

plt.plot(accs)
df = pd.DataFrame(accs, columns=['MFA'])
df.to_csv('./MFA_' + str(split_num) + '.csv', index=False)



