# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:33:01 2018
MDP 人脸识别
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
def within_class(X_train_std, y_train, class_num, k):
    m = X_train_std.shape[1]
    S_W = np.zeros((m, m))
    mean_vectors = class_mean(X_train_std, y_train, class_num)
#    计算类内离散度
    for cl, mv in zip(range(1, class_num+1), mean_vectors):
        samples = X_train_std[y_train==cl]
        sample_num = samples.shape[0]
        dist = np.arange(0,sample_num)
#        计算每一类样本距离类均值的距离
        for j in range(sample_num):
            dist[j] = (samples[j]-mv).dot((samples[j]-mv).T)
        index = np.argsort(-dist)
        class_sc_mat = np.zeros((m, m))
        for i in range(k):
            sample, mv = (samples[index[i]]).reshape(m, 1), mv.reshape(m, 1)
            class_sc_mat += (sample-mv).dot((sample-mv).T)
        S_W += class_sc_mat
#    print(S_W)
    return S_W

# 类间离散度
def between_class(X_train_std, y_train, class_num, k):
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

# MFA 降维
def mdp(X_train_std, y_train, X_test_std, y_test, class_num=40, n_component=39):
    m = X_train_std.shape[1]
    S_W = within_class(X_train_std, y_train, class_num, k=1)
    S_B = between_class(X_train_std, y_train, class_num, k=15)
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
    w = mdp(X_train_pca, y_train, X_test_pca, y_test, n_component=i)
    X_train_mdp = X_train_pca.dot(w)
    X_test_mdp = X_test_pca.dot(w)
    KNN = KNeighborsClassifier(n_neighbors=1)
    KNN.fit(X_train_mdp, y_train)
    accuracy = KNN.score(X_test_mdp, y_test)
    accs.append(accuracy)

plt.plot(accs)
df = pd.DataFrame(accs, columns=['MDP'])
df.to_csv('./MDP_' + str(split_num) + '.csv', index=False)


