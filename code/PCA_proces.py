# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:07:55 2018
PCA model用于数据预处理
@author: wyk
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from loadORL import loadImg

def pca_prepro(X_train_std, n_com):
    m = X_train_std.shape[1]
    # 计算协方差矩阵，进行特征值，特征向量分解
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # 按照特征值对特征值，特征向量排序
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                    for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    
    vecs = [eigen_pairs[i][1].reshape(m, 1) for i in range(n_com)]
    w = np.hstack(vecs)
    print('pca投影矩阵：')
    print(w)
    return w

split_num = 1
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)
#print(train_imgs)
w = pca_prepro(train_imgs, n_com=50)
print('投影矩阵')
print(w)



