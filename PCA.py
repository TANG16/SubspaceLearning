# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:54:34 2018
PCA 实验
@author: wyk
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载数据，从 csv文件读取
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-'
'learning-databases/wine/wine.data', header=None)

# 数据预处理，分割测试集与训练集，数据标准化
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 计算协方差矩阵，进行特征值，特征向量分解
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# 显示各个特征值所占的比重
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in 
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='单个特征值比重')
plt.step(range(1, 14), cum_var_exp, where='mid',
        label='前n个特征值比重')
plt.ylabel('比例')
plt.xlabel('主成分')
plt.legend(loc='best')
plt.show()

# 按照特征值对特征值，特征向量排序
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# 选择两个特征向量，便于可视化降维后的结果
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
              eigen_pairs[1][1][:, np.newaxis]))

# 计算训练集降维的结果
X_train_pca = X_train_std.dot(w)
print(X_train_pca)

# 可视化降维后的结果
colors = ['r', 'g', 'b']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    print('%s %s %s' %(l, c, m))
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='best')
plt.show()






