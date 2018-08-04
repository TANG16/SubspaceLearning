# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:15:29 2018
LDA 线性判别分析:
    1.数据标准化
    2.计算每一类别的均值
    3.计算类间离散度矩阵Sb和类内离散度矩阵Sw
    4.分解inv(Sb)Sw,求特征值与特征向量
    5.选择前K个最大的特征值对应的特征向量组成W
    6.用W对数据进行降维
@author: wyk
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载数据，进行预处理（标准化，0均值，1方差）
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-'
'learning-databases/wine/wine.data', header=None)

# 数据预处理，分割测试集与训练集，数据标准化
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 计算类内离散度与类间离散度矩阵
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
#    print('mean vector %s: %s' %(label, mean_vecs[label-1]))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
    
print('类内离散度矩阵形状：' + str(S_W.shape))

mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec-mean_overall).T)

print('类间离散度矩形状：' + str(S_B.shape))

# 分解矩阵 inv(S_W)S_B
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
                for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# 画判别信息能量图
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha=0.5, align='center',
        label='individual discr')
plt.step(range(1,14), cum_discr, where='mid',
         label='cumulative discr')
plt.ylabel('discr ratio')
plt.xlabel('linear discr')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

# 利用判别信息创建投影矩阵 W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('投影矩阵 W:', w)

# 计算训练集上降维后的结果并可视化
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1],
                c=c, label=l, marker=m)






