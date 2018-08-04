# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:26:30 2018
LDA 应用到人脸识别
@author: wyk
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 加载图片,分割训练集与测试集
def loadImg(split_num):
    img_path = './ORL/s'
    classNum = 40
    imgNum = 10
#    split_num = 8
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    for i in range(1,classNum+1):
        for j in range(1, imgNum+1):
            path = img_path + str(i) + '/' + str(j) + '.bmp'
            img = mpimg.imread(path)
            if j <= split_num:
                train_imgs.append(img.flatten().tolist())
                train_labels.append(i)
            else:
                test_imgs.append(img.flatten().tolist())
                test_labels.append(i)
    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    return train_imgs, train_labels, test_imgs, test_labels

# K近邻算法预测
def KNNClassifer(K, X_train_pca, train_labels, X_test_pca, test_labels):
#    K 近邻分类器
    acc = 0
    for i in range(len(test_labels)):
        dist = []
        for j in range(len(train_labels)):
            temp = np.linalg.norm(X_test_pca[i]-X_train_pca[j])
            dist.append(temp)
#        res = np.argmin(dist)
#        print('K近邻预测label：')
        idx = np.argsort(dist)
#        print(idx)
        voteLabels = [train_labels[k] for k in idx]
        voteLabels = voteLabels[0:K]
#        print(voteLabels)
#        print('===============')
        labelCounter = Counter(voteLabels)
        top_one = labelCounter.most_common(1)
#        print(top_one)
        if test_labels[i] == top_one[0][0]:
            acc += 1
            
    print('预测正确数目：%d' %(acc))
    accuracy = acc/((10-split_num)*40)
    print('准确率：%.3f' %(accuracy))
    return accuracy


# 加载数据
split_num = 6
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)

# 数据预处理，标准化
sc = StandardScaler()
X_train_std = sc.fit_transform(train_imgs)
X_test_std = sc.fit_transform(test_imgs)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Fisher LDA降维
# 计算类内离散度与类间离散度矩阵
mean_vecs = []
for i in range(1,41):
    mean_vecs.append(np.mean(X_train_std[y_train==i], axis=0))

# 类内离散度
d = X_train_std.shape[1]
S_W = np.zeros((d, d))
for label, mv in zip(range(1,41), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('类内离散度矩阵形状：' + str(S_W.shape))

# 类间离散度
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

# 取前 k个特征向量组成投影矩阵 w
k = 2
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('投影矩阵 W:', w)

# 计算训练集上降维后的结果并可视化
X_train_lda = X_train_std.dot(w)
X_test_lda = X_test_std.dot(w)

# K 近邻预测
KNNClassifer(1, X_train_lda, y_train, X_test_lda, y_test)







