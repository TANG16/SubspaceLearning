# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:39:57 2018
PCA 人脸识别实验
@author: wyk
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

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

# 加载数据
split_num = 6
train_imgs, train_labels, test_imgs, test_labels = loadImg(split_num=split_num)

# 测试函数
#img = train_imgs[2].reshape(112,92)
#plt.imshow(img, cmap='gray')
#print(train_labels[2])
#plt.show()

# 数据预处理
sc = StandardScaler()
X_train_std = sc.fit_transform(train_imgs)
X_test_std = sc.fit_transform(test_imgs)

# PCA 降维
def demensionReduction(n_com, X_train_std, X_test_std):
#    PCA降维
    pca = PCA(n_components=n_com)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    return X_train_pca, X_test_pca

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
            
#    print('预测正确数目：%d' %(acc))
    accuracy = acc/((10-split_num)*40)
#    print('准确率：%.3f' %(accuracy))
    return accuracy

accs = []
for i in range(2,50):
    X_train_pca, X_test_pca = demensionReduction(i, X_train_std, X_test_std)
    accuracy = KNNClassifer(1, X_train_pca, train_labels, X_test_pca, test_labels)
    accs.append(accuracy)

# 画准确率曲线
plt.plot(accs)

