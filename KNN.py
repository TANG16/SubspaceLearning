# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:31:45 2018
KNN 分类器使用
@author: wyk
"""

from sklearn.neighbors import KNeighborsClassifier

# 数据
X = [[0],[1],[2],[3]]
y = [0, 0, 1, 1]

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X, y)
pre = KNN.predict([[2.1]])




