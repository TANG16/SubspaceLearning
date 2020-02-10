# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:30:24 2018
加载 Yale人脸库
@author: wyk
"""

import matplotlib.image as mpimg
import numpy as np

# 加载图片,分割训练集与测试集
def loadYale(split_num):
    img_path = 'F:/论文阅读/2018-7月论文/python降维实验/常用数据集/Yale人脸库/yalefaces/'
    classNum = 15
    imgNum = 11
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    for i in range(1,classNum+1):
        for j in range(1, imgNum+1):
            path = img_path + ('%02d' %i) + '/s' + str(j) + '.bmp'
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





