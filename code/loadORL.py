# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:04:29 2018
加载 ORL数据集
@author: wyk
"""
import matplotlib.image as mpimg
import numpy as np

# 加载图片,分割训练集与测试集
def loadImg(split_num):
    img_path = 'F:/论文阅读/2018-7月论文/python降维实验/常用数据集/ORL人脸库/ORL92112/bmp/s'
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