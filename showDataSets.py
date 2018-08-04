from loadFERET import loadFERET
from loadORL import loadImg
from loadYale import loadYale
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# 显示 FERET
def showFERET():
    train_imgs, train_labels, test_imgs, test_labels = loadFERET(7)
    for i in range(len(train_imgs)):    
        img = train_imgs[i].reshape(80,80)
        plt.imshow(img,cmap='gray')
        plt.show()

# 显示 ORL
def showORL():
    train_imgs, train_labels, test_imgs, test_labels = loadImg(10)
    for i in range(len(train_imgs)):
        img = train_imgs[i].reshape(112,92)
        plt.imshow(img,cmap='gray')
        plt.show()

# 显示 Yale
def showYale():
    train_imgs, train_labels, test_imgs, test_labels = loadYale(11)
    for i in range(len(train_imgs)):
        img = train_imgs[i].reshape(100,100)
        plt.imshow(img,cmap='gray')
        plt.show()

## 拼接图片
#def collapsingORL():
#    train_imgs, train_labels, test_imgs, test_labels = loadImg(10)
#    for i in range(10):
#        img = train_imgs[i].reshape(112,92)
#        hmerge = np.hstack()
#        plt.imshow(img,cmap='gray')
#        plt.show()

showORL()


