# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:59:54 2018
加载人脸数据集
ORL
FERET
Yale
@author: wyk
"""

from loadORL import loadImg
from loadFERET import loadFERET
from loadYale import loadYale

def loadDataSet(dataSet='ORL', splitNum=3):
    if dataSet == 'ORL':
        return loadImg(splitNum)
    elif dataSet == 'FERET':
        return loadFERET(splitNum)
    elif dataSet == 'Yale':
        return loadYale(splitNum)
    else:
        print('No this DataSet!')





