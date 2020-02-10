# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:01:20 2018
numpy 练习
@author: wyk
"""
import numpy as np
import heapq
from collections import Counter

## zip
#a = [1,2,3]
#b = [4,5,6]
#c = [7,8,9]
#
#for i, j, k in zip(a,b,c):
#    print('%s %s %s' %(i, j, k))
##    print('下一次循环')
#
## numpy unique()
#a = np.random.randint(0,5,8)
#print(a)
#print(np.unique(a))
#
## numpy argsort升序排序输出下标
#a = [2,1,4,3,5]
#c = ['b', 'a', 'd', 'c', 'e']
#b = np.argsort(a)
#print(b)
#w = [c[i] for i in b]
#print(w)

# Q,R分解
a = np.matrix([[1,2,3],
               [4,5,6]])

print(a)
print('reduced(default)'.center(30, '='))
q, r = np.linalg.qr(a)
print(q, r, np.dot(q,r), sep='\n\n')













