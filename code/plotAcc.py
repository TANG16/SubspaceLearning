# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:20:07 2018
绘制准确率曲线
@author: wyk
"""

import matplotlib.pyplot as plt
import pandas as pd

L = 5

def showAcc(L):
    df_pca = pd.read_csv('./PCA_' + str(L) + '.csv')
    df_lda = pd.read_csv('./LDA_' + str(L) + '.csv')
    df_lda_sk = pd.read_csv('./LDA_sk_' + str(L) + '.csv')
    df_mmc = pd.read_csv('./MMC_' + str(L) + '.csv')
    df_mfa = pd.read_csv('./MFA_' + str(L) + '.csv')
    df_mdp = pd.read_csv('./MDP_' + str(L) + '.csv')
    df_mdp_kk = pd.read_csv('./MDP_kk_' + str(L) + '.csv')
    
    plt.plot(df_pca['PCA'])
#    plt.plot(df_lda['LDA'])
    plt.plot(df_lda_sk['LDA_sk'])
    plt.plot(df_mmc['MMC'])
    plt.plot(df_mfa['MFA'])
#    plt.plot(df_mdp['MDP'])
    plt.plot(df_mdp_kk['MDP_kk'])
    
    plt.legend()
    plt.xlabel("dimension")  
    plt.ylabel("accuracy")
    plt.ylim(0.6,0.95)
#    plt.xlim(3,69)
    plt.title('L=' + str(L) + ", accuracy")  
    plt.legend(loc='best')
    plt.show()
    
for i in range(3,6):
    showAcc(i)




