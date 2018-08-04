#### 人脸数据集上的降维实验
* python降维算法实现
* 降维算法：PCA，LDA，MMC，MFA，MDP
* 人脸数据集：ORL，Yale，MIT，FERET，AT&T

#### 项目模块
* 加载数据集：loadDataSet(dataSet='ORL', splitNum=3)，加载 ORL人脸数据集，每一类人脸中随机选择3个作为训练样本，其余的作为测试集样本；输出的结果为X_train，y_train， X_test，y_test，其中X_train的形状为(n,d)，n为训练集样本数目，d为图像展开成列向量后的维数。
* 显示数据集：showDataSets.py为显示数据集中人脸图像的文件，使用loadDataSet()函数加载数据，输出数据集中的所有图像。
* 降维算法：PCA，LDA，MMC，MFA，MDP，对应 LDA.py等文件，文件包含降维算法的实现以及在数据集上的测试，输出准确率结果以csv文件保存。
* 绘制各个算法对应的准确率曲线：plotAcc.py文件，将不同的降维算法对应的准确率画在一个曲线图中方便对比。

