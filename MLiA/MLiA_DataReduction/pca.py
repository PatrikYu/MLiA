#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块


# PCA算法
def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):   # topNfeat：应用的N个特征
    meanVals = mean(dataMat,axis=0)  # 按列取平均值
    meanRemoved = dataMat-meanVals   # 去平均值
    covMat = cov(meanRemoved,rowvar=0)  # 若rowvar=0，说明传入的数据一行代表一个样本，若非0，说明传入的数据一列代表一个样本
    eigVals,eigVects = linalg.eig(mat(covMat))   # 计算特征值和特征向量
    eigValInd = argsort(eigVals)    # 对特征值进行从小到大的排序
    eigValInd = eigValInd[-1:-(topNfeat+1):-1] # 切片取出倒数第一到倒数第N个，即最大的N个特征值
    redEigVects = eigVects[:,eigValInd]  # 最大的N个特征值对应的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 利用N个特征将原始数据转换到新空间中
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 得到降维之后的数据集
    return lowDDataMat,reconMat

dataMat=loadDataSet('testSet.txt')
lowDMat,reconMat = pca(dataMat,1)    # 转为1维
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()


# 示例：利用PCA对半导体制造数据降维

# 将NaN（缺失数据）换为平均值的函数,用 isnan 函数判断
def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ') # 载入数据并去除空格
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])  # 计算第i个特征值中非NaN的平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal         # NaN项用平均值替换
    return datMat

dataMat = replaceNanWithMean()












