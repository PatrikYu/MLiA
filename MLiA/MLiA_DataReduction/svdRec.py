#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# 相似度计算
from numpy import linalg as la

def euclidSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):
    if len(inA) < 3:return 1.0   # 如果点的数量在3个以下，返回1.0，因为此时两个向量完全相关？
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]
    # corrcoef输出结果是一个相关系数矩阵, 参数rowvar=0表示对列求相似度，[i][j]表示第i个随机变量与第j个随机变量的相关系数

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)   # 把取值范围归一化到0和1之间


# 示例：餐馆菜肴推荐引擎

# 基于物品相似度的推荐引擎
def standEst(dataMat,user,simMeas,item):  # 参数：数据矩阵（行用户，列物品），用户编号，相似度计算方法，物品编号
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j] # 该用户对某一物品的评分
        if userRating == 0:continue  # 某物品分值为0，意味着用户没有对该物品评分，跳过这个物品！这个物品不对我们要预测的item产生影响
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        # 寻找两个都被同一用户评分的物品，其中一个物品是指定的item
        if len(overLap) == 0:similarity = 0
        else: similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j]) # 计算这两个物品间的相似度
        simTotal += similarity  # 对相似度进行累加
        ratSimTotal += similarity * userRating # 还考虑相似度和当前用户评分的乘积
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal   # 得到预测的评分
    # 通过除以所有的评分总和，归一化，使得最后的评分值在0到5之间

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):   # estMethod:估计方法
    unratedItems = nonzero(dataMat[user,:].A==0)[1]  # 寻找未评级的物品
    if len(unratedItems) == 0:return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)  # 预测该物品的评分
        itemScores.append((item,estimatedScore))   # 该物品的编号和估计得分值会放在一个元素列表中
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]  # 按照估计得分对该列表进行排序（从大到小逆序排序）并返回

# 利用SVD提高推荐的效果
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
U,Sigma,VT = la.svd(mat(loadExData2()))
print Sigma
Sig2=Sigma**2
S1 = sum(Sig2)*0.9  # 计算总能量的90%
S2 = sum(Sig2[:3])  # 计算前3个元素所包含的能量，发现该值高于总能量的90%，于是我们利用SVD将所有的菜肴映射到一个三维空间中去

# 基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4=mat(eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # 利用U矩阵将物品转换到低维空间中
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating ==0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
        if simTotal == 0:return 0
        else: return ratSimTotal/simTotal     # 返回这个值用于估计评分的计算


# 示例：基于SVD的图像压缩
def printMat(inMat,thresh=0.8): # 此函数用于打印矩阵，由于矩阵包含了浮点数，因此必须定义浅色和深色，这里通过阈值thresh来界定
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else: print 0,
        print ''

def imgCompress(numSV=3,thresh=0.8):  # 此函数实现了图像的压缩，它允许基于任意给定的奇异值数目来重构图像
    my1 = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))  # 从文本文件中以数值方式读入字符
        my1.append(newRow)
    myMat = mat(my1)
    print "**** original matrix ****"
    printMat(myMat,thresh)
    U,Sigma,VT = la.svd(myMat)    # 对原始图像进行SVD分解
    SigRecon = mat(zeros((numSV,numSV)))  # 建立一个全0矩阵并将前面那些奇异值填充到对角线上
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]  # 得到重构后的矩阵
    print "**** reconstructed matrix using %d singular values ****" % numSV
    printMat(reconMat,thresh)

imgCompress(2)
# 可以看到，只需要两个奇异值就能相当精确地对图像实现重构。U和V都是32*2的矩阵，有两个奇异值。因此总数目是64+64+2=130，
# 和原数目相比，我们获得了几乎10倍的压缩比























