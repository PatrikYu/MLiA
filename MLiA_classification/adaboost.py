#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# 元算法（mean-algorithm）是对其他算法进行组合的一种方式，AdaBoost就是最流行的元算法
# 前面已经介绍了五种不同的分类算法，它们各有优缺点，我们自然可以将不同的分类器组合起来：这种组合结果就是集成方法或者元算法
# Boosting算法要涉及到两个部分，加法模型和前向分步算法。加法模型就是说强分类器由一系列弱分类器线性相加而成
# 前向分步就是说在训练过程中，下一轮迭代产生的分类器是在上一轮的基础上训练得来的 Fm(x)=Fm−1(x)+βm*hm*(x;am)
# 由于采用的损失函数不同，Boosting算法也因此有了不同的类型，AdaBoost就是损失函数为指数损失的Boosting算法

# 基于单层决策树(单节点决策树)构建弱分类器
def loadSimpData():
    datMat=matrix([[1.,2.1],
                   [2.,1.1],
                   [1.3,1.],
                   [1.,1.],
                   [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels
datMat,classLabels=loadSimpData()

# 接下来通过构建多个函数来建立单层决策树
# 第一个函数将用于测试是否有某个值小于我们正在测试的阈值
# 第二个函数会在一个加权数据集中循环，并找到具有最低错误率的单层决策树
# 这三个for循环其实就是为了完成决策树的每个特征维度上对应的最佳阈值以及表示是大于阈值还是小于阈值为正样本的标识符

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):   # 通过阈值（分类点）比较对数据进行分类
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':          # lt 高于，此时将高于阈值作为正样本，低于阈值的标签定为-1（负样本）
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:                         # gt 小于，此时将低于阈值作为正样本，高于阈值的标签定为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0;       # 用于在特征的所有可能性上进行遍历
    bestStump={};        # 这个字典用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst=mat(zeros((m,1)))
    minError=inf         # 初始化为正无穷大，之后用于寻找可能的最小的最小错误率
    for i in range(n):   # 第一层for循环在数据集的所有特征上遍历
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps   # 计算最小值和最大值来了解应该需要多大的步长，确定阈值
        for j in range(-1, int(numSteps) + 1):  # 在当前维度（特征）的所有范围内循环，[-1,0,1...,10]
            for inequal in ['lt', 'gt']:  # go over less than and greater than，lt,gt分别表示大于和小于
                threshVal = (rangeMin + float(j) * stepSize)   # 阈值的选择靠增加步长来寻找，阈值就是分类点
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0     # 分类正确记为0，分类错误记为1
                temp1=type(predictedVals == labelMat)
                weightedError = D.T * errArr              # 仅保留分类错误的样本的权重，计算加权错误率
                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i                  # 基于第i个特征的分类效果最好
                    bestStump['thresh'] = threshVal       # 阈值
                    bestStump['ineq'] = inequal           # 是大于阈值还是小于阈值取为正样本得到的加权错误率小
    return bestStump, minError, bestClasEst    # 返回最好的决策树参数，最小的加权错误率，以及预测值
# D=mat(ones((5,1))/5)              # 初始化每个样本的权重
# print buildStump(datMat,classLabels,D)


# 完整AdaBoost算法的实现

def adaBoostTrainDS(dataArr,classLabels,numIt=40):     # numIt是迭代次数，DS代表单层决策树
    weakClassArr=[] # 每次生成的最佳决策树信息都放入这个列表中，下一个弱分类器会加深上一个弱分类器未能分类正确的样本的权重，
                    # 最后利用所有的弱分类器加权进行分类
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)               # 初始化权重矩阵
    aggClassEst=mat(zeros((m,1)))      # 列向量aggClassEst用于记录每个数据点的类别估计累计值
    for i in range(numIt):          # 循环运行numIt次直到训练错误率为0为止
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)  # 建立一个单层决策树，返回的是利用D得到的具有最小错误率的单层决策树
        #print "D:",D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # max(error, 1e-16)确保error为0时不会发生除0溢出
        # alpha是此时此弱分类器的权重，根据错误率计算得到，始终大于0
        bestStump['alpha'] = alpha      # alpha被添加到字典bestStump中，
        weakClassArr.append(bestStump)  # 字典又添加到列表中
        # print "classEst: ",classEst.T                 # classEst是预测的类别
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 如果正确分类，那么expon=-alpha，错误分类就=alpha
        D = multiply(D, exp(expon))
        D = D / D.sum()  # 为下次循环计算新的D值（D值是样本的权重）
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst   # 列向量aggClassEst用于记录每个样本的类别估计累计值
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        #  # aggClassEst的符号与classLabels一致则说明分类成功，预测正确的样本项为1，预测错误的项为0
        errorRate = aggErrors.sum() / m
        # 训练误差不为0则再次将最新的权重D，建立一个新的决策树！！！！！！！这句很重要
        print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst
# 测试算法：基于AdaBoost的分类
# 每个弱分类器的结果以其对应的alpha值作为权重，所有这些弱分类器的结果加权求和就得到了最后的结果

def adaClassify(datToClass,classifierArr):       # datToClass是一个或者多个待分类样本，classifierArr是训练出的多个弱分类器
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):   # 遍历classifierArr中的所有弱分类器
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst  # 输出的类别预测值乘上该单层决策树的alpha权重然后累加到aggClassEst
        #print aggClassEst
    return sign(aggClassEst)      # 大于0则返回+1，小于0则返回-1
#datArr,labelArr=loadSimpData()
#classifierArr,aggClassEst=adaBoostTrainDS(datArr,labelArr,30)
#print classifierArr    # 这就是最终的强分类器（由许多弱分类器组成）
#print adaClassify([0,0],classifierArr)
# 输出的aggClassEst依次为[[-0.69314718]]，[[-1.66610226]]，[[-2.56198199]]，可见，分类效果越来越强

# 示例：在一个难数据集上应用 AdaBoost （注意确保类别标签是+1和-1）

def loadDataSet(fileName):       # 自适应数据加载函数
    temp1=open(fileName).readline().split('\t')
    numFeat=len(open(fileName).readline().split('\t'))   # 自动检测出特征的数目
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))    # 默认最后一个特征是类别标签
    return dataMat,labelMat
datArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,50)      # 10为迭代次数，一般训练误差达不到0，所以分类器数目一般也为10

testArr,testlabelArr=loadDataSet('horseColicTest2.txt')
prediction10=adaClassify(testArr,classifierArray)
temp2=shape(testlabelArr)[0]       # 67
errArr=mat(ones((67,1)))
print "错误率：",errArr[prediction10 != mat(testlabelArr).T].sum()/67*100,'%'   # 由于有30%的数据缺失，所以效果不好，但也比逻辑回归好


# ROC曲线的绘制及AUC计算函数

def plotROC(predStrengths,classLabels):        # predStrengths是分类器的预测强度
    import matplotlib.pyplot as plt
    cur=(1.0,1.0)           # 此浮点数二元组保留的是绘制光标的位置，右上角(1.0,1.0)对应的是将所有样本判定为正样本的情况
    ySum=0.0                # 用于计算AUC的值
    numPosClas=sum(array(classLabels)==1.0)       # 计算正例的数目（通过数组过滤方式）
    yStep=1/float(numPosClas)                     # y轴是真阳率，因此步长为 1/正例数目
    xStep=1/float(len(classLabels)-numPosClas)    # x轴是假阳率，因此步长为 1/反例数目
    sortedIndicies=predStrengths.argsort()        # 将元素从小到大排列，提取其对应的index(索引)
    print aggClassEst[sortedIndicies]
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    # temp1=sortedIndicies.tolist()       # [[...]]
    # temp2=sortedIndicies.tolist()[0]    # [...]
    for index in sortedIndicies.tolist()[0]:    # tolist将数组或矩阵转换为list
        # index从左至右分别是预测为负样本程度最高的到预测为正样本程度最高的，循环从左至右进行
        # 先从排名最低的样例开始，所有排名更低的例子都被判为反例，而所有排名更高的例子都被判为正例
        # 以此为评判标准，画出图上每个点
        if classLabels[index] == 1.0:           # 每得到一个标签为1.0的类，说明错误地将一个正样本预测为了负样本，
                                                # 则要沿着y轴的方向下降一个步长，即降低真阳率
            delX=0;delY=yStep;
        else:   # 当得到其他标签的类，说明这个负样本预测正确，那么向左移一步，使假阳率降低
            delX=xStep;delY=0;
            ySum += cur[1]     # 为计算AUC，需要计算多个小矩形的面积，宽是xStep，可以先累积高度
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b') # 在当前点cur[0]，cur[1]和新点之间画出一条线段
        cur=(cur[0]-delX,cur[1]-delY)   # 更新当前点cur
    ax.plot([0,1],[0,1],'b--')          # x轴0，y轴0 画折线图到 x轴1，y轴1
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print "the area under the curve is: ",ySum*xStep
plotROC(aggClassEst.T,labelArr)
# 10个弱分类器对应的AUC值为 0.858，50个对应的是0.895，可见，性能提高了








