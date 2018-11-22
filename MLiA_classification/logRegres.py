#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# logistic回归梯度上升优化算法
def loadDataSet():          # 便利函数：功能是打开文本文件并逐行读取
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()    # strip只能删除开头或是结尾的字符或是字符串。不能删除中间的字符或是字符串，
                                        # ()里无参数时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  # 将第0维特征X0设为1，X1=lineArr[0]，X2=lineArr[1]
        labelMat.append(int(lineArr[2]))   # 标签存入labelMat
    return dataMat,labelMat    # 返回的是matrix类型

def sigmoid(inX):              # S型函数
    return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn,classLabels):   # dataMatIn是numpy数组（三列）
    dataMatrix=mat(dataMatIn)   # 转换为矩阵
    labelMat=mat(classLabels).transpose()   # 为方便运算，将行向量转置为列向量
    m,n=shape(dataMatrix)   # m为样本个数，n为特征个数
    alpha=0.001  # 学习率，步长
    maxCycles=500
    weights=ones((n,1))     # 权重设为1，为什么要设为1？复习的时候思考一下，
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)   # 执行矩阵运算，h是一个列向量，列向量的元素个数等于样本个数
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
        # https://blog.csdn.net/kaka19880812/article/details/46993917 有推导过程，对于逻辑回归，x*error得到的就是梯度
        # 对于线性回归，差一个0.5的倍数。面试的时候公式推导得仔细掌握,记得复习吴恩达的教程
    return weights
dataArr,labelMat=loadDataSet()
weights=gradAscent(dataArr,labelMat)
print weights   # 得到的是matrix形式的权重


# 画出数据集和logistic回归最佳拟合曲线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)        # 将每个数据点的x,y坐标存为矩阵的形式
    n=shape(dataArr)[0]           # 取总样本个数
    xcord1=[];ycord1=[];xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])   # 将两个特征的值分别存入xcord1和ycord1中
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')   # 's'=square
    ax.scatter(xcord2,ycord2,s=30,c='green')            # 默认的marker为'o'
    x=arange(-3.0,3.0,0.1)    # 横坐标范围，此时得到的x是array格式，所以需要用getA()方法将weights转换为array形式
    y=(-weights[0]-weights[1]*x)/weights[2]
    # 拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2(y) = (-w0*x0-w1*x1)/w2, x0为1,x1为x, x2为y
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()
plotBestFit(weights.getA())    # 通过getA()将matrix形式转换为array
# array相乘是对应项相乘，matrix相乘则必须满足矩阵相乘的条件


# 训练算法：随机梯度上升
# 一次处理所有数据被称为“批处理”，随机梯度上升算法：一次仅用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))#每次只对一个样本进行计算，而不是一个包含所有样本的向量，计算的结果都是数值
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights   # 没有矩阵的转换过程，所有变量的数据类型都是NumPy数组array
weights=stocGradAscent0(array(dataArr),labelMat)
plotBestFit(weights)
# 可以看到，这里的分类器错分了三分之一的样本
# 批量梯度上升的结果是在整个数据集上迭代了500次才得到的，一个判断优化算法优劣的可靠方法是看它是否收敛，也就是说
# 参数是否达到了稳定值，是否还会不断地变化？实际上，随机梯度上升训练得到的权重矩阵波动较为严重

# 改进的随机梯度上升算法，减少波动且迭代速度更快
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):  # 迭代次数，，算法将默认迭代150次
        dataIndex=range(m)
        for i in range(m):   # 每次只用一个样本来更新参数，但这个样本是随机选择的
            alpha=4/(1.0+j+i)+0.01   # alpha每次迭代的时候都会调整，这会缓解数据波动或高频波动，
            # 但不能为0，保证多次迭代之后新数据依然具有一定的影响
            # 当j<<max(i)时，alpha就不是严格下降的？？？没懂
            randIndex=int(random.uniform(0,len(dataIndex)))   # 随机选取样本来更新回归系数，这样能减少周期性的波动
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
weights=stocGradAscent1(array(dataArr),labelMat,20)
plotBestFit(weights)  # 迭代20次达到的效果就很好来


# 示例：从疝气病症预测病马的死亡率
# 为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练阶段，通过改变迭代的次数和步长等参数来得到更好的回归系数

# 准备数据：处理数据中的缺失值
# 1.所有的缺失值必须用一个实数值来替换，因为我们用的NumPy数据类型不允许包含缺失值。这里选择实数0来替换所有缺失值，恰适用于logistic回归
# 如果datamatrix的某特征对应值为0，那么该特征的系数将不做更新。由于sigmoid(0)=0.5，即它对结果的预测不具有任何倾向性
# 2.如果在测试数据集中发现了一条数据的类别标签已经缺失，那么最简单的方法是将该条数据丢弃。这是因为类别标签与特征不同，很难确定采用某个值
# 来替换。采用logistic回归对这种做法是合理的，而采用kNN的方法就不太可行。

# 测试算法：用logistic回归进行分类
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else: return 0.0
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))   # 存储某一个样本
        trainingSet.append(lineArr)    # 以list格式存入所有样本
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount += 1
    errorRate=(float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate
def multiTest():       # 做多次实验，取错误率的平均值
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests))
multiTest()   # 由于有百分之30的数据缺失，错误率达到了35%，如果调整迭代次数和步长，还能进一步降低错误率

