#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *
import operator


def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1     # 默认文件每行最后一个值是目标值(标签)
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def standRegress(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    xTx=xMat.T*xMat                 # 计算X(T)*X,判断行列式是否为0，若为0，则不可逆
    if linalg.det(xTx) == 0.0:      # linalg是numpy中线性代数的库，det用于计算行列式
        print "this matrix is singular,cannot do inverse"
        return
    ws=xTx.I * (xMat.T*yMat)            # .I 求逆
    # linalg提供了一个函数来解未知矩阵，若使用该函数，那么写作：ws=linalg.solve(xTx,xMat.T*yMatT)
    return ws
xArr,yArr=loadDataSet('ex0.txt')
print xArr[0:2]         # 第一个值总是1.0，即x0，我们假定偏移量就是一个常数，第二个值x1也就是我们图中的横坐标值
ws=standRegress(xArr,yArr);print ws
xMat=mat(xArr);yMat=mat(yArr)
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])   # 绘制了原始的数据，先用xMat操作再转为array
#
# xCopy=xMat.copy()  # 这样操作就不会改变xMat
# xCopy.sort(0)      # 将直线上的数据点升序排列，这样画出来的就是一条直线(此时欠拟合)
# yHat=xCopy*ws
# ax.plot(xCopy[:,1],yHat)
#
# plt.show()

# 通过计算两个序列的相关系数来计算预测值yHat序列和真实值y序列的匹配程度
yHat=xMat*ws
print corrcoef(yHat.T,yMat)
# 左：预测值，右：实际值，需要将yMat转置，以保证两个向量都是行向量
# 对角线上数据为1.0，因为和自己百分百相关



# 线性回归求的是具有最小均方误差的无偏估计，所以有可能出现欠拟合现象，因此有些方法允许在估计中引入一些偏差，从而降低预测的均方误差
# 局部加权线性回归函数

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr);yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))      # 创建对角矩阵
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))         # 注意 diffMat*diffMat.T 应该要开根号才和p142页的公式符合呀
        # 随着距离接近，权重以指数级衰减
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):  # 用于为数据集中每个点调用lwlr()，这有助于求解k的大小
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):   # 对每一个测试点求出附近每个点的权重
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
print yArr[0]
print lwlr(xArr[0],xArr,yArr,1.0),"k=1.0时预测的y"                   # 欠拟合，效果和最小二乘法效果差不多
print lwlr(xArr[0],xArr,yArr,0.01),"k=0.001时预测的y"                # 刚刚好
yHat=lwlrTest(xArr,xArr,yArr,0.003)     # 得到数据集里所有点的估计    # 过拟合
# 下面绘出这些估计值和原始值，看看yHat的拟合效果。所用的绘图函数需要将数据点按序排列，首先对xArr排序：

# srtInd=xMat[:,1].argsort(0)       # 按升序排序，返回下标
# temp1=xMat[srtInd]                                 # [[[1 0.014]]\n\n[[1 0.015]]\n\n[[1 0.034]]...
# xSort=xMat[srtInd][:,0,:]     #将xMat按照升序排列   # [[1 0.014]\n[1 0.015]\n[1 0.034]...
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T.flatten().A[0],s=2,c='red')
# plt.show()


#示例：预测鲍鱼的年龄

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()
abX,abY=loadDataSet('abalone.txt')
# yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
# print rssError(abY[0:99],yHat01.T)
# yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
# print rssError(abY[0:99],yHat1.T)
# yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
# print rssError(abY[0:99],yHat10.T)
# # 可以看到，使用较小的核将得到较低的训练误差。但使用最小的核可能会造成过拟合
# # 看看在新数据上的表现
# yHat01=lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
# print rssError(abY[100:199],yHat01.T)   # 测试误差并非是最小的，核=10的测试误差反而最小
# # 那么，最佳的核大小是10吗？maybe，如果想得到更好的效果，应该用10个不同的样本集做10次测试来比较结果



# 缩减系数来“理解”数据
# 如果数据的特征比样本点还多，则输入数据的矩阵不是满秩矩阵，所以X(T)X 无法求逆，就无法使用线性回归和原来的方法来做预测
# 为了解决这个问题，可使用岭回归(ridge regression)、lasso法(效果很好但计算复杂)、前向逐步回归(效果不错且容易实现)

# 岭回归
def ridgeRegres(xMat,yMat,lam=0.2):  # 由于lambda是python保留的关键字，因此程序中用了lam来代替
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "this matrix is singular,cannot do inverse"
        return
    ws=denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)                # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)                  # axis可以取0或1，取0求样本方差的无偏估计值（除以N-1；对应取1求得的是方差（除以N）
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))    # 在30个不同的lambda下调用ridgeRegres函数，注意这里的lambda应以指数级变化
        wMat[i,:]=ws.T
    return wMat
# 看一下在鲍鱼数据集上的运行结果
ridgeWeights=ridgeTest(abX,abY)
ax.plot(ridgeWeights)
# 横坐标为1-30，代表矩阵中的30行（lambda的值逐渐增大），纵坐标为8个特征值的权重改变线图（是矩阵的每一行上面的值）
a=ridgeWeights
plt.show()


def regularize(xMat):            # 把之前标准化程序提出来，方便使用
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    return xMat

# 前向逐步线性回归！！！找出重要的特征

def stageWise(xArr,yArr,eps=0.01,numIt=100):  # eps表示每次迭代需要调整的步长
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)            # 数据标准化，使其分布满足0均值和单位方差
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))       # 每次循环得到的权重存储在这
    ws=zeros((n,1));wsTest=ws.copy();wsMax=ws.copy()       # 权重初始化为0
    for i in range(numIt):
        print ws.T
        lowestError=inf  # 设置当前最小误差为无穷
        for j in range(n):         # 对所有特征进行循环
            for sign in [-1,1]:    # 分别计算增加或减少该特征对误差的影响
                wsTest=ws.copy()
                wsTest[j] += eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)     # 注意要先转换为array，再计算代价函数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
xArr,yArr=loadDataSet('abalone.txt')
print stageWise(xArr,yArr,0.01,200)
# 可以观察到w1、w6都是0，这表明它们不对目标值造成任何影响，也就是这些特征很可能是不需要的
# 另外，在参数eps设置为0.01的情况下，一段时间后系数就已经饱和了，并在0.04和0.05之间来回震荡，这是因为步长太大的缘故
# 逐步线性回归算法的优点在于它可以帮助人们理解现有的模型并做出改进，当构建了一个模型后，可以运行算法找出重要的特征，并停止对不重要特征的收集
# 如果用于测试，该算法可以用来选择使误差最小的模型
# 当应用缩减方法(如逐步线性回归或岭回归)时，模型增加了偏差（bias），与此同时也减小了模型的方差


# 示例：预测乐高玩具套餐的价格
# 收集数据：使用Google购物的API来抓取价格
# 发送HTTP请求后API将以JSON格式返回所需的产品信息，python提供了JSON解析模块，我们可以从返回的JSON格式里整理出所需数据

# 购物信息的获取函数：
from time import sleep
import json
import urllib2

def searchForSet(retX, retY, setNum, yr, numPce, origPrc):    # 调用Google购物API并保证数据抽取得正确性
    sleep(10)          # 为了防止短时间内有过多的API调用，一开始要休眠10秒
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (\
        myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
lgX=[];lgY=[]
setDataCollect(lgX,lgY)


# 训练算法：建立模型
m,n=shape(lgX)
lgX1=mat(ones((m,n+1)))
lgX1[:,1:5]=mat(lgX)
# 添加常数项特征X0（X0=1）
ws=standRegress(lgX1,lgY)
# 这样得到的结果仅仅对数据拟合得很好，但公式本身却是错误的
# 下面使用岭回归再进行一次实验，并演示如何用缩减法确定最佳回归系数

# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):     # numVal是算法中交叉验证的次数
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30)) # create error mat numVal rows 30 columns
    for i in range(numVal):    # i是实验次数
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        # 对indexList进行混洗（shuffle），从而实现数据集数据点的随机选取
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])             # 分割训练集和测试集
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY) # 保存岭回归中的所有回归系数，ridgeTest 使用30个不同的lambda值创建了30组不同的回归系数
                                        # 得到30行x列的矩阵
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            # 注意要用训练样本对测试样本做归一化
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            # 注意得到预测的数据需要加上mean(trainY)
            errorMat[i,k]=rssError(yEst.T.A,array(testY))  # 计算误差，并存入errorMat中
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)             # 对列求均值
    minMean = float(min(meanErrors))          # 取不同lambda中误差最小的那个
    bestWeights = wMat[nonzero(meanErrors==minMean)]   # 取此时权重
    # 注意岭回归使用了数据标准化，standRegres则没有，为了将上述比较可视化还需将数据还原
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    # 权重还原的公式，除以xMat的偏差即可，注意需要简单证明
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)
crossValidation(lgX,lgY,10)





