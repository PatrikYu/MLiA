#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# 支持向量机有很多种实现，下文只介绍其中最流行的一种实现，即序列最小化(Sequential Minimal Optimization,SMO)算法
# 再介绍使用核函数(kernel)的方式将SVM扩展到更多数据集上

# 接下来用SMO(序列最小优化)算法对6.2.1节的两个式子进行优化:  1.最小化的目标函数  2.优化过程中必须遵循的约束条件
# Platt的SMO算法是将大优化问题分解为多个小优化问题来求解的。这些小优化问题往往很容易求解，并且
# 对它们进行顺序求解的结果与将它们作为整体求解的结果是完全一致的，但缩减了许多时间
# 原理：每次循环中选择两个alpha进行优化处理，增大其中一个同时减少另一个（因为要满足 alpha*label的和=0的约束条件）
# 一对合适的alpha条件:1.两个alpha必须要在间隔边界之外 2.两个alpha还没有进行过区间化处理或者不在边界上

# 把收藏的几篇文章看完，用心做笔记，自己推导一遍，不知道上课的时候会不会讲到呢？可能考虑买一本李航的统计书

# Platt SMO算法中的外循环确定要优化的最佳alpha对，首先实现不带外循环的简化版
# 先构建辅助函数，用于在某个区间范围内随机选择一个整数，同时采用另一个辅助函数用于在数值太大时对其进行调整

def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])   # 获得整个数据的特征值矩阵
        # type为list，[[3.542485, 1.977398], [3.018896, 2.556416],
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
def selectJrand(i,m):      # i是第一个alpha的下标，m是所有alpha的数目
    j=i
    while(j==i):
        j=int(random.uniform(0,m))    # 只有当另一个alpha的下标j不等于i的时候，才会结束循环
    return j
def clipAlpha(aj,H,L):          # 用于调整大于H或小于L的alpha值
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
dataArr,labelArr=loadDataSet('testSet_SMO.txt')
# print mat(dataArr)

# 简化版SMO算法

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # toler:容错率
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()  # 转置，得到一个列向量
    # mat()将list转换为matrix格式，另外， 若某行以\符号结束，意味着该行语句没有结束并会在下一行延续
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))  # alphas为拉格朗日因子，初始化为0
    iter = 0         # iter存储的是在没有任何alpha改变的情况下遍历数据集的次数
    while (iter < maxIter):   # 只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
        alphaPairsChanged = 0     # 变量alphaPairsChanged用于记录alpha是否已经进行优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b   # 预测的类别
            # W=alpha*label*X，FXi=W*Xi
            temp1=multiply(alphas,labelMat).T
            temp2=dataMatrix[i,:].T
            temp3=dataMatrix*dataMatrix[i,:].T
            # multiply是对应项相乘函数，如果两个数组的shape不同的话，会进行广播处理
            Ei = fXi - float(labelMat[i]) # 计算误差
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):  # alphas[i]为什么要这样分布
                # 无论是正间隔或负间隔都会被测试,当误差超过容错率时，需要对alpha进行优化
                # 一旦alphas等于0或C，那么它们就巳经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了
                j = selectJrand(i,m)  # 随机选择另一个alpha值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();   # 对新值和旧值进行比较
                # python会通过引用的方式传递所有列表，所以必须明确地告知python要为alphaIold和alphaJold分配新的内存
                if (labelMat[i] != labelMat[j]):   # 当y1和y2异号，计算alpha的取值范围
                    L = max(0, alphas[j] - alphas[i])   # L和H用于将alpha[j]调整到0和C之间
                    H = min(C, C + alphas[j] - alphas[i])
                else:  # 异号时候
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue     # 如果L=H，则结束本次循环，alphaj的取值范围为空，跳出循环
                # 下面的推导没仔细看，具体见 https://www.tuicool.com/articles/RRZvYb
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # eta是alpha[j]的最优修改量。eta = 2*K12-K11-K22,也是f(x)的二阶导数
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta    # 利用公式更新alpha[j]
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                # 检查alpha[j]是否只是轻微改变，如果是的话，退出for循环
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 对alphas[i]做同样的改变，但是是相反的方向，然后给这两个alpha设置一个常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1     # 把新值b给原来的旧值b，为了后续的循环
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1   #检査alpha值是否做了更新，如果有更新则将iter为0后继续运行程序
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas
# b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
# print alphas[alphas>0]   # 观察alphas矩阵中大于0的元素的数量
# print shape(alphas[alphas>0])  # 得到支持向量的个数
# for i in range(100):
#     if alphas[i]>0.0 : print dataArr[i],labelArr[i]      # 了解哪些数据点是支持向量,可以在画图时子在这些点上画一个大圆圈

# 在几百个点组成的小规模数据集上，简化版SMO算法可以运行，但在更大的数据集上的运行速度就会变慢。
# 完整版的Platt SMO算法通过一个外循环来选择第一个alpha值，其选择过程会在两种方式之间进行交替：一种是在所有数据集上进行单遍扫描
# 另一种方式则是在非边界alpha（不等于边界0或C的alpha值）中实现单遍扫描（首先要建立这些alpha的值）
# 在选择第一个alpha值后，算法会通过一个内循环来选择第二个alpha值。在优化过程中，通过最大化步长的方式来获得第二个alpha值
# 我们会建立一个全局的缓存用于保存误差值，并从中选择使得步长（Ei-Ej）最大的alpha值

# 完整版Platt SMO的支持函数
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler,kTup):  # 建立一个数据结构（对象）来保存所有的重要值,kTup是核函数的信息
        # 需要构建一个仅包含init方法的optStruct类,该方法可以实现其成员变量的填充
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C            # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) # 第一列：是否有效的标志位，第二列：实际的E值
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def calcEk(oS, k):   # 计算E值并返回
    #fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b   # 计算序列号为k样本的预测值
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b         #核函数形式
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):         # 选择第二个alpha或者说内循环的alpha值
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  # 将Ei设置为有效的 #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]   # 返回非0 E值 的序列
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   # 如果这是第一次循环（只有一个非0的E值），那么就随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS,k):    # 计算误差值并存入缓存当中，在上面子函数中用不到，但集成函数时可以用到
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

# 完整Platt SMO算法中的优化例程（用于寻找决策边界的优化历程）内循环

def innerL(i, oS):          # 参照 smoSimple() 函数 ，ctrl+F 查找
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)   # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T   # “直线”可分
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]               # 核函数
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)                               # 在alpha值改变时，更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j]) # update i by the same amount as j
        updateEk(oS, i)                               # added this for the Ecache ，the update is in the oppostie direction
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j] * \
        #      (oS.alphas[j] - alphaJold) * oS.X[i,:]*oS.X[j,:].T
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
        #      (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# 完整版Platt SMO的外循环代码，外循环用于选择第一个alpha的值

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):  # kTup????
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True;alphaPairsChanged=0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        # 当迭代次数超过最大值，或者遍历整个集合都未对任意alpha对进行修改时，就退出循环
        # 与smoSimple函数不同，这里的一次迭代定义为一次循环过程
        alphaPairsChanged=0
        if entireSet:                     # 遍历所有的值
            for i in range(oS.m):         # 此for循环在数据集上遍历任意可能的alpha
                alphaPairsChanged += innerL(i,oS)   # 调用innerL来选择第二个alpha，并在可能时对其进行优化处理
                                                    # 如果有任意一对alpha值发生改变，那么会返回1
                print "fullSet,iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs=nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 不在边界0或C上的值  .A 用于将矩阵转换为array
            for i in nonBoundIs:           # 此for循环遍历所有的非边界alpha值，也就是不在边界0或C上的值
                alphaPairsChanged += innerL(i,oS)
                print "non-bound,iter: %d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet:entireSet=False   # for循环在非边界和完整遍历之间进行切换，并打印出迭代次数
        # 当某一次遍历发现没有非边界数据样本得到调整时，遍历所有数据样本，以检验是否整个集合都满足KKT条件。
        # 如果整个集合的检验中又有数据样本被进一步进化，则有必要再遍历非边界数据样本。
        elif (alphaPairsChanged == 0):entireSet = True
        print "iteration nunber: %d" % iter
    return oS.b,oS.alphas
#b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)

# 由alpha计算权重
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):  # 对数组进行操作，算出每个特征对应的权重
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
# ws=calcWs(alphas,dataArr,labelArr)                    # ctrl + /
# dataMat=mat(dataArr)
# label=dataMat[0]*mat(ws)+b
# print label,' ：',labelArr[0]


#                       核函数
# 以上的样本中，两个类的数据点分布在一条直线的两边，但是，倘若两类数据点分别分布在一个圆的内部和外部，那么会得到什么样的分类面呢？
# 使用核函数（kernel）可以将数据转换成易于分类器理解的形式，下面介绍 径向基函数（radial basis function）这一最流行的核函数

def kernelTrans(X,A,kTup):     # 元组kTup给出的是核函数的信息             # 这个得好好看书，没看懂
    m,n=shape(X)
    K=mat(zeros((m,1)))
    # 根据键值选择相应核函数
    if kTup[0]=='lin': K=X*A.T           # lin ：线性核函数，此时，内积计算在“所有数据集”和“数据集中一行”这两个输入之间展开
    elif kTup[0]=='rbf':                 # rbf表示径向基核函数
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))    # 在numpy矩阵中，除法运算对矩阵元素展开计算，而不是matlab中计算逆
    else:raise NameError('that kernel is not recognized')            # 如果无法识别，就报错
    return K

# 在测试中使用核函数
# 前面提到的径向基函数有一个用户定义的输入kl(用于确定函数值跌落到0的速度参数)，首先我们需要确定它的大小

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1)) # C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    temp1=nonzero(alphas.A>0)
    svInd=nonzero(alphas.A>0)[0]    # 注意这里要取[0]
    sVs=datMat[svInd]   # 得到仅包含支持向量的矩阵
    labelSV = labelMat[svInd];
    temp2=shape(sVs)
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):  # 测试数据1
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))  # 得到转换后的数据,仅对支持向量进行维度的转换
        m1=shape(kernelEval)
        m2=shape(labelSV)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b # 得到预测值
        # 仅需要支持向量数据就可以进行分类
        temp3=alphas[svInd]
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):  # 测试数据2
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)
testRbf(k1=1.3)


# 第二章使用的knn分类算法很不错，但是需要保存所有的训练样本。而对于支持向量机而言，其需要保留的样本少了很多（即只保留支持向量）
# 将 knn.py 中的img2Vector()函数复制过来，构造 基于SVM的数字识别 代码

def img2vector(filename):            # 将一个32*32的二进制图像矩阵转换为1*1024的向量
    returnVect=zeros((1,1024))
    fr = open(filename)
    for i in range(32):             # 注意：range(32)产生的是0-31
        lineStr=fr.readline()      # 循环读出文件的前32行,readline每次只读一行
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])  # 将每行的头32个字符值存储在Numpy数组中
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr == 9:hwLabels.append(-1)  # 遇到数字9，则输出类别标签-1，否则输出+1.这里我们只做二类分类，除1和9外的数字都被去掉了
        else:hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' % (dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('trainingDigits_SVM')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    datMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV=labelMat[svInd]
    print "there are %d support vector" % shape(sVs)[0]
    m,n=shape(datMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr=loadImages('testDigits_SVM')
    errorCount = 0
    datMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1   # sign函数：当x>0，sign(x)=1;
                                                                           # 当x=0，sign(x)=0;
                                                                           # 当x<0， sign(x)=-1
    print "the test error rate is: %f" % (float(errorCount) / m)
testDigits(kTup=('rbf',20))

# 支持向量机之所以称为‘机’是因为它会产生一个二值决策结果，即他是一种决策‘机’
# 支持向量机的泛化错误率较低，也就是说它具有良好的学习能力，且学到的结果具有很好的推广性
