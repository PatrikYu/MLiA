#  coding: utf-8
import sys

reload(sys)
sys.setdefaultencoding('utf8')  # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置作图中显示中文字体

from numpy import *
import operator


# CART算法的实现代码

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 对curLine中每个元素float,将每行的内容保存为一组浮点数
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):  # 通过数组过滤方式将上述数据集切分得到两个子集并返回
    # 以第 feature 个特征的value为分界点切分两个子集
    temp1 = nonzero(dataSet[:, feature] > value)
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1



# 将CART算法用于回归

# 回归树的切分函数
def regLeaf(dataSet):            # 当chooseBestSplit()确定不再对数据进行切分时，将调用此函数来得到叶节点的模型
    return mean(dataSet[:,-1])   # 在回归树中，该模型其实就是目标变量的均值
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]      # 用均方差乘以数据集中样本个数，得到总方差
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]    # 容许的误差下降值
    tolN=ops[1]    # 切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:         # 统计不同剩余特征值的数目，若只有一种，那么不需要切分
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)
    bestS=inf;bestIndex=0;bestValue=0
    for featIndex in range(n-1):                         # 对每个特征循环
        temp2=dataSet[:, featIndex]
        temp3=dataSet[:, featIndex].T.A.tolist()
        temp4=dataSet[:, featIndex].T.A.tolist()[0]
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]): # 对每个特征可能的取值循环,注意set不能对matrix操作，要先转为array
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue  # 开始下次循环
            newS=errType(mat0)+errType(mat1)       # 计算此时总方差
            if newS<bestS:
                bestIndex=featIndex          # 特征编号
                bestValue=splitVal           # 切分特征值
                bestS=newS
    if (S-bestS) < tolS:
        # 如果切分数据集后效果提升不够大，那么就不应进行切分操作而直接创建叶节点
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)     # 按照之前得到的最好切分方法进行切分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        # 如果切分出的数据集很小则退出
        return None,leafType(dataSet)
    return bestIndex,bestValue

 # 树构建函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # leafType给出建立叶节点的函数，ops是一个包含树构建所需其他参数的元组
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # 切分由chooseBestSplit函数完成
    if feat == None: return val  # 满足停止条件时返回某类模型的值
                                 # 如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程
    retTree = {}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

myDat=loadDataSet('ex00.txt')
myMat=mat(myDat)
print createTree(myMat)    # 小数据集，只需切分一次
myDat1=loadDataSet('ex0_regTrees.txt')
myMat1=mat(myDat1)
print createTree(myMat1)   # 中等大小数据集，切分几次,树中包含5个叶节点
# 迄今已完成回归树的构建，但是需要某种措施来检查构建过程是否得当。下面介绍树减枝(tree pruning)技术，它通过对决策树剪枝来达到更好的预测效果


# 一棵树如果节点过多，表明该模型可能对数据进行了“过拟合”，通过降低决策树的复杂度来避免过拟合的过程称为剪枝（pruning）
# 在函数chooseBestSplit()中的提前终止条件，实际上就是所谓预剪枝（prepruning）操作
# 另一种形式的剪枝需要使用测试集和训练集，称为后剪枝（postpruning），本节将分析后剪枝的有效性

# 预剪枝
# 树构建算法对输入参数tolS和tolN非常敏感，如果使用其他值，构建的效果可能会大打折扣
# 停止条件tolS对误差的数量级十分敏感，然而，通过不断修改停止条件来得到合理结果并不是很好的方法

# 后剪枝
# 首先指定参数，使得构建出的树足够复杂，便于剪枝
# 接下来从上而下找到叶节点，用测试集来判断这些叶节点合并是否能降低测试误差

def isTree(obj):
    return (type(obj).__name__=='dict')    # 该子集是否是子树

def getMean(tree):   # 这是个递归函数，它从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算它们的平均值（对树进行塌陷处理）
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    if isTree(tree['left']):tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)         # 确认测试集是否为空
    if (isTree(tree['right']) or isTree(tree['left'])):      # 如果是子树，那么生成对应的测试集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)  # 若为子树，进行剪枝,反复调用
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):  # 若两个分支已经不再是子树（即它们都是叶节点）
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))    # 剪枝前误差
        treeMean = (tree['left']+tree['right'])/2.0    # 对两个分支进行合并并计算剪枝后误差
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean   # 返回合并后的
        else: return tree
    else: return tree   # 依然存在子树，不合并，直接返回

myDat2=loadDataSet('ex2_regTrees.txt')
myMat2=mat(myDat2)
myTree=createTree(myMat2,ops=(0,1))
myDatTest=loadDataSet('ex2test.txt')
myMat2Test=mat(myDatTest)
print prune(myTree,myMat2Test)
# 可以看到，大量的节点已经被剪枝掉了，但没有像预期的那样剪枝为两部分，这说明后剪枝可能不如预剪枝有效，一般同时使用这两种技术


# 模型树（将叶节点设定为分段线性函数)
# 分段线性(piecewise linear)是指模型由多个线性片段组成
def linearSolve(dataSet):
    m,n=shape(dataSet)
    X=mat(ones((m,n)));Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('cannot do inverse,\n try increasing the second value of ops')
    ws=xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):     # 当数据不再需要切分的时候它负责生成叶节点的模型
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):      # 在给定数据集上计算误差
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))

myMat3=mat(loadDataSet('exp2.txt'))
print createTree(myMat3,modelLeaf,modelErr,(1,10))
# 可以看到，该代码以 0.285477 为界创建了两个模型（实际数据在0.3处分段），生成的两个线性模型分别是
# [[3.46877936],[1.18521743]]；[[1.69855694e-03],[1.19647739e+01]] 与真实模型十分接近

# 思考一个问题：模型树、回归树以及第8章里的其他模型，哪一种模型更好呢？一个比较客观的方法是计算相关系数
# 该相关系数可以通过调用numpy库中的命令corrcoef(yHat,y,rowvar=0)来求解
# 示例：树回归和标准回归的比较

# 用树回归进行预测的代码
def regTreeEval(model,inDat):
    return float(model)
def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat       # 在原来数据上增加第0列，X0=1，然后计算并返回预测值
    return X*model
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):return modelEval(tree,inData)   # 要对回归树叶节点进行预测，就调用函数 regTreeEval
    if inData[tree['spInd']] > tree['spVal']: # inData[最好的切分特征的序列]>最好的切分特征的值，即将其归于左子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)   # 到达叶节点，调用函数 regTreeEval
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)   # 以向量形式返回一组预测值
    return yHat
# 利用该数据创建一棵回归树：
trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree=createTree(trainMat,ops=(1,20))
yHat=createForeCast(myTree,testMat[:,0])
print "回归树:",corrcoef(yHat,testMat[:,1],rowvar=0) [0,1]        # 值越接近1越好，0.9640852318222141
# 同样地，再创建一棵模型树：
myTree=createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat=createForeCast(myTree,testMat[:,0],modelTreeEval)
print "模型树:",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]         # 0.9760412191380593
# 看看标准的线性回归效果如何
ws,X,Y=linearSolve(trainMat)
print "线性回归的权重是：",ws            # 得到的ws是个矩阵，ws[1,0]表示第二行第一列
for i in range(shape(testMat)[0]):
    yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
print "标准线性回归:",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]    # 0.9434684235674766














