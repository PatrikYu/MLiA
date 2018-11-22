#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

# 1.计算给定数据集的香农熵
from numpy import *      # 导入科学计算包,这样使用numpy中的函数时不用加前缀numpy.
import operator          # 导入numpy中的运算符模块
from math import log

def calcShannonEnt(dataSet):
    numEntries=len(dataSet)     # 数据集中样本总数
    labelCounts={}          # 创建一个字典，它的键值是最后一列的数值
    for featVec in dataSet:    # 读入一个个样本
        currentLabel=featVec[-1]  # 送入该样本最后一列的数值(可能性之一)
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel]=0    # 这些用法得熟悉：labelCounts.keys
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典，并且在下一句中将次数加1
        labelCounts[currentLabel]+=1  # 如果已经存在，那直接加1。每个键值都记录了当前类别出现的次数
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries    # 计算概率
        shannonEnt -= prob*log(prob,2)      # 累加所有可能值包含的信息期望值，得到熵
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']   # 浮出水面是否能生存，是否有脚蹼
    return dataSet,labels
myDat,labels=createDataSet()
#print calcShannonEnt(myDat)   # 熵越高，则混合的数据也越多，如果在数据集中添加更多的分类，可以看到熵的值也增大了
# 得到熵之后，我们就可以按照获取最大信息增益的方法划分数据集

# 2.按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    # 三个参数：待划分数据集，划分数据集的特征序列号为axis，返回的特征的值为value
    retDataSet=[]       # 创建新的list对象
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            print featVec[:axis]    # 对list执行切片操作，取出前axis个元素，不包括索引为axis的那项
            reducedFeatVec.extend(featVec[axis+1:])     # 切片操作，取索引为 axis+1 的那项一直到最后一项
# 如果某个特征和我们指定的特征值相等, 除去这个特征然后创建一个子特征,将满足条件且经过切割后的样本都加入到新建的样本中
            # 注意append和extend的区别，p38有写
            print featVec[axis+1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet
print splitDataSet(myDat,0,1)     # 即按照第一列元素是否为1来分，并取出第一列为1且除去了第一列特征值的样本
# [[1,'yes'],[1,'yes'],[0,'no']]
# trees.splitDataSet(myDat,0,0)     # 即按照第一列元素是否为0来分
# python语言不用考虑内存分配问题，在函数中传递的是列表的引用，在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期
# 为了消除这个不良影响，我们需要在函数的开始声明一个新列表对象

#注意：在函数中调用的dataSet有一定的要求：1.数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度
#                                        2.数据的最后一列或每个实例的最后一个元素是当前实例的类别标签
# 3.选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1      # 因为最后一项是类别标签，所以要减1
    baseEntropy=calcShannonEnt(dataSet)    # 计算数据集的原始香农熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):     # 第一个循环遍历所有特征
        featList=[example[i] for example in dataSet]   # 用了个列表生成式，取到了数据集中第i+1列所有的值
        # 首先得到第一个特征值可能的取值，然后把它赋值给一个链表,第一个特征值取值是[1,1,1,0,0]
        uniqueVals=set(featList)  # 从列表中创建集合，得到无序且不重复的唯一元素值的集合 [1,0]或[0,1]
        newEntropy=0.0
        for value in uniqueVals:  # 保存的是我们某个样本的特征值的所有的取值的可能性 0，1
            subDataSet=splitDataSet(dataSet,i,value)   # 得到划分之后的列表
            prob=len(subDataSet)/float(len(dataSet))   # 根据第一个特征可能的取值划分出来的子集的概率
            # 第一个特征值的第一个可能取值0划分满足的子集的概率为0.4（得到两个子集），value=1时概率为0.6
            newEntropy += prob*calcShannonEnt(subDataSet)  # 根据数学期望的定义，算出第一个特征划分对应的香农熵
                                                           # 0.4与0.6分别乘以划分之后的列表的香农熵
        infoGain=baseEntropy-newEntropy    # 信息增益=熵的减少量
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
print chooseBestFeatureToSplit(myDat)   # 返回0，表示按照第一个特征值划分得到的效果好

# 4.递归构建决策树

def majorityCnt(classList):
    classCount={}    # 字典classCount中存储了classList中每个类标签出现的次数
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True) # 降序排列
    return sortedClassCount[0][0] # 返回次数最多的分类名称

# 创建树
def createTree(dataSet,labels):   # 输入：数据集，标签列表['no surfacing','flippers']
    classList=[example[-1] for example in dataSet]   # 包含数据集的所有类标签(dataSet的最后一列)['yes','yes','no','no']
    if classList.count(classList[0]) == len(classList):
        # 若classList第一个元素的个数=classList的总个数，即classList仅有一个类标签，则直接返回该类标签'yes'
        return classList[0]  # 不需要继续划分，直接将此特征值返回
    if len(dataSet[0]) == 1:   # 使用完了所有特征，依然不能把数据集划分为仅包含唯一类别的分组
        return majorityCnt(classList)  # 将出现次数最多的类别作为返回值，比如有两个yes一个no，则返回yes，认为属于鱼类
    bestFeat=chooseBestFeatureToSplit(dataSet)   # 当前返回划分效果最好的特征的索引
    bestFeatLabel=labels[bestFeat]               # 划分效果最好的特征,'no surfacing'
    myTree={bestFeatLabel:{}}                    # 用myTree这个字典存储树的所有信息
    del(labels[bestFeat])                        # 将此时划分的特征从标签列表中删去，便于下一次迭代
    featValues=[example[bestFeat] for example in dataSet]  # 取这个特征所有可能的取值（下一句去除重复的元素）
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        # python中列表是按照引用方式传递的，为了保证每次调用函数createTree()时不改变原始列表的内容,复制类标签将其存储在新列表变量中
        myTree[bestFeatLabel][value]=createTree( splitDataSet(dataSet,bestFeat,value) , subLabels)
        # splitDataSet得到划分之后的列表，注意myTree[bestFeatLabel][value]的用法
        # 第一轮循环，将no存入myTree[flippers][0]，  {'flippers': {0: 'no'
    return myTree
print createTree(myDat,labels)        # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

# 测试算法：使用决策树执行分类
# 依靠训练数据构造了决策树之后，我们可以把它应用于实际数据的分类。在执行数据分类时，需要使用决策树以及用于构造决策树的标签向量。
# 决策树在treePlotter.py中retrieveTree函数中构造
# 然后，程序比较测试数据与决策树上的数值，递归执行该过程直到进入叶子节点，最后将测试数据定义为叶子节点所属的类型

def classify(inputTree,featLabels,testVec):
    #       决策树为['no surfacing'...]    测试数据testVec为[1,1],代表两个特征都为肯定
    firstStr=inputTree.keys()[0]  # 第一个节点
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)   # 使用index方法查找当前列表中第一个匹配firstStr变量的元素，返回第一个符合的索引值
    key=testVec[featIndex]         # 找出'no surfacing'对应的那一项的key，是0还是1呢？
    valueOfFeat=secondDict[key]    # 得到这个key对应的value，可能是个字典（决策节点）（key=1），也可能是个叶子节点（key=0）
    if isinstance(valueOfFeat,dict):
        classLabel=classify(valueOfFeat,featLabels,testVec)    # 若是个决策节点，那么继续递归
    else: classLabel=valueOfFeat   # 若达到叶节点，则返回当前节点的分类标签
    return classLabel
myDat,labels=createDataSet()
import treePlotter
myTree=treePlotter.retrieveTree(0)
print classify(myTree,labels,[1,1])

# 使用算法：决策树的存储
# 构造决策树是很耗时的任务，然而用创建好的决策树解决分类问题，则可以很快完成
# 这就要求使用pickle序列化对象，序列化对象可以在磁盘上保存对象，并在需要的时候读取出来，任何对象都可以序列化操作

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)
# 可以把上面两个函数当做 模块 来使用
storeTree(myTree,'classifierStorage.txt')
# 将myTree 存储为 'classifierStorage.txt'文件
print grabTree('classifierStorage.txt')
# 打开 'classifierStorage.txt'文件

# 示例：使用决策树预测隐形眼镜类型

# 隐形眼镜数据集是非常著名的数据集，它包含很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型（包含硬材质，软材质，不适合戴隐形眼镜）
# 将隐形眼镜数据集作为训练样本，解析tab键分隔的数据行
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
# 每一行数据的各个特征之间用制表符来分隔
# 得到的是以  ，连接的各个特征，各个样本均以list形式表示
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)    # 训练算法
treePlotter.createPlot(lensesTree)  # 画出树图























