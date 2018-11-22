#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包,这样使用numpy中的函数时不用加前缀numpy.
# import numpy as np      每次使用时都得加上前缀np.
import operator          # 导入numpy中的运算符模块


# 1.准备：使用python导入数据
def createDataSet():     # 这种合成词为了阅读更清楚可以在分词处大写，其他的参数名字就不用大写了
    group=array([[1.0,1.1],[1.0,1.0],[0,0,],[0,0.1]])  # 这是个4行2列的数组
    labels=['A','A','B','B']
    return group,labels

# 2.实施kNN分类算法
def classify0(inx,dataSet,labels,k):
#dataSet=group
#inx=[0,0]
#k=3
    dataSetSize=dataSet.shape[0]    # shape函数是numpy自带的，shape[0]返回行数，shape[1]返回列数，datasetsize就是训练样本的个数
    # 接下来进行距离计算，其中inx是输入向量，此例子中应该是个二维(两列)的，dataset为训练样本(4行两列),labels为训练样本对应的标签(类别)
    diffMat=tile(inx,(dataSetSize,1))-dataSet  # tile函数用于将inx扩展为dataSetSize*inx原来的行数，1*inx原来的列数
    # diffMat得到的就是他们之间特征值之差，得到的是4行2列的
    sqDiffMat=diffMat**2    # 做平方
    sqDistances=sqDiffMat.sum(axis=1)  # axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances=sqDistances**0.5   # 开根号，此时得到样本点到训练数据的欧氏距离
    sortedDistIndicies=distances.argsort()   # argsort函数返回数组值从小到大的索引值
    classCount={}     # 创建了一个空的字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]      # 提取出k个最小的索引值对应的标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  # 记录出现次数的好方法啊
        # 字典的get方法，查找classCount中是否包含voteIlabel，是则返回该值，不是则返回defValue（0），由于classCount是空的字典，
        # 所以for循环，key=voteIlabel处设置value为0，并不断加1，记录此标签出现的次数
        sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        # sorted是个排序函数，接收3个参数，一个可迭代的对象；一个回调函数（回调函数只能有一个参数（如果有多个参数，请用偏函数），
        # 根据这个函数的返回值进行排序；一个布尔值，默认为False 升序排列，True即此处按降序排列，
        # 那么sortedClassCount[0][0]对应的标签就是出现个数最多的
        # items()返回的是一个 list，iteritems()返回一个迭代器,节约内存
        # operator.itemgetter(1)表示获取对象的第2个域的值，即字典classCount的value值，即各个标签对应的个数
        # operator.itemgetter(1,0)则表示获取对象的第2个域和第1个的值，注意返回的key是一个函数，通过key(对象)的方式才能获取具体的值
    return sortedClassCount[0][0]
    print sortedClassCount[0][0]
# 也可在python console中输入：
# import kNN
# group,labels=kNN.createDataSet()
# kNN.classify0([0,0],group,labels,3)

# 示例：使用k-近邻算法改进约会网站的配对效果

# 3.准备数据：从文本文件中解析数据
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()          # readlines()自动将文件内容分析成一个行的列表
    numberOfLines=len(arrayOLines)      # 得到文件行数
    returnMat=zeros((numberOfLines,3))  # 创建返回的Numpy矩阵，1000行3列，此例中特征数为3
    classLabelVector=[]
    index=0
    for line in arrayOLines:      # 循环，逐行进行操作
        line=line.strip()     # 移除字符串头尾某字符，默认为空格或换行符(回车字符)，这样就得到了一整行数据
        #print line       # 40920	8.326976	0.953952	3
        listFromLine=line.split('\t')   # split通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅做num次分割，分为num+1次
        #print listFromLine   # ['40920', '8.326976', '0.953952', '3']  这样就将整行数据分割成了一个元素列表
        returnMat[index,:]=listFromLine[0:3]   # 选取前3个元素，存储到特征矩阵中的第 index 行中，从 0 行开始
        #print returnMat[index,:]
        #print listFromLine[-1]
        classLabelVector.append(int(listFromLine[-1]))  # 最后一个元素是约会的优先级，存储到这个向量中 [3,2,1,1,1...]
        # 此处给出的是1，2，3 ，分别代表着讨厌，还行，喜欢
        # 若后面给出的是字符串like之类的，可以用if语句给它们定标签，这也属于预处理的一部分
        # if listFromLine[-1]=='didntLike': classLabelVector.append(1)
        index +=1 # 行数+1
    return returnMat,classLabelVector
datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')

# 也在console中输入如下语句
#import kNN
#load kNN
#datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
# 有时点击 运行 恢复程序 ，能看到之前运行过的变量的值

# 4.恢复数据：使用Matplotlib创建散点图

import matplotlib
import matplotlib.pyplot as plt    # pyplot是matplotlib里最常用的作图模块,将matplotlib缩写为plt
fig=plt.figure()                   # 创建figure窗口
ax=fig.add_subplot(111)            # 不能通过空figure绘图。必须使用add_subplot()创建一个或多个subplot，
# 111创建了1*1个图，ax在1中绘制
# ax.scatter(datingDataMat[:,2],datingDataMat[:,1],15.0*array(datingLabels),array(datingLabels))
#            横坐标              纵坐标           点的大小，*15得到的点大一点  颜色，同一类的颜色相同
# 这样可以同时画出所有的点，但是没有办法写标签注释，要写标签注释的话可以用下面的方法
# ax.scatter(x,y,s=20,color='b',marker='o',cmap=None,norm=None,vmin、vmax、alpha、linewidths、verts)


idx_1 = nonzero(array(datingLabels)==1)   # nonzero就和matlab里的find函数差不多，注意datingLabels必须先转为array
# print idx_1
# print datingLabels
# print array(datingLabels)[idx_1]   # 验证是否取出标签为1的项
p1 = plt.scatter(datingDataMat[idx_1,2], datingDataMat[idx_1,1], marker = 'x', color = 'm', label='dislike', s = 15)
idx_2 = nonzero(array(datingLabels)==2)
p2 = plt.scatter(datingDataMat[idx_2,2], datingDataMat[idx_2,1], marker = '+', color = 'c', label='justsoso', s = 30)
idx_3 = nonzero(array(datingLabels)==3)
p3 = plt.scatter(datingDataMat[idx_3,2], datingDataMat[idx_3,1], marker = 'o', color = 'r', label='charming', s = 45)
plt.legend(loc = 'upper left')
ax.set_title('class')
plt.xlabel('fly')
plt.ylabel('time on games')   # 需要改编码才能显示中文,改日再改
plt.legend()
plt.show()

# 5.准备数据：归一化数值，将特征转换为等权重特征
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    # 0:取列的最小值，1：取行的最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    # 用shape读取dataset的维度
    m=dataSet.shape[0]
    # shape[0]为行数，样本个数
    normDataSet=dataSet-tile(minVals,(m,1))
    # 数据集中每个特征减去每个特征对应的最小值
    normDataSet=normDataSet/tile(ranges,(m,1))
    # 除以最大值与最小值的差
    return normDataSet,ranges,minVals
normDataSet,ranges,minVals=autoNorm(datingDataMat)

# 6.测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio=0.10
    # 取总样本的十分之一作为测试样本
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in random.randint(0,m,numTestVecs):
        # 在m个样本随机挑选出numTestVecs个作为测试样本,区间左开右闭（即索引为0到m-1）
        # 得到的测试样本为 normMat[1:(numTestVecs-1),:]
        # 训练样本就为 normMat[numTestVecs:m,:]
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d,the real answer is: %d" \
        % (classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
#datingClassTest()

# 7.使用算法:构建完整可用系统

# 选中多行，ctrl+/ ，可以注释或恢复

# def classifyPerson():
#     resultList=['not at all','in small doses','in large doses']
#     percentTats=float(raw_input("percentage of time spent playing video games?"))
#     ffMiles=float(raw_input("frequent flier miles earned per year?"))
#     iceCream=float(raw_input("liters of ice cream consumed per year?"))
#     datingDataMat,datingLabels=file2matrix('datingTestSet2')
#     normMat,ranges,minVals=autoNorm(datingDataMat)
#     inArr=array([ffMiles,percentTats,iceCream])
#     classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
#     # 将输入的数据做与样本数据相同的归一化处理，传入分类函数中，k=3,返回的是出现次数最多的标签值
#     print "you will probably like this person: ",resultList[classifierResult-1]  # 需要将标签值-1，得到的才是索引值



# 示例：手写识别系统

# 1.将一个32*32的二进制图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr=fr.readline()
        # 循环读出文件的前32行,readline每次只读一行
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 2.测试算法：使用k-近邻算法识别手写数字
from os import listdir
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    # 利用listdir返回指定的文件夹包含的文件或文件夹的名字的列表
    m= len(trainingFileList)      # m为样本数量
    trainingMat=zeros((m,1024))   # trainingMat每一行存储一个样本
    for i in range(m):            # 取出一个样本
        fileNameStr=trainingFileList[i]       # '0_113.txt'
        fileStr=fileNameStr.split('.')[0]
        # split通过指定分隔符对字符串进行切片,分为0_013和txt，
        # 取[0]得到文件名0_013，代表这是数字0的第13个样本
        classNumStr=int(fileStr.split('_')[0])
        # 取得真实含义，0，即它的标签为0
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
        # 将图像矩阵转换为向量，便于计算距离

    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        #print "the classifier came back with: %d,the real answer is: %d " % (classifierResult,classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\n the total number of errors is:%d" % errorCount
    print "\n the total error rate is: %f" % (errorCount/float(mTest))
handwritingClassTest()


# 总结

# 实际使用这个算法时，算法的执行效率并不高，因为算法需要为每个测试向量做2000次距离计算，每个距离计算包括了1024个
# 维度浮点计算，总计要执行900次。此外，我们还需要为测试向量准备2MB的存储空间。
# 是否存在一种算法减少存储空间和计算时间的开销呢？使用k近邻算法的优化版——k决策树吧！





