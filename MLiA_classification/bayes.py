#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

# 我们将文本看成单词向量或词条向量，单词的数据库是经过精心挑选的，然后将每一篇文档转换为词汇表上的向量
from numpy import *
# 1.准备数据：从文本中构建词向量

# 词表到向量的转换函数
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]    # 进行词条切分后的文档集合
    classVec=[0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论，人为标注得到
    return postingList,classVec

def createVocabList(dataSet):
    # 此函数创建一个包含在所有文档中出现的不重复词的列表
    vocabSet=set([])
    for document in dataSet:
        temp=document
        vocabSet=vocabSet | set(document)
        # 求两个集合的并集，不断将文档中出现的新词加入vocabSet中
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):    # 输入参数为词汇表及某个文档
    returnVec=[0]*len(vocabList)   # 创建一个其中元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 如果文档中某个词存在于列表中，则向量的对应位置计为1
        else:print "the word: %s is not in my vocabulary!"
    return returnVec

# 之前我们将每个词的出现与否作为一个特征，这是词集模型(set-of-words model)
# 如果记录下每个词出现的次数，这种方法被称为词袋模型(bag-of-words model),词袋模型可以收集到更多有用的信息
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    # 创建一个其中元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


listOPosts,listClasses=loadDataSet()
myVocabList=createVocabList(listOPosts)    # 构造了一个含有所有词的列表
print setOfWords2Vec(myVocabList,listOPosts[0])

# 2.训练算法：从词向量计算概率

def trainNB0(trainMatrix,trainCategory):
    # trainMatrix：包含所有特征向量的矩阵，trainCategory：[0,1,0,1,0,1]
    numTrainDocs=len(trainMatrix)  # 总样本个数
    numWords=len(trainMatrix[0])   # 每个样本中的元素个数（包含的词汇数）
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算侮辱性样本出现的概率
    p0Num = ones(numWords);   # 初始化0出现的次数为1，下面会解释
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):   # 循环操作每个样本
        if trainCategory[i] == 1:   # 若这个样本是侮辱性的
            p1Num += trainMatrix[i]    # 将此样本的特征向量加到p1Num上，出现过的每个词都在侮辱性词语计数中+1
            p1Denom += sum(trainMatrix[i])  # 统计侮辱性词语的总数（一个词语只要出现在被判定为侮辱性句子中，就认为是一个侮辱性词语）
        else:                                          # 同理：一个词语只要出现在被判定为不是侮辱性句子中，就认为不是辱性词语
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # 对向量中每个元素操作，计算每个词属于侮辱性词汇的概率，得到类别为侮辱性的概率向量
    p0Vect = log(p0Num / p0Denom)  # 计算每个词不属于侮辱性词汇的概率
    return p0Vect, p1Vect, pAbusive

trainMat=[]
for postinDoc in listOPosts:      # 该for循环使用词向量来填充trainMat列表,listOPosts为文档集合（每一行对应一个样本）
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))   # 得到的trainMat是每个样本的词向量组成的列表，里面的元素也是一个个列表
p0V,p1V,pAb=trainNB0(trainMat,listClasses)   # listClasses是侮辱词语的标注，[0,1,0,1,0,1]
print trainMat

# 注意两点:1.利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档（样本）属于某个类别的概率，如果其中一个概率值为0，
#            那么最后的乘积也为0，因此所有词的出现数初始化为1，并将分母初始化为2
#          2.计算多个概率的乘积时，由于大部分因子都非常小，所以程序会下溢出。解决方法是对乘积取自然对数。

# 朴素贝叶斯分类函数

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #         要分类的对象 ... 侮辱性的概率向量 侮辱性样本出现的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    # log(a)+log(b)=log(a*b),p1就是P(W0,W1...Wn|Ci)*P(Ci)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  # 相乘指的是元素之间的相乘
    if p1 > p0:
        return 1       # 如果P1>P2，就认为这是一段侮辱性的话
    else:
        return 0

# 这是一个便利函数(convenience function),该函数封装所有操作，以节省输入代码的时间
def testingNB():
    listOPosts, listClasses = loadDataSet()     # 载入训练样本
    myVocabList = createVocabList(listOPosts)   # 构造了一个含有所有词的列表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))   # 训练完成
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))    # 计算testEntry的特征向量
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)  # 判断其分类
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
testingNB()




# 示例：使用朴素贝叶斯过滤垃圾邮件

# 首先需要将文本文件解析成词条向量，对于一个文本字符串，
# 可以先使用python的string.split()函数
mySent='This book is the best book on python or M.L. I have ever laid eyes upon'
print mySent.split()    # 此时标点符号也被当作了词的一部分
# 可以使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
import re
regEx=re.compile('\\W*')
# compile创建一个pattern，其中分隔符是除单词、数字外的任意字符串
listOfTokens=regEx.split(mySent)
# 上面两句等价于 listOfTokens=re.split(r'\W*',mySent)
listOfTokens=[tok for tok in listOfTokens if len(tok) > 0]
# 去除空字符
listOfTokens=[tok.lower() for tok in listOfTokens if len(tok)>0]
# 将所有字符串转换为小写或大写 .upper()
print listOfTokens

# 现在来看数据集中一封完整的电子邮件的实际处理结果，该数据集放在email文件夹中，该文件夹又包含spam、ham两个子文件夹
emailText=open('email/ham/6.txt').read()
listOfTokens=regEx.split(emailText)   # 下面使用一个简单的文本解析规则来实现单词的切分

# 文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)   # 分隔符是除单词、数字外的任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok)>2]  # 全部转为小写字母

def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):   # 依次打开email中的1-25个txt文件,并将它们解析为词列表
        wordList=textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)   # docList存放所有样本，包括25个垃圾邮件和25个正常邮件
        fullText.extend(wordList)  # 将50个txt得到的分词全部放入这个list中，注意以extend的形式，以单独list的形式投入
        classList.append(1)  # 将垃圾邮件的标签设为1
        wordList=textParse(open('email/ham/%d.txt' % i ).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 正常邮件设置标签0，这样就得到1 0 1 0 ...将docList所有样本都标记好了
    vocabList=createVocabList(docList)
    trainingSet=range(50);testSet=[]
    for i in range(10):     # 随机取10个样本作为测试样本
        randIndex=int(random.uniform(0,len(trainingSet)))
        # uniform:在区间(左闭右开)中取一随机数
        # trainingSet是一个整数列表，其中的值从0到49，len(trainingSet)=50
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])   # 将测试样本的索引从训练样本索引中删去
        # 随机选择训练集将剩余部分作为测试集：留存交叉验证
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:   # 训练样本的索引
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])     # 将类别型号存放到trainClasses中
    poV,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))  # 训练
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),poV,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
spamTest()   # 重复10次，统计错误率


# 示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
# 应用：基于一个人的用词来推测他的年龄和其他信息
# 我们将分析两个不同城市的人们发布的征婚广告信息，来比较这两个城市的人们在广告用词上是否不同。如果确实不同，那么他们各自常用的词是哪些？
# 从人们的用词当中，我们能否对不同城市的人所关心的内容有所了解？

# RSS是站点用来和其他站点之间共享内容的一种简易方式（也叫聚合内容），通常被用于新闻和其他按顺序排列的网站，例如Blog
# universal feed parser是 Python 中最常用的RSS程序库，在Anaconda Prompt下输入：conda install feedparser安装，conda list检查库是否已安装
import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')   # 打开craigslist中的RSS源，发现资源已经消失了
# 可以构建一个类似于spamText()的函数来对测试过程自动化
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}   # 创建一个字典存储词语出现的频率
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)   # 降序排列
    return sortedFreq[:30]     # 返回频率最高的30个词汇

def localWords(feed1,feed0):   # 使用两个RSS源作为参数，RSS源要在函数外导入，这是因为RSS源会随时间而改变
    import feedparser
    docList=[]; classList = []; fullText =[]
    q1=len(feed1['entries'])     # 二者都为0，资源消失了
    q2=len(feed0['entries'])
    minLen = min(len(feed1['entries']),len(feed0['entries']))   # 取其短，使访问的两条RSS源的单词量相同
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])     # 每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) # NY is class 1，纽约市的标签设为1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)            # create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   # 将top 30 words存入这个list中
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])  # 去掉vocabList中的top 30 words
        # 语言中大部分都是冗余和结构上的辅助词，如的地得。这些高频词不能作为特征词，移除top30或其他可以通过比较准确率来适当选择
        # 另一个常用的方法是不仅移除高频词，而且还同时从某个预定的词表（stop word list）中移除结构上的辅助词，的地得之类的
    trainingSet = range(2*minLen); testSet=[]           # create training set，长度为2个RSS长度之和
    for i in range(20):   # 随机选出20个作为测试样本
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet: # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,pSF,pNY=localWords(ny,sf)

# 分析数据：显示地域相关的用词

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 :topSF.append((vocabList[i],p0V[i]))   # 返回概率值大于某个阈值的所有单词
        if p1V[i] > -6.0 :topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    # 用了个lambda匿名函数，依照p0V对字典中的单词进行排序，按概率降序排列
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]

# 可能会输出许多停用词 :(


# 对于分类来说，使用概率有时比使用硬规则更为有效。尽管条件独立性假设并不正确，但朴素贝叶斯仍然是一种有效的分类器








