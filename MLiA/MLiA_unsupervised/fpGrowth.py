#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# FP树的类定义
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue       # 包含 nameValue，numOccur 这两个变量
        self.count = numOccur
        self.nodeLink = None        # nodeLink变量用于链接相似的元素项（参考图中的虚线）
        self.parent = parentNode    # 父变量parent用于指向当前节点的父节点
        self.children = {}          # 用于存放节点的子节点

    def inc(self,numOccur):         # 对count变量增加给定值
        self.count += numOccur

    def disp(self,ind=1):           # 将树以文本形式显示
        print ' '*ind,self.name, ' ',self.count
        for child in self.children.values():
            child.disp(ind+1)

rootNode=treeNode('pyramid',9,None)     # 创建树中的一个单节点
rootNode.children['eye']=treeNode('eye',13,None)  # 为其增加一个子节点
rootNode.disp()

# FP树构建函数
def createTree(dataSet,minSup=1):     # minSup:最小支持度
    headerTable={}
    for trans in dataSet:
        for item in trans:
            temp0=headerTable.get(item,0)
            headerTable[item]=headerTable.get(item,0) + dataSet[trans]
    for k in headerTable.keys():        # 移除不满足最小支持度的元素项
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:return None,None   # 如果没有元素项满足要求，则退出
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]  # 拓展头指针，以便保存计数值及指向每种类型第一个元素项的指针
    retTree = treeNode('Null Set',1,None)       # 创建只包含空集合的根节点
    for tranSet,count in dataSet.items():
        localD={}
        for item in tranSet:        # 再一次遍历数据集，将频繁项存入字典localD中 [过滤]
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
                                                           # 按照元素项的出现频率从大到小进行排序
            updateTree(orderedItems,retTree,headerTable,count)  # 让FP树生长，调用updateTree
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):  # 没看懂？？？学习一下python的链表吧
    if items[0] in inTree.children:   # 测试第一个元素项是否作为子节点存在，若存在，就更新该元素项的计数
        inTree.children[items[0]].inc(count)
    else:                             # 若不存在，则创建一个新的treeNode并将其作为一个子节点添加到树中,头指针表也要更新以指向新的节点
        inTree.children[items[0]] = treeNode(items[0],count,inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:                         # 更新头指针需要调用函数 updateHeader
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1:   # 对剩下的元素（去掉了列表的第一个元素 items[1::] ）迭代调用 updateTree 函数
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):      # 此函数确保节点链接指向树中该元素项的每一个实例
    # 从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾，这就是一个链表
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 简单数据集及数据包装器
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpDat=loadSimpDat()
print simpDat
initSet=createInitSet(simpDat)
print initSet
myFPtree,myHeaderTab=createTree(initSet,3)
myFPtree.disp()
# 上面给出的是元素项及其对应的概率计数值，其中每个缩进表示所处的树的深度

# 发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode,prefixPath):
    if leafNode.parent != None:            # 迭代上溯整棵树，并收集所有遇到的元素项的名称
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath) # 访问父节点

def findPrefixPath(basePat,treeNode):    # 用头指针指向该类型的第一个元素项
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        temp=treeNode
        if len(prefixPath) > 1:          # 列表返回后添加到条件模式基字典condPats中，
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink     # ？？？ 查找其他的路径？？？
    return condPats
findPrefixPath('r',myHeaderTab['r'][1])


# 递归查找频繁项集的mineTree函数
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p:p[1])]
    # 先对头指针表中的元素按照其出现的频率进行排序（这里的默认顺序是从小到大）
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)    # 将每一个频繁项添加到频繁项集列表中
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])   # 创建条件基
        myCondTree,myHead = createTree(condPattBases,minSup)    # 从条件模式基来构建条件FP树
        if myHead != None:   # 树中有元素项的话，递归调用mineTree()函数
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)

freqItems = []            # 建立一个空列表来存储所有的频繁项集
mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)


# 示例：在Twitter源中发现一些共现词
# 访问Twitter Python库的代码
# import twitter
# from time import sleep
# import re  # 用于正则表达式的库，后面会使用正则表达式来帮助解析文本
#
# def getLotsOfTweets(searchStr):   # 处理认证然后创建一个空列表
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     #you can get 1500 results 15 pages * 100 per page
#     resultsPages = []
#     for i in range(1,15):
#         print "fetching page %d" % i
#         searchResults = api.GetSearch(searchStr, per_page=100, page=i)
#         resultsPages.append(searchResults)
#         sleep(6)
#     return resultsPages
# lotsOtweets = getLotsOfTweets('RIMM')    # 接下来搜索一支名为RIMM的股票，包含14个子列表，每个子列表有100条推文
# print lotsOtweets[0][4].txt
#
# # 文本解析代码（来自第四章）
# def textParse(bigString):
#     urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
#     # 调用正则表达式模块去除URL
#     listOfTokens = re.split(r'\W*', urlsRemoved)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
# # 为每条推文调用textParse,并构建FP树对其进行挖掘
# def mineTweets(tweetArr, minSup=5):
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text))
#     initSet = createInitSet(parsedList)
#     myFPtree, myHeaderTab = createTree(initSet, minSup)
#     myFreqList = []
#     mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#     return myFreqList
# listOfTerms = mineTweets(lotsOtweets,20)
# print len(listOfTerms)     # 看看有多少集合出现频率在20次以上

# 从新闻网站点击流中挖掘（大数据！）

parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
# 将数据集导入到列表
initSet = createInitSet(parsedDat)    # 对初始集合格式化
myFPtree,myHeaderTab = createTree(initSet,100000)
# 构建FP树，并从中寻找那些至少被10万人浏览过的新闻报道
myFreqList = []  # 创建一个空列表来保存这些频繁项集
mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
print len(myFreqList)
# 看看有多少新闻报道或报道集合曾经被10万或者更多的人浏览过
print myFreqList        # 看看具体有哪些新闻


