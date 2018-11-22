#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# Apriori算法中的辅助函数

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createCl(dataSet):
    Cl=[]     # C1簇存储所有不重复的项值
    for transaction in dataSet:      # 对每一条购物记录
        for item in transaction:     # 遍历记录中的每个物品项
            if not [item] in Cl:
                Cl.append([item])    # 注意添加的是只包含该物品项的一个列表，因为后面需要做集合操作
    Cl.sort()   # 默认升序排序       # python不能创建只有一个整数的集合，因此这里必须使用列表
    return map(frozenset,Cl)      # frozenset是不可变集合，无法add、remove
                                  # 这里必须使用frozenset而不是set，因为之后必须将这些集合作为字典键值使用

def scanD(D,Ck,minSupport): # D：数据集，Ck：候选项集列表(即上一轮得到的C1，所有的项集)，minSupport：感兴趣项集的最小支持度
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):               # issubset用于判断tid是否存在于集合can中
                # 如果C1中的集合是记录的一部分，那么增加字典中对应的计数值
                if not ssCnt.has_key(can): ssCnt[can]=1         # has_key用于判断键是否存在于字典中
                else:ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems    # 计算支持度
        if support >= minSupport:
            retList.insert(0,key)    # 在列表的首部插入任意新的集合（越大的项集就在越前面）
        supportData[key] = support
    return retList,supportData     # 返回最频繁项集的支持度 supportData

dataSet=loadDataSet();
print dataSet
C1=createCl(dataSet)      # 构建第一个候选项集集合C1
print C1
D=map(set,dataSet)
L1,suppData0=scanD(D,C1,0.5)
print L1,'\n'    # 该列表中的每个单物品项至少出现50%以上的记录中


# 组织完整的Apriori算法
def aprioriGen(Lk,k):  # 输入为频繁参数项集列表Lk，项集元素个数k，输出为Ck，例如k=2时，生成最大的项集就由两个元素构成
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2];L1.sort()
            L2=list(Lk[j])[:k-2];L2.sort()
            if L1==L2:
                retList.append(Lk[i] | Lk[j])  # 利用集合的并操作
            # 如果这两个集合的前k-1个元素都相等，那么就将这两个集合合并成一个大小为k的集合（最终会产生一个空的集合）
            # 这样做是为了确保遍历列表的次数最少
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createCl(dataSet)   # 构建第一个候选项集集合C1
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)   # 计算支持度
    L=[L1]   # 将L1放入列表L中，L会包含L1、L2、L3...    L[0]就是L1
    k=2
    while (len(L[k-2]) > 0):
        # 只要 第 L k-1 项不为空，即上一项不为空，这样每次得到最后一个Lk（按顺序的话由k+1个元素组成）是[],之后退出循环
        Ck=aprioriGen(L[k-2],k)          # Ck是一个候选集列表
        Lk,supK=scanD(D,Ck,minSupport)   # scanD会遍历Ck，丢掉不满足最小支持度要求的项集
        supportData.update(supK)         # 更新supK（支持度）的值
        L.append(Lk)
        k += 1
    return L,supportData
    # L是[[frozenset([1]),frozenset([3])],frozenset([1,3]),frozenset([1,2,3])]等等这样的一个形式
    # 变量supportData 是一个字典，包含我们项集的支持度值

# L,suppData=apriori(dataSet,minSupport=0.7)
# print '\n',L


# 关联规则生成函数
def generateRules(L,supportData,minConf=0.7):    # 输入参数:频繁项集列表，包含那些频繁项集支持数据的字典，最小可信度阈值
    bigRuleList = []            # 一个包含可信度的规则列表
    for i in range(1,len(L)):   # 遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
        for freqSet in L[i]:    # 如果从{0,1,2}开始，那么H1为[{0},{1},{2}]
            H1 = [frozenset([item]) for item in freqSet]
            if(i>1):      # 若元素数目超过2，做进一步的合并
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:         # 只有两个元素，计算其可信度值
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

# 计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    prunedH=[]           # 满足最小可信度要求的规则列表
    for conseq in H:     # 遍历H中的所有项集并计算它们的可信度值
        temp1=supportData[freqSet]
        temp2=supportData[freqSet-conseq]
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)   # 添加满足最小可信度要求的项集
    return prunedH

# 为最初的项集中生成更多的关联规则（没看懂？？？）
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7): # freqSet：频繁项集，H：可以出现在规则右部的元素列表H
    # 分级法：规则右部包含1个元素并对这些规矩进行测试，然后合并所有剩余规则来创建新的规则列表，其中规则右部包含两个元素...
    # 初始得到的H [{0},{1},{2}]
    m =len(H[0])
    if (len(freqSet)) > (m+1):
        Hmp1 = aprioriGen(H,m+1)   # 生成H中元素的无重复组合，Hmp1包含所有的规则
        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf) # 测试Hmp1的可信度以确定规则是否满足要求
        if (len(Hmp1) > 1):   # 若不止一条规则满足要求，那么使用Hmp1迭代调用函数rulesFromConseq()来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

L,suppData=apriori(dataSet,minSupport=0.5)
rules=generateRules(L,suppData,minConf=0.7)
print rules,'\n'
rules=generateRules(L,suppData,minConf=0.5)  # 降低可信度阈值后就可以获得更多的规则
print rules


# 示例：发现国会投票中的模式，具体见P213

# from time import sleep
# from votesmart import votesmart  # 需要先安装votesmart模块
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'，需要自己申请一个key
#
# # 收集美国国会议案中 action ID 的函数
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt')
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print 'bill: %d has actionId: %d' % (billNum, actionId)
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print "problem getting bill %d" % billNum
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList     # 返回 actionId,标题
#
# # 基于投票数据的事务列表填充函数
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)    # 在议题标题后添加Nay(反对)或者Yea(同意)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print 'getting votes for actionId: %d' % actionId
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:  # 遍历所有的投票信息
#                 if not transDict.has_key(vote.candidateName):
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print "problem getting actionId: %d" % actionId
#         voteCount += 2
#     return transDict, itemMeaning

# 示例;发现毒蘑菇的相似特征
# 寻找毒蘑菇中的一些公共特征，利用这些特征就能避免吃到那些有毒的蘑菇
mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L,suppData = apriori(mushDatSet,minSupport=0.3)
# 在结果中搜索包含有毒特征值2的频繁项集
for item in L[1]:
    if item.intersection('2'):print item,'单个特征'
# 也可以对更大的项集来重复上述过程
for item in L[3]:
    if item.intersection('2'):print item,'同时出现3个特征'