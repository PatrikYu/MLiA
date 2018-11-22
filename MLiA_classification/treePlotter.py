#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      #python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 使坐标轴能显示中文

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']           # 使plotNode能显示中文，注意'中文'前要加U

# 在python中使用Matplotlib注释绘制树形图
# matplotlib提供了一个非常有用的注释工具annotations,它可以在数据图形上添加文本注解

# 1.使用文本注解绘制树节点
import matplotlib.pyplot as plt
# 定义作图属性
# 用字典来定义决策树决策结果的属性，下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
decisionNode = dict(boxstyle="sawtooth", fc="0.8")   # 决策节点
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
leafNode = dict(boxstyle="round4", fc="0.8")         # 叶节点
arrow_args = dict(arrowstyle="<-")            # 箭头形状

def plotNode(nodeTxt, centerPt, parentPt, nodeType):      # 子函数：画线并标注
    # nodeTxt为要显示的文本，centerPt为文本的中心点，即箭头所在的点，parentPt为指向文本的点，nodeType为文本类型
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    # 第一个参数是注释的内容，xy设置箭头尖的坐标，xytext设置注释内容显示的起始位置，arrowprops 用来设置箭头
    # axes fraction：轴分数，annotate是关于一个数据点的文本

def createPlot():
    fig = plt.figure(1,facecolor='white') # 定义一个画布，背景为白色
    fig.clf() # 把画布清空
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
    createPlot.ax1 = plt.subplot(111,frameon=False)
    # 111表示figure中的图有1行1列，即1个，最后的1代表第一个图，frameon表示是否绘制坐标轴矩形
    plotNode(U'决策节点 ' ,(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(U'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
createPlot()

# 精细作图
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# 我们必须知道有多少个叶节点，用于确定x轴的长度，还得知道树的层数，用于确定y轴的高度
def getNumLeafs(myTree):   # 获取叶节点数目
    numLeafs=0
    firstStr=myTree.keys()[0]    # 第一个键即第一个节点，'no surfacing'
    secondDict=myTree[firstStr]  # 这个键key的值value，即该节点的所有子树
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':   # 如果secondDict[key]是个字典，即该节点下面还有子树，说明这是个决策节点
            numLeafs += getNumLeafs(secondDict[key]) # 递归，看看这个决策节点下有几个叶节点
        else:  numLeafs += 1   # 是叶节点，自加1
    return numLeafs

def getTreeDepth(myTree):   #   确定树的层数，即决策节点的个数+1
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 为了节省时间，函数 retrieveTree输出预先存储的树信息
def retrieveTree(i):
    listOfTrees=[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                 {'no surfacing': {0: 'no', 1: {'flippers': {0:{ 'head':{0:'no', 1: 'yes'}},1:'no'}}}}]
    return listOfTrees[i]
print getNumLeafs(retrieveTree(1))

def createPlot(inTree):    # 这是主函数，首先阅读它
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])# 定义横纵坐标轴
    #createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # 绘制图像,无边框,无坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False)             # 无边框，有坐标轴
    # 注意图形的大小是0-1 ，0-1，例如绘制3个叶子结点，最佳坐标应为1/3,2/3,3/3
    plotTree.totalW = float(getNumLeafs(inTree))   #全局变量totalW(树的宽度)=叶子数
    # 树的宽度用于计算放置决策(判断)节点的位置，原则是将它放在所有叶节点的中间
    plotTree.totalD = float(getTreeDepth(inTree))  #全局变量 树的高度 = 深度
    # 同时我们用两个全局变量plotTree.xoff和plotTree.yoff追踪已经绘制的节点位置，以及放置下一个节点的合适位置
    plotTree.xOff = -0.5/plotTree.totalW;  # 向左移半格
    #但这样会使整个图形偏右因此初始的，将x值向左移一点。
    plotTree.yOff = 1.0;  # 最高点，(0.5,1.0)为第一个点的位置
    plotTree(inTree, (0.5,1.0), '')   # 调用plotTree子函数，并将初始树和起点坐标传入
    plt.show()

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 当前树的叶子数
    depth = getTreeDepth(myTree)    # 深度，函数中没用到
    firstStr = myTree.keys()[0]     # 第一个节点
    # cntrPt文本中心点   parentPt 指向文本中心的点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  # 定位到中间位置，不太清楚
    plotMidText(cntrPt, parentPt, nodeTxt) # 画分支上的键：在父子节点之间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)   # 画决策节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD # 从上往下画
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':# 如果是字典则是一个决策（判断）结点
            plotTree(secondDict[key],cntrPt,str(key))  # 继续递归
        else:   # 打印叶子结点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) # 画叶节点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) # 在父子节点之间填充文本信息
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD  # 重新确定下一个节点的纵坐标

def plotMidText(cntrPt,parentPt,txtString):           # 在父子节点之间填充文本信息
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]    # 得到中间位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotation=30)

createPlot(retrieveTree(0))
myTree=retrieveTree(0)    # 得先赋值再改
myTree['no surfacing'][3]='maybe'
createPlot(myTree)







