#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# K-均值聚类支持函数
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat               # 返回值是一个包含许多其他列表的列表

def distEclud(vecA,vecB):        # 计算两个向量的欧氏距离
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):         # 构建一个包含k个随机质心（n维）的集合
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])  # 取每个特征的最小值
        rangeJ=float(max(dataSet[:,j])-minJ)         # rangeJ是一个具体值（此特征的取值范围）
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)  # rand(k,1)产生k行1列的0--1之间的随机值，即 minJ+rangeJ*[0,1]
    return centroids
datMat=mat(loadDataSet('testSet.txt'))

# K-均值聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。不断重复此过程，直到数据点的簇分配结果不再改变为止

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):          # distMeas用于计算距离，createCent用于创建初始质心
    m=shape(dataSet)[0]      # 确定数据集中数据点的总数
    clusterAssment = mat(zeros((m,2)))   # 簇分配结果矩阵包含两列：第一列记录簇索引值，第二列存储误差（距离的平方）
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist=inf;minIndex=-1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex: clusterChanged = True    # 只有不再改变以后，clusterChanged才会为False
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]    # 获得第k个簇的所有样本点
            centroids[cent,:]=mean(ptsInClust,axis=0)    # 对各列求均值，得到新的质心k
    return centroids,clusterAssment
myCentroids,clustAssing = kMeans(datMat,4)

import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
# ax.scatter(datMat[:,0].flatten().A[0],datMat[:,1].flatten().A[0],s=20,c='red',marker='o')
ax.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],s=60,c='blue',marker='x')
m=shape(datMat)[0]
print type(clustAssing)
print clustAssing[0,1]
ptsInClust1 = datMat[nonzero(clustAssing[:,0].A==1)[0]]
ptsInClust2 = datMat[nonzero(clustAssing[:,0].A==2)[0]]
ax.scatter(ptsInClust1[:,0].flatten().A[0],ptsInClust1[:,1].flatten().A[0],s=20,c='red',marker='o')
ax.scatter(ptsInClust2[:,0].flatten().A[0],ptsInClust2[:,1].flatten().A[0],s=20,c='red',marker='s')
plt.show()


# 使用后处理来提高聚类性能
# 如何判断簇数目的选择是最合理的呢？在包含簇分配结果的矩阵中保存着每个点的误差，即该点到簇质心的距离平方值。
# K-均值算法收敛但聚类效果较差的原因是，K-均值算法收敛到了局部最小值，而非全局最小值
# 利用SEE(sum of squared error,误差平方和)度量聚类效果，即clusterAssment矩阵的第二列之和
# 因为对误差取了平方，因此更加重视那些远离中心的点。注意，聚类的目标是在保持簇数目不变的情况下提高簇的质量
# 方法1：将具有最大SEE值的簇划分为两个簇，将最大簇包含的点过滤出来并在这些点上运行k=2的k-均值算法
#        为了保证簇总数不变，可以将某两个簇进行合并（2 ways）

# 二分K-均值算法
# 该算法首先将所有点作为一个簇，然后将该簇一份为二，之后选择一个簇继续划分，选择哪个簇取决于对其划分是否可以最大程度降低SEE的值
# 另一种做法是选择SEE最大的簇进行划分，知道簇数目达到用户指定的数目为止


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]          # 计算整个数据集的质心
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error # 遍历数据集中的每个点到质心的误差值
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):      # 遍历每个簇来决定最佳的簇进行划分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]    # 取出这个簇的所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])      # 计算每个簇划分完之后的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  # 计算剩余数据集的误差平方和
                                               # 未进一步划分的数据集，依然留在原来的那个簇中，也计算他们的误差平方和
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #  一旦决定了要划分的簇，接下来就要实际执行划分操作，只需要将要划分的簇中所有点的簇分配结果进行修改即可
        #  k=2时会得到两个编号分别为0和1的结果簇
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 将簇的序号改为簇总数，即最后一个序号值
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 此簇的序号继承此时拆分的序号值
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        temp1=bestNewCents
        temp2=bestNewCents[0,:].tolist()[0]
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 将第i个质心用分割之后的一个质心替换
        centList.append(bestNewCents[1,:].tolist()[0])  # 另外一个新产生的质心直接加到最后一行即可
        temp4=nonzero(clusterAssment[:,0].A == bestCentToSplit)[0]
        temp5=bestClustAss
        temp6=clusterAssment
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
        # 把之前的这个簇的样本点分配结果更新了一下

    return mat(centList), clusterAssment

datMat3=mat(loadDataSet('testSet2.txt'))
centList,myNewAssments=biKmeans(datMat3,3)

# 示例：对地图上的点进行聚类

## Yahoo! PlaceFinder API
# import urllib
# import json
# def geoGrab(stAddress, city):
#     apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
#     params = {}
#     params['flags'] = 'J' # 将返回类型设为JSON，JavaScript Object Notation
#     params['appid'] = 'aaa0VN6k'
#     params['location'] = '%s %s' % (stAddress, city)
#     url_params = urllib.urlencode(params)
#     yahooApi = apiStem + url_params      # print url_params
#     print yahooApi               # 打印输出的url
#     c=urllib.urlopen(yahooApi)   # 使用JSON的Python模块将其解码为一个字典
#     return json.loads(c.read())
#
# from time import sleep          # 避免频繁调用API，防止被封
# def massPlaceFind(fileName):
#     fw = open('places.txt', 'w')
#     for line in open(fileName).readlines():
#         line = line.strip()
#         lineArr = line.split('\t')
#         retDict = geoGrab(lineArr[1], lineArr[2])
#         if retDict['ResultSet']['Error'] == 0:
#             lat = float(retDict['ResultSet']['Results'][0]['latitude'])
#             lng = float(retDict['ResultSet']['Results'][0]['longitude'])
#             print "%s\t%f\t%f" % (lineArr[0], lat, lng)
#             fw.write('%s\t%f\t%f\n' % (line, lat, lng))
#         else: print "error fetching"
#         sleep(1)
#     fw.close()
# geoResults=geoGrab('1 VA Center','Augusta,ME')    # 输入街道地址和城市
# massPlaceFind('portlandClubs.txt')                # 获取俱乐部的名称以及经纬度

# 对地理坐标进行聚类（用球面余弦定理来计算两个经纬度之间距离）
# 球面距离计算及簇绘图函数

def distSLC(vecA,vecB):    # 这里的经纬度用角度作为单位，将角度除以180然后乘以pi转换为弧度
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a+b)*6371.0
import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])   # 获得经纬度
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)      # 绘制一幅图，图0，**axprops使得两个坐标轴兼容。
    # 这个rect用来干嘛的？去掉以后，'NoneType' object has no attribute 'imshow' 'NoneType'对象没有属性'imshow'
    imgP = plt.imread('Portland.png')                   # 调用 imread 函数，基于一幅图像，来创建矩阵
    ax0.imshow(imgP)                                    # 调用imshow ，绘制（基于图像创建）矩阵的图
    ax1=fig.add_axes(rect, label='ax1', frameon=False)  # 绘制图1。 作用：使用两套坐标系统（不做任何偏移或缩放）
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        # 注意！使用索引i % len(scatterMarkers) 来选择标记形状，这样就可以循环地使用这些标记
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

clusterClubs(5)          # ValueError: max() arg is an empty sequence!!!我不懂，之前一直运行正常，突然就有bug了？？？

# warning；此图包括与tight_layout不兼容的轴，因此结果可能不正确
