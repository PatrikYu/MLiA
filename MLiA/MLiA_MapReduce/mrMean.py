#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块


# 分布式均值方差计算的mrjob实现
# mrjob是用来写能在hadoop运行的python程序的最简便方法。
# 其最突出的特点就是在mrjob的帮助下，无需安装hadoop或部署任何集群，我们可以在本地机器上运行代码（进行测试）。
# 同时，mrjob可以轻松运行于Amazon Elastic MapReduce。
# 为了达到简便实用的目的，一些功能被mrjob省去了。如果追求更多的功能，可以尝试Dumbo，Pydoop等package。

# 首先安装mrjob库，进入 anaconda prompt,输入：conda install -c conda-forge mrjob

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRmean(MRJob):  # 为了使用mrjob类，需要创建一个新的MRjob继承类，在本例中该类的类名为MRmean，mapper和reducer都是该类的方法
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):  # 接收输入数据流，对输入值进行积累
        if False: yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal

    def map_final(self):  # 所有输入到达后开始处理
        mn = self.inSum / self.inCount
        mnSq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnSq])  # 这里所有的mapper都使用“1”作为key，这样所有的中间值都能在同一个reducer里加起来

    def reduce(self, key, packedValues):
        cumVal = 0.0;
        cumSumSq = 0.0;
        cumN = 0.0
        for valArr in packedValues:  # reducer的输入存放在迭代器对象里，为了读取所有输入，需要使用类似for循环的迭代器
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN   # 分别计算均值，方差
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)  # 最后一条输出语句没有key，因为输出的key值已经固定，如果该reducer之后要执行另一个mapper，那么key依然需要赋值

    # steps()方法定义了执行的步骤
    def steps(self):   # 在steps()方法里，需要为mrjob指定mapper和reducer的名称，否则将默认调用mapper和reducer方法
        return ([MRStep(mapper=self.map, mapper_final=self.map_final, \
                         reducer=self.reduce, )])

if __name__ == '__main__':
    MRmean.run()

# python mrMean.py --mapper < inputFile.txt   先运行一下mapper
# python mrMean.py < inputFile.txt            运行整个程序（移除 --mapper 即可）

## 代码2（下面的代码）的运行结果和上面的代码结果相同，都有5errors，跑出来的均值和方差和网上的相同，但是方差值和书本上有所不同

# from mrjob.job import MRJob
# import mrjob
#
# class MRmean(MRJob):
#     def __init__(self, *args, **kwargs):
#         super(MRmean, self).__init__(*args, **kwargs)
#         self.inCount = 0
#         self.inSum = 0
#         self.inSqSum = 0
#     def map(self, key, val):
#         if False:
#             yield
#         inVal = float(val)
#         self.inCount += 1
#         self.inSum += inVal
#         self.inSqSum += inVal*inVal
#     def map_final(self):
#         mn = self.inSum/self.inCount
#         mnSq = self.inSqSum/self.inCount
#         yield(1, [self.inCount, mn, mnSq])
#     def reduce(self, key, packedValues):
#         cumVal = 0.0; cumSumSq = 0.0; cumN = 0.0
#         for valArr in packedValues:
#             nj = float(valArr[0])
#             cumN += nj
#             cumVal += nj*float(valArr[1])
#             cumSumSq += nj*float(valArr[2])
#         mean = cumVal/cumN
#         var = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
#         yield(mean, var)
#     def steps(self):
#         return ([mrjob.step.MRStep(mapper=self.map, reducer=self.reduce,\
#             mapper_final=self.map_final)])
#
# if __name__ == '__main__':
#     MRmean.run()

# 要在Amazon的EMR上运行本程序，参见 p286



# 示例：分布式SVM的Pegasos算法

# SVM的Pegasos算法
