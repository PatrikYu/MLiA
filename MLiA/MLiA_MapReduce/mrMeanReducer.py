#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# 分布式计算均值和方差的reducer
# reducer 接收mapper的输出，并将它们合并为全局的均值和方差
def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)  # 读取map端的输出，共有三个字段，按照'\t'分隔开来
mapperOut = [line.split('\t') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])  # 第一个字段是数据个数
    cumN += nj
    cumVal += nj * float(instance[1])  # 第二个字段是一个map输出的均值，均值乘以数据个数就是数据总和
    cumSumSq += nj * float(instance[2])  # 第三个字段是一个map输出的平方和的均值，乘以元素个数就是所有元素的平方和
mean = cumVal / cumN  # 得到所有元素的均值
var = (cumSumSq / cumN - mean * mean)  # 得到所有元素的方差
print "%d\t%f\t%f" % (cumN, mean, var)
print >>sys.stderr,"report: still alive"
# 在DOS窗口输入以下命令  python mrMeanMapper.py < inputFile.txt | python mrMeanReducer.py