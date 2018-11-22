#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# 分布式均值和方差计算的mapper
def read_input(file):
    for line in file:
        yield line.rstrip()
        # 这里使用的是海量数据集，如果直接对文件对象调用read()方法，会导致不可预测的内存占用。
        # 好的方法是利用固定长度的缓冲区来不断读取文件内容。通过yield，我们不再需要编写读文件的迭代类，就可以轻松实现文件读取。
# 注意缩进！！！第二次犯这样低级的错误了！！！妈耶，搞了半天执行不出来结果居然是缩进问题，后面的语句压根没执行啊
input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

print "%d\t%f\t%f" % (numInputs,mean(input),mean(sqInput))  # 将均值和平方后的均值发送出去
print >> sys.stderr,"report: still alive"  # 标准错误输出，对主节点做出响应报告，表明本节点工作正常
# 在DOS窗口输入以下命令  python mrMeanMapper.py < inputFile.txt

