# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体


# 1.基本plot操作


#创建数据
x = np.linspace(-5, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)

#创建figure窗口
plt.figure(num=3, figsize=(8, 5))
#画曲线1
plt.plot(x, y1)
#画曲线2
plt.plot(x, y2, color='blue', linewidth=5.0, linestyle='--')
#设置坐标轴范围
plt.xlim((-5, 5))
plt.ylim((-2, 2))
#设置坐标轴名称
plt.xlabel('xxxxxxxxxxx')
plt.ylabel('yyyyyyyyyyy')
#设置坐标轴刻度
my_x_ticks = np.arange(-5, 5, 0.5)
my_y_ticks = np.arange(-2, 2, 0.3)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

#显示出所有设置
plt.show()

# 另外一个例子：
x=np.linspace(-np.pi,np.pi,256,endpoint=True)
#              起点   终点  样本点数
C,S=np.cos(x),np.sin(x)
plt.xlim(x.min()*1.1, x.max()*1.1)
# 设置横轴和纵轴的界面
plt.ylim(C.min()*1.1, C.max()*1.1)
plt.plot(x,C,color='red',linewidth=2.5,linestyle='-',label=r'$cos(t)$') #红线标记为cos(t)
plt.plot(x,S,color='blue',linewidth=2.5,linestyle='-',label=r'$sin(t)$')
plt.legend(loc='upper left',frameon=False)    # 标注在左上角
plt.show()

from numpy import *      # 导入科学计算包,这样使用numpy中的函数时不用加前缀numpy.
import matplotlib


# 2.用scatter绘制散点图


import matplotlib.pyplot as plt    # pyplot是matplotlib里最常用的作图模块,将matplotlib缩写为plt
fig=plt.figure()                   # 创建figure窗口
ax1=fig.add_subplot(221)
# 创建 x 为 0 和  之间的 200 个等间距值。创建 y 为带随机干扰的余弦值。然后，创建一个散点图
x = linspace(0,3*pi,200);
y = cos(x) + random.rand(1,200);
ax1.scatter(x,y)
# 使用大小不同的圆圈创建一个散点图。以平方磅为单位指定大小
# x、y 和 sz 中的相应元素确定每个圆圈的位置和大小。要按照相同的面积绘制所有圆圈，请将 sz 指定为数值标量
ax2=fig.add_subplot(222)
x = linspace(0,3*pi,200)
y = cos(x) + random.rand(1,200)
#设置标题
ax2.set_title('Scatter Plot')
#设置X轴标签
plt.xlabel('X')
#设置Y轴标签
plt.ylabel('Y')
#设置图标
plt.legend('x2')
#画散点图
ax2.scatter(x,y,c = 'r',marker = 'o')
plt.show()





