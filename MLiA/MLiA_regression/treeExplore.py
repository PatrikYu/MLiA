#  coding: utf-8
import sys

reload(sys)
sys.setdefaultencoding('utf8')  # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置作图中显示中文字体

from numpy import *
import operator

## 使用Python的Tkinter库创建GUI(Graphical User Interface,图形用户界面)，并对回归模型进行定性的比较

from Tkinter import *
from ScrolledText import ScrolledText

# root = Tk()
# myLabel=Label(root,text="hello,world")
# myLabel.grid()
# root.mainloop()     # 这个命令将启动事件循环，使该窗口在众多事件中可以响应鼠标点击、按键、重绘等动作


# import regTrees
# def reDraw(tolS,tolN):
#     pass
# def drawNewTree():
#     pass
# Label(root,text="Plot Place Holder").grid(row=0,columnspan=3)      # 通过设定columnspan（允许跨列）和rowspan（允许跨行）
# Label(root,text="tolN").grid(row=1,column=0)
# tolNentry=Entry(root)      # 文本输入框 ， Entry部件是一个允许单行文本输入的文本框
# tolNentry.grid(row=1,column=1)
# tolNentry.insert(0,'10')
# Label(root,text="tolS").grid(row=2,column=0)
# tolNentry=Entry(root)
# tolNentry.grid(row=2,column=1)
# tolNentry.insert(0,'1.0')
# Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)
# ## 加了允许跨行值为3以后，从第一行跑到第二行来了？试试改变rowspan的值
# chkBtnVar=IntVar()       # 按钮整数值
# chkBtn=Checkbutton(root,text="Model Tree",variable=chkBtnVar)   # 复选按钮
# chkBtn.grid(row=3,column=0,columnspan=2)   # 试试调节columnspan的值，观察位置会如何变化
# ## 为了读取 Checkbutton 的状态需要创建一个变量，也就是IntVar



# 集成Matplotlib和Tkinter
# Matplotlib的构建程序包含一个前端，也就是面向用户的一些代码，如plot()和scatter()方法
# 事实上，它同时创建了一个后端，用于实现绘图和不同应用之间的接口，通过改变后端可以将图像绘制在PNG、PDF等格式的文件上
# 下面将设置后端为TkAgg(Agg是一个C++的库，可以从图像创建光栅图（即位图、像素图）)

# Matplotlib和Tkinter的代码集成

import regTrees      # 导入树回归的模块
import matplotlib
matplotlib.use('TkAgg')         # 导入Matplotlib文件并设定后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):         # tolS是容许的误差下降值，tolN是切分的最少样本数
    # tolN=200，则不切分树，用一条直线来拟合；tolN=50，5直线拟合；10，仅需要8直线便可拟合，
    # 为构建尽量大的树，将tolN设置为1，tolN设为0.此时构建模型树，过拟合严重
    reDraw.f.clf()              # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():         # 检查复选框是否选中，选中则构建模型树
        if tolN < 2: tolN = 2   # tolN设置的最小值为2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat,regTrees.modelTreeEval)
    else:             # 否则构建回归树
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))   # 中间两个参数采用默认参数regLeaf
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    #   注意加上 .tolist() 将matrix或array转换为list，才能使用scatter函数，否则会报错
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 预测值采用plot()方法绘制
    reDraw.canvas.show()

def getInputs():
    try:         # 如果出错后续代码不会执行，跳转至except函数执行，最后执行finally语句块（此处没有）
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNentry.delete(0, END)      # 清除错误的输入并恢复默认值
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():      # 当点击ReDraw按钮时会调用此函数
    tolN, tolS = getInputs()  # get values from Entry boxes
    reDraw(tolS, tolN)

root=Tk()

## 先用画布来替换绘制占位符，删掉对应标签
reDraw.f=Figure(figsize=(5,4),dpi=100)
reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

## 最后初始化一些与reDraw()关联的全局变量
reDraw.rawDat=mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat=arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)

root.mainloop()