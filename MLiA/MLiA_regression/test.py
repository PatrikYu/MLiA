#!/usr/bin/python
# -*- coding: utf-8 -*-
#coding=utf-8

from numpy import *
from Tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')  #设置后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #将TkAgg和matplot连接起来
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()  #清除图形
    reDraw.a = reDraw.f.add_subplot(111)  #添加子图
    if chkBtnVar.get(): #如果复选框选中
        if tolN < 2:
            tolN = 2
        #构建模型树
        myTree = regTree.createTree(reDraw.rawDat, regTree.modelLeaf, regTree.modelErr, (tolS, tolN))
        yHat = regTree.createForeCast(myTree, reDraw.testDat, regTree.modelTreeEval)
    else:
        #构建回归树
        myTree = regTree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTree.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1], s=5)  #真实值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  #预测值
    reDraw.canvas.show()

#得到用户输入并防止程序崩溃
def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNentry.delete(0, END)  #清空输入框
        tolNentry.insert(0, '10') #恢复默认值
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

#按下ReDraw按钮时，调用该函数
def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = Tk()  #先创建一个Tk类型的根部件，然后插入标签

reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root) #画布
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

#用.grid()方法设定行和列的位置，通过columnspan和rowspan值告诉布局管理器是否
#允许一个小部件跨行或跨列
Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)  #文本输入框
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)  #按钮

chkBtnVar = IntVar()  #按钮整数值，为了读取Checkbutton的状态而创建
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)  #复选按钮
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTree.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]), 0.01)

reDraw(1.0, 10)

root.mainloop() #启动事件循环，使该窗口在众多事件中可以响应鼠标点击、按键和重绘等动作