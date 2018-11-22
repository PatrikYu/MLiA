#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

from numpy import *      # 导入科学计算包
import operator          # 导入numpy中的运算符模块

# mrjob中分布式Pegasos算法的外围代码

from mrjob.job import MRJob

import pickle



class MRsvm(MRJob):  # 创建一个mrjob类
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def __init__(self, *args, **kwargs):
        super(MRsvm, self).__init__(*args, **kwargs)
        self.data = pickle.load(open('C:\Users\Peter\machinelearninginaction\Ch15\svmDat27'))
        self.w = 0
        self.eta = 0.69
        self.dataList = []
        self.k = self.options.batchsize
        self.numMappers = 1
        self.t = 1  # iteration number

    def configure_options(self):
        super(MRsvm, self).configure_options()
        self.add_passthrough_option(
            '--iterations', dest='iterations', default=2, type='int',
            help='T: number of iterations to run')
        self.add_passthrough_option(
            '--batchsize', dest='batchsize', default=100, type='int',
            help='k: number of data points in a batch')

# 分布式Pegasos算法的mapper和reducer代码

    def map(self, mapperId, inVals):  # needs exactly 2 arguments
        # input: nodeId, ('w', w-vector) OR nodeId, ('x', int)
        if False: yield
        if inVals[0] == 'w':  # accumulate W-vector
            self.w = inVals[1]
        elif inVals[0] == 'x':
            self.dataList.append(inVals[1])  # accumulate data points to calc
        elif inVals[0] == 't':
            self.t = inVals[1]
        else:
            self.eta = inVals  # this is for debug, eta not used in map

    def map_fin(self):
        labels = self.data[:, -1];
        X = self.data[:, 0:-1]  # reshape data into X and Y
        if self.w == 0: self.w = [0.001] * shape(X)[1]  # init w on first iteration
        for index in self.dataList:
            p = mat(self.w) * X[index, :].T  # calc p=w*dataSet[key].T
            if labels[index] * p < 1.0:
                yield (1, ['u', index])  # make sure everything has the same key
        yield (1, ['w', self.w])  # so it ends up at the same reducer
        yield (1, ['t', self.t])

    def reduce(self, _, packedVals):
        for valArr in packedVals:  # get values from streamed inputs
            if valArr[0] == 'u':
                self.dataList.append(valArr[1])
            elif valArr[0] == 'w':
                self.w = valArr[1]
            elif valArr[0] == 't':
                self.t = valArr[1]
        labels = self.data[:, -1];
        X = self.data[:, 0:-1]
        wMat = mat(self.w);
        wDelta = mat(zeros(len(self.w)))
        for index in self.dataList:
            wDelta += float(labels[index]) * X[index, :]  # wDelta += label*dataSet
        eta = 1.0 / (2.0 * self.t)  # calc new: eta
        # calc new: w = (1.0 - 1/t)*w + (eta/k)*wDelta
        wMat = (1.0 - 1.0 / self.t) * wMat + (eta / self.k) * wDelta
        for mapperNum in range(1, self.numMappers + 1):
            yield (mapperNum, ['w', wMat.tolist()[0]])  # emit w
            if self.t < self.options.iterations:
                yield (mapperNum, ['t', self.t + 1])  # increment T
                for j in range(self.k / self.numMappers):  # emit random ints for mappers iid
                    yield (mapperNum, ['x', random.randint(shape(self.data)[0])])


    def steps(self):
        return ([self.mr(mapper=self.map, reducer=self.reduce,
                         mapper_final=self.map_fin)] * self.options.iterations)


if __name__ == '__main__':
    MRsvm.run()
