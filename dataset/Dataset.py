# coding=utf8

import numpy as np
#from ..preprocess.utils import readJson
import sys
import os
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024')) 
from preprocess.utils import readJson

class SentiRecDataset:
    def __init__(self, filename, fields=None, trainTestFrac=0.7):
        '''
        Parameters
        ----------
        filename : str
            预处理后的文件路径
        fields : list of str
            使用的字段列表
        '''
        self.data = np.array([i for i in readJson(filename)])
        print('len(data):',len(self.data))
        self.fields = fields
        self.trainTestFrac = trainTestFrac
        self.index = np.arange(len(self.data))
        np.random.shuffle(self.index)
        splitPoint = int(len(self.data)*self.trainTestFrac)
        self.trainIndex = self.index[:splitPoint]
        self.testIndex = self.index[splitPoint:]

    def _getBatch(self, index, batchSize, func=None):
        batchSize = int(batchSize)
        dataSize = len(index)
        for i in range(dataSize / batchSize+1):
            batchIndex = index[i*batchSize:(i+1)*batchSize]
            batch = self.data[batchIndex].tolist()
            if func:
                batch = [func(d) for d in batch]
            yield batch

    def getTrainBatch(self, batchSize, func=None):
        for batch in self._getBatch(self.trainIndex, batchSize, func):
            yield batch

    def getTestBatch(self, batchSize, func=None):
        for batch in self._getBatch(self.testIndex, batchSize, func):
            yield batch

    def getSentLen(self):
        return len(self.data[0]["reviewText"])
