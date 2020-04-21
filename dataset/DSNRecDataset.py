# coding=utf8
import os
import random
from collections import defaultdict
from collections import OrderedDict
from itertools import izip, cycle, islice
import numpy as np
from glob import glob

import sys
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024')) 
from utils import pkload, pkdump
from preprocess.utils import readJson


class DSNRecDataset:
    def __init__(self, input_dir, cold_dir,
                 src_domain, tgt_domain, if_exchange, overlap_rate=1.0):#output_dir=None,
        """
        if_exchange:文件夹名 domain1_domain2，默认src-domain1,tgt-domain2，若if_exchange==True,则src-domain2，tgt-domain1
        """
        # input_dir e.g: exam/data/preprocess/uirepresent
        # cold_dir e.g: exam/data/preprocess/cold
        self.overlap_rate=overlap_rate
        self.cold_dir = cold_dir
        # self._seperateUser()
        self.input_dir = input_dir
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
#         self.output_dir = output_dir
        # 源领域与目标领域item的向量表示
        # type: {item_id: np.array(feature_size)}
        self.src_item, self.tgt_item = self._buildData("item*")
        # 源领域与目标领域user的向量表示
        # type: {user_id: np.array(feature_size)}
        self.src_user, self.tgt_user = self._buildData("user*")
        self.src_rating_time, self.tgt_rating_time = self._buildData("rating_time*")

        self.w = self._getCommentEmbedding("comment*",if_exchange)
        
        # rating 跟 time 信息存储在同一个文件中, 需要分开
        self.src_timeinfo, self.src_rating = self._seperateRateTime(self.src_rating_time)
        self.tgt_timeinfo, self.tgt_rating = self._seperateRateTime(self.tgt_rating_time)

        self.timesplit = self.__getTimeSplitTrainTest()
        self.usersplit = self.__getUserSplitTrainTest()

    def getUIShp(self):
        return len(self.src_user.values()[0]), len(self.src_item.values()[0])#return self.src_user.values()[0].shape[0], self.src_item.values()[0].shape[0]

    def __getUserSplitTrainTest(self):
        for fn in os.listdir(self.cold_dir):
            if self.src_domain in fn:
                fn = os.path.join(self.cold_dir, fn)
                self.src_user_cold = pkload(fn)
            elif self.tgt_domain in fn:
                fn = os.path.join(self.cold_dir, fn)
                self.tgt_user_cold = pkload(fn)
            else:
                fn = os.path.join(self.cold_dir, fn)
                self.overlap_user = pkload(fn)
                random.shuffle(self.overlap_user)
                test_num = int(len(self.overlap_user) * self.overlap_rate)
                self.overlap_user = self.overlap_user[:test_num]

        src_u = defaultdict(list)
        tgt_u = defaultdict(list)
        for ui, rating in self.src_rating.items():
            src_u[ui[0]].append((ui, rating))

        for ui, rating in self.tgt_rating.items():
            tgt_u[ui[0]].append((ui, rating))

        src_train = []
        for u in self.src_user_cold:
            src_train.extend(src_u[u])

        src_test = []
        for u in self.overlap_user:
            src_test.extend(src_u[u])

        tgt_train = []
        for u in self.tgt_user_cold:
            tgt_train.extend(tgt_u[u])

        tgt_test = []
        for u in self.overlap_user:
            tgt_test.extend(tgt_u[u])

        return {"src": {"train": src_train, "test": src_test},
                "tgt": {"train": tgt_train, "test": tgt_test}}

    def __getTimeSplitTrainTest(self):
        self.splitTime = self._getSplitTime()
        src_train, src_test = self._splitTrainTest(
                self.splitTime, "src")
        tgt_train, tgt_test = self._splitTrainTest(
                self.splitTime, "tgt")
        return {"src": {"train": src_train, "test": src_test},
                "tgt": {"train": tgt_train, "test": tgt_test}}

    def _seperateRateTime(self, data):
        dtime = defaultdict(list)
        rating = {}
        for ku_i, vtime in data.items():
            dtime[vtime[1]].append(ku_i)
            rating[ku_i] = vtime[0]

        return dict(dtime), rating

    def _buildData(self, type):
        print ('os.path.join(self.input_dir, type+self.src_domain+"*.pk") :',os.path.join(self.input_dir, type+self.src_domain+"*.pk")) 
      
        for f in glob(os.path.join(self.input_dir,
                                   type+self.src_domain)+"*.pk"):
            src_data = pkload(f)

        for f in glob(os.path.join(self.input_dir,
                                   type+self.tgt_domain)+"*.pk"):
            tgt_data = pkload(f)

        return src_data, tgt_data
    
    def _getCommentEmbedding(self, type, if_exchange):
        print ('os.path.join(self.input_dir, type+self.src_domain+"*.pk") :',os.path.join(self.input_dir, type+self.src_domain+"*.pk")) 
      
        for f in glob(os.path.join(self.input_dir,
                                   type+self.src_domain)+"*.pk"):
            src_Comment_Embedding = pkload(f)

        for f in glob(os.path.join(self.input_dir,
                                   type+self.tgt_domain)+"*.pk"):
            tgt_Comment_Embedding = pkload(f)
        if if_exchange:
            src_Comment_Embedding,tgt_Comment_Embedding = tgt_Comment_Embedding,src_Comment_Embedding

        a = list(src_Comment_Embedding.keys())
        b = list(tgt_Comment_Embedding.keys())
        a.sort()
        b.sort()
        w = []
        w.append([0]*len(src_Comment_Embedding[a[0]]))
        for i in list(range(len(a))):
            w.append(src_Comment_Embedding[a[i]])
        for i in list(range(len(b))):
            w.append(tgt_Comment_Embedding[b[i]])
        w = np.array(w)
        print('comment_embedding.shape:',w.shape)
        return w

    def _getSplitTime(self):
        odt = OrderedDict()
        for t in set(self.src_timeinfo.keys()+self.tgt_timeinfo.keys()):
            odt[t] = len(self.src_timeinfo.get(t, []))+len(self.tgt_timeinfo.get(t, []))
        total = len(self.src_rating)+len(self.tgt_rating)
        threshold = total * 0.7
        s = 0
        for splittime, c in odt.items():
            s += c
            if s < threshold:
                continue
            return splittime

    def _splitTrainTest(self, splitTime, type):
        rating = getattr(self, type+"_rating")
        timeinfo = getattr(self, type+"_timeinfo")
        train, test = [], []
        for time in timeinfo:
            if time <= splitTime:
                train.extend(timeinfo[time])
            else:
                test.extend(timeinfo[time])
        train = [(ui, rating[ui]) for ui in train]
        test = [(ui, rating[ui]) for ui in test]
        return train, test

    def generateTrainBatch(self, type, batchSize):
        u'''
        type in ['user', 'time']
        default = 'user'
        '''
        data = self.timesplit if type == 'time' else self.usersplit
        #return self._generateBatch(data, batchSize, "train")
        #return islice(self._generateBatch(data, batchSize, "train"),max(self.srctrainbatch,self.tgttrainbatch)*2)
        return self._generateBatch(data, batchSize, "train"),max(self.srctrainbatch,self.tgttrainbatch)*2

    def generateTestBatch(self, type, batchSize):
        data = self.timesplit if type == 'time' else self.usersplit
        # return islice(self._generateBatch(data, batchSize, "test"),
                      # self.srctestbatch),self.srctestbatch
        return self._generateBatch(data, batchSize, "test")

    def _generateBatch(self, data, batchSize, train_or_test):

        setattr(self, "src"+train_or_test+"batch",
                int(len(data["src"][train_or_test])/batchSize)+1)
        print("self.src%sbatch"%(train_or_test),int(len(data["src"][train_or_test])/batchSize)+1,'len of data:',len(data["src"][train_or_test]))
        setattr(self, "tgt"+train_or_test+"batch",
                int(len(data["tgt"][train_or_test])/batchSize)+1)
        print("self.tgt%sbatch"%(train_or_test),int(len(data["tgt"][train_or_test])/batchSize)+1,'len of data:',len(data["tgt"][train_or_test]))
        return self.__generateBatch(batchSize, data, train_or_test)

    def __generateBatch(self, batchSize, data, train_or_test):
        u'''
        Parameters
        ----------
        batchSize: int
        data: dict
            {"src": {"train": dataset, "test": dataset},
             "tgt": {"train": dataset, "test": dataset}}
            dataset: [((userid, itemid), rating)]
        train_or_test: str
            belong in ['train', 'test']

        Yield
        -----
        {"src": {"user": np.array, "item": np.array, "rating": np.array},
         "tgt": {"user": np.array, "item": np.array, "rating": np.array}}
        '''

        def takenBatch(data):
            total = len(data)
            for i in range(int(total/batchSize)+1):
                yield data[i*batchSize: (i+1)*batchSize]
#         if train_or_test == "train":
#             self.train_num_batch = int(len(data)/batchSize)+1
#             print('self.train_num_batch:',self.train_num_batch)
        # 训练过程中, 用源领域用户进行训练, 测试过程用tgt领域用户的特征预测
        if train_or_test == "train":
            suser, tuser = self.src_user, self.tgt_user
            for src, tgt in izip(
                    cycle(takenBatch(data["src"][train_or_test])),
                    cycle(takenBatch(data["tgt"][train_or_test]))): #cycle('abc')    #重复序列的元素，既a, b, c, a, b, c ...
                srcu_vec, srci_vec = [], []
                src_rating = []
                #print('train type(src)',type(src))#('train type(src)', <type 'list'>)
                for (u, i), r in src:
                    srcu_vec.append(suser[u])
                    srci_vec.append(self.src_item[i])
                    src_rating.append(r)

                srcu_vec = np.array(srcu_vec)
                srci_vec = np.array(srci_vec)
                src_rating = np.array(src_rating)

                tgtu_vec, tgti_vec = [], []
                tgt_rating = []
                for (u, i), r in tgt:
                    tgtu_vec.append(tuser[u])
                    tgti_vec.append(self.tgt_item[i])
                    tgt_rating.append(r)

                tgtu_vec = np.array(tgtu_vec)
                tgti_vec = np.array(tgti_vec)
                tgt_rating = np.array(tgt_rating)

                yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating},
                       "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating}}
    
        elif train_or_test == "test":
            suser, tuser = self.tgt_user, self.src_user
            src = data["src"][train_or_test]
            tgt = data["tgt"][train_or_test]
            srcu_vec, srci_vec = [], []
            src_rating = []
            #print('type(src)',type(src))#('type(src)', <type 'list'>)
            for (u, i), r in src:
                srcu_vec.append(suser[u])
                srci_vec.append(self.src_item[i])
                src_rating.append(r)

            srcu_vec = np.array(srcu_vec)
            srci_vec = np.array(srci_vec)
            src_rating = np.array(src_rating)

            tgtu_vec, tgti_vec = [], []
            tgt_rating = []
            for (u, i), r in tgt:
                tgtu_vec.append(tuser[u])
                tgti_vec.append(self.tgt_item[i])
                tgt_rating.append(r)

            tgtu_vec = np.array(tgtu_vec)
            tgti_vec = np.array(tgti_vec)
            tgt_rating = np.array(tgt_rating)

            yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating},
                   "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating}}
        else:
            raise Exception("train_or_test should in ['train', 'test']")

