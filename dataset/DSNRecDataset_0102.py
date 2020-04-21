# coding=utf8
import os
import random
from collections import defaultdict
from collections import OrderedDict
from itertools import izip, cycle, islice
import numpy as np
import pandas as pd
from glob import glob

import sys
sys.path.append(os.path.abspath('/software/home/jinyaru/DSNRec_1024')) 
from utils import pkload, pkdump
from preprocess.utils import readJson


class DSNRecDataset:
    def __init__(self, args):#output_dir=None,
        self.input_dir = args.data_dir
        self.src_domain = args.src_domain
        self.tgt_domain = args.tgt_domain 
        self.u_num_comment = args.u_num_comment 
        self.i_num_comment = args.i_num_comment 
        self.u_less = args.u_less 
        self.i_less = args.i_less
        self.fold = args.fold   
        self.drop_keep = args.drop_keep
        self.usersplit = self.__getUserTrainTest()
        
    def getUIShp(self):
        return self.u_num_comment, self.i_num_comment
    
    def __getUserTrainTest(self):
        df_SrcTrain = pd.read_csv(self.input_dir+'/%s_%s/uThan%s_iThan%s/fold_%s/src_train.csv'%(self.src_domain,self.tgt_domain,self.u_less,self.i_less,self.fold))
        df_SrcTest = pd.read_csv(self.input_dir+'/%s_%s/uThan%s_iThan%s/fold_%s/src_test.csv'%(self.src_domain,self.tgt_domain,self.u_less,self.i_less,self.fold))
        df_TgtTrain = pd.read_csv(self.input_dir+'/%s_%s/uThan%s_iThan%s/fold_%s/tgt_train.csv'%(self.src_domain,self.tgt_domain,self.u_less,self.i_less,self.fold))
        df_TgtTest = pd.read_csv(self.input_dir+'/%s_%s/uThan%s_iThan%s/fold_%s/tgt_test.csv'%(self.src_domain,self.tgt_domain,self.u_less,self.i_less,self.fold))
        df_SrcTrain = df_SrcTrain.sample(frac=1.0,random_state=2020)
        df_SrcTrain = df_SrcTrain.reset_index(drop=True)
        
        df_SrcTest = df_SrcTest.sample(frac=1.0,random_state=2020)
        df_SrcTest = df_SrcTest.reset_index(drop=True)
        
        df_TgtTrain = df_TgtTrain.sample(frac=1.0,random_state=2020)
        df_TgtTrain = df_TgtTrain.reset_index(drop=True)
        
        df_TgtTest = df_TgtTest.sample(frac=1.0,random_state=2020)
        df_TgtTest = df_TgtTest.reset_index(drop=True)
        
        def reget_u_list(arr):
            arr = eval(arr)
            arr.sort()
            if len(arr)<self.u_num_comment:
                arr = [0]*(self.u_num_comment-len(arr)) + arr
            else:
                arr = arr[-self.u_num_comment:]
            return str(arr)
        def reget_i_list(arr):
            arr = eval(arr)
            arr.sort()
            if len(arr)<self.i_num_comment:
                arr = [0]*(self.i_num_comment-len(arr)) + arr
            else:
                arr = arr[-self.i_num_comment:]        
            return str(arr)
        df_SrcTrain['u_list'] = map(reget_u_list, df_SrcTrain['u_list'].to_list())
        df_SrcTrain['i_list'] = map(reget_i_list, df_SrcTrain['i_list'].to_list())
        df_SrcTest['u_list'] = map(reget_u_list, df_SrcTest['u_list'].to_list())
        df_SrcTest['i_list'] = map(reget_i_list, df_SrcTest['i_list'].to_list())
        df_TgtTrain['u_list'] = map(reget_u_list, df_TgtTrain['u_list'].to_list())
        df_TgtTrain['i_list'] = map(reget_i_list, df_TgtTrain['i_list'].to_list())
        df_TgtTest['u_list'] = map(reget_u_list, df_TgtTest['u_list'].to_list())
        df_TgtTest['i_list'] = map(reget_i_list, df_TgtTest['i_list'].to_list())
        uid_all = df_SrcTrain['uid'].to_list()+df_SrcTest['uid'].to_list()+df_TgtTrain['uid'].to_list()+df_TgtTest['uid'].to_list()
        uid_list_all = df_SrcTrain['u_list'].to_list()+df_SrcTest['u_list'].to_list()+df_TgtTrain['u_list'].to_list()+df_TgtTest['u_list'].to_list()
        iid_all = df_SrcTrain['iid'].to_list()+df_SrcTest['iid'].to_list()+df_TgtTrain['iid'].to_list()+df_TgtTest['iid'].to_list()
        iid_list_all = df_SrcTrain['i_list'].to_list()+df_SrcTest['i_list'].to_list()+df_TgtTrain['i_list'].to_list()+df_TgtTest['i_list'].to_list()
        self.dict_u_list = dict(zip(uid_all,uid_list_all)) ######
        self.dict_i_list = dict(zip(iid_all,iid_list_all)) ######
           
        src_train = zip(zip(df_SrcTrain['uid'].to_list(),df_SrcTrain['iid'].to_list()),df_SrcTrain['rating'].to_list())
        src_test = zip(zip(df_SrcTest['uid'].to_list(),df_SrcTest['iid'].to_list()),df_SrcTest['rating'].to_list())
        tgt_train = zip(zip(df_TgtTrain['uid'].to_list(),df_TgtTrain['iid'].to_list()),df_TgtTrain['rating'].to_list())
        tgt_test = zip(zip(df_TgtTest['uid'].to_list(),df_TgtTest['iid'].to_list()),df_TgtTest['rating'].to_list())
        # print('type(src_train):',type(src_train),'type(src_test):',type(src_test),'type(tgt_train):',type(tgt_train),'type(tgt_test):',type(tgt_test))
        return {"src": {"train": src_train, "test": src_test},
                "tgt": {"train": tgt_train, "test": tgt_test}}
                
        
        
    def generateTrainBatch(self, batchSize):
        data = self.usersplit
        # print('type(data)',type(data))
        return self._generateBatch(data, batchSize, "train"),self.srctrainbatch,self.tgttrainbatch#(max(self.srctrainbatch,self.tgttrainbatch)//min(self.srctrainbatch,self.tgttrainbatch)+1) *min(self.srctrainbatch,self.tgttrainbatch)

    def generateTestBatch(self, batchSize=10000000):
        data = self.usersplit
        # return islice(self._generateBatch(data, batchSize, "test"),
                      # self.srctestbatch),self.srctestbatch
        return self._generateBatch(data, batchSize, "test")

    def _generateBatch(self, data, batchSize, train_or_test):
        if train_or_test=='train':
            # print("len(data['src']['train']):",len(data['src']['train']),'batchSize:',batchSize)
            self.srctrainbatch = int(len(data['src']['train'])/batchSize)+1
            self.tgttrainbatch = int(len(data['tgt']['train'])/batchSize)+1
            # self.srctrainbatch = 100
            # self.tgttrainbatch = 200
        elif train_or_test=='test':
            self.srctestbatch = int(len(data["src"]['test'])/batchSize)+1
            self.tgttestbatch = int(len(data["tgt"]['test'])/batchSize)+1
            # self.srctestbatch = 100
            # self.tgttestbatch = 200
        return self.__generateBatch(data, train_or_test,batchSize)

        
    def __generateBatch(self, data, train_or_test,batchSize):
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

        if train_or_test == "train":
            for src, tgt in izip(
                    cycle(takenBatch(data["src"][train_or_test])),
                    cycle(takenBatch(data["tgt"][train_or_test]))): #cycle('abc')    #重复序列的元素，既a, b, c, a, b, c ...
                srcu_vec, srci_vec = [], []
                src_rating = []
                #print('train type(src)',type(src))#('train type(src)', <type 'list'>)
                for (u, i), r in src:
                    srcu_vec.append(self.dict_u_list[u])
                    srci_vec.append(self.dict_i_list[i])
                    src_rating.append(r)
                
                srcu_vec = map(eval,srcu_vec)
                srci_vec = map(eval,srci_vec)
                srcu_vec = np.array(srcu_vec)
                srci_vec = np.array(srci_vec)
                src_rating = np.array(src_rating)

                tgtu_vec, tgti_vec = [], []
                tgt_rating = []
                for (u, i), r in tgt:
                    tgtu_vec.append(self.dict_u_list[u])
                    tgti_vec.append(self.dict_i_list[i])
                    tgt_rating.append(r)

                tgtu_vec = map(eval,tgtu_vec)
                tgti_vec = map(eval,tgti_vec)
                tgtu_vec = np.array(tgtu_vec)
                tgti_vec = np.array(tgti_vec)
                tgt_rating = np.array(tgt_rating)

                yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating},
                       "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating},
                       'keep_prob':self.drop_keep}
                       
        elif train_or_test == "test":
            src = data["src"][train_or_test]
            tgt = data["tgt"][train_or_test]
            srcu_vec, srci_vec = [], []
            src_rating = []
            #print('train type(src)',type(src))#('train type(src)', <type 'list'>)
            for (u, i), r in src:
                srcu_vec.append(self.dict_u_list[u])
                srci_vec.append(self.dict_i_list[i])
                src_rating.append(r)
            
            srcu_vec = map(eval,srcu_vec)
            srci_vec = map(eval,srci_vec)
            srcu_vec = np.array(srcu_vec)
            srci_vec = np.array(srci_vec)
            src_rating = np.array(src_rating)

            tgtu_vec, tgti_vec = [], []
            tgt_rating = []
            for (u, i), r in tgt:
                tgtu_vec.append(self.dict_u_list[u])
                tgti_vec.append(self.dict_i_list[i])
                tgt_rating.append(r)

            tgtu_vec = map(eval,tgtu_vec)
            tgti_vec = map(eval,tgti_vec)
            tgtu_vec = np.array(tgtu_vec)
            tgti_vec = np.array(tgti_vec)
            tgt_rating = np.array(tgt_rating)

            yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating},
                   "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating},
                   'keep_prob':1.0}
            
        else:
            raise Exception("train_or_test should in ['train', 'test']")    
