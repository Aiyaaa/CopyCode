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
        self.negRatio = args.negRatio
        self.train_neg_sample = args.train_neg_sample
        self.test_neg_sample = args.test_neg_sample
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
        
        #####ranking
        df_SrcTrain['rating'] = [[1,0]]*len(df_SrcTrain)# = 1
        df_SrcTest['rating'] = [[1,0]]*len(df_SrcTest)# = 1
        df_TgtTrain['rating'] = [[1,0]]*len(df_TgtTrain)# = 1
        df_TgtTest['rating'] = [[1,0]]*len(df_TgtTest)# = 1   
        ###neg_dic_src_train
        neg_path = self.input_dir + '/%s_%s/uThan%s_iThan%s/fold_%s/'%(self.src_domain,self.tgt_domain,self.u_less,self.i_less,self.fold)
        src_trainNegatives = {} 
        src_trainNegatives_init = pkload(neg_path+'train_%s_NegItemsListUDict_sample%s.pk'%(self.src_domain,self.train_neg_sample))
        src_tr_items_keys = src_trainNegatives_init.keys()
        for key in src_tr_items_keys:
            src_trainNegatives[key] = src_trainNegatives_init[key][:self.negRatio]
        ###neg_dic_src_test
        src_testNegatives = pkload(neg_path+'test_%s_NegItemsListUDict_sample%s.pk'%(self.src_domain,self.test_neg_sample))    
        ###neg_dic_tgt_train
        tgt_trainNegatives = {} 
        tgt_trainNegatives_init = pkload(neg_path+'train_%s_NegItemsListUDict_sample%s.pk'%(self.tgt_domain,self.train_neg_sample)) #args.negRatio,self.train_neg_sample,self.test_neg_sample
        tgt_tr_items_keys = tgt_trainNegatives_init.keys()
        for key in tgt_tr_items_keys:
            tgt_trainNegatives[key] = tgt_trainNegatives_init[key][:self.negRatio] 
        ###neg_dic_tgt_test
        tgt_testNegatives = pkload(neg_path+'test_%s_NegItemsListUDict_sample%s.pk'%(self.tgt_domain,self.test_neg_sample))         
        ###neg_df
        def get_new_df(negUDict):
            new_neg_df_dic = {}
            iid_list = []
            uid_list = []
            u_list_list = []
            i_list_list = []
            rat_list = []
            for (user,neg_items) in negUDict.items(): #src_trainNegatives:
                cur_iid_list = []
                cur_uid_list = []
                cur_u_list_list = []
                cur_i_list_list = []
                cur_rat_list = []
                for neg_item in neg_items:
                    cur_iid_list.append(neg_item)
                    cur_uid_list.append(user)
                    cur_u_list_list.append(self.dict_u_list[user])
                    cur_i_list_list.append(self.dict_i_list[neg_item])
                    cur_rat_list.append([0,1]) #cur_rat_list.append(0)
                iid_list.extend(cur_iid_list)
                uid_list.extend(cur_uid_list)
                u_list_list.extend(cur_u_list_list)
                i_list_list.extend(cur_i_list_list)
                rat_list.extend(cur_rat_list)
            new_neg_df_dic['uid'] = uid_list
            new_neg_df_dic['iid'] = iid_list
            new_neg_df_dic['u_list'] = u_list_list
            new_neg_df_dic['i_list'] = i_list_list
            new_neg_df_dic['rating'] = rat_list
            new_neg_df = pd.DataFrame.from_dict(new_neg_df_dic)
            return new_neg_df
        print('src_trainNegatives.items()[:2]:',src_trainNegatives.items()[:2])   
        neg_df_src_train = get_new_df(src_trainNegatives)    
        neg_df_src_test = get_new_df(src_testNegatives)
        neg_df_tgt_train = get_new_df(tgt_trainNegatives)
        neg_df_tgt_test = get_new_df(tgt_testNegatives)
        print('len(df_SrcTrain):%s,len(df_SrcTest):%s,len(df_TgtTrain):%s,len(df_TgtTest):%s,'%(len(df_SrcTrain),len(df_SrcTest),len(df_TgtTrain),len(df_TgtTest)))
        print('len(neg_df_src_train):%s,len(neg_df_src_test):%s,len(neg_df_tgt_train):%s,len(neg_df_tgt_test):%s,'%(len(neg_df_src_train),len(neg_df_src_test),len(neg_df_tgt_train),len(neg_df_tgt_test)))
        df_SrcTrain = df_SrcTrain.append(neg_df_src_train)
        df_SrcTest = df_SrcTest.append(neg_df_src_test)
        df_TgtTrain = df_TgtTrain.append(neg_df_tgt_train)
        df_TgtTest = df_TgtTest.append(neg_df_tgt_test)
        df_SrcTrain = df_SrcTrain.sample(frac=1.0,random_state=2020)
        df_SrcTrain = df_SrcTrain.reset_index(drop=True)
        df_SrcTest = df_SrcTest.sample(frac=1.0,random_state=2020)
        df_SrcTest = df_SrcTest.reset_index(drop=True)
        df_TgtTrain = df_TgtTrain.sample(frac=1.0,random_state=2020)
        df_TgtTrain = df_TgtTrain.reset_index(drop=True)
        df_TgtTest = df_TgtTest.sample(frac=1.0,random_state=2020)
        df_TgtTest = df_TgtTest.reset_index(drop=True)
        print('len(df_SrcTrain):%s,len(df_SrcTest):%s,len(df_TgtTrain):%s,len(df_TgtTest):%s,'%(len(df_SrcTrain),len(df_SrcTest),len(df_TgtTrain),len(df_TgtTest)))
        src_train = zip(zip(df_SrcTrain['uid'].to_list(),df_SrcTrain['iid'].to_list()),df_SrcTrain['rating'].to_list())
        src_test = zip(zip(df_SrcTest['uid'].to_list(),df_SrcTest['iid'].to_list()),df_SrcTest['rating'].to_list())
        tgt_train = zip(zip(df_TgtTrain['uid'].to_list(),df_TgtTrain['iid'].to_list()),df_TgtTrain['rating'].to_list())
        tgt_test = zip(zip(df_TgtTest['uid'].to_list(),df_TgtTest['iid'].to_list()),df_TgtTest['rating'].to_list())
        print('len(src_train):%s,len(src_test):%s,len(tgt_train):%s,len(tgt_test):%s,'%(len(src_train),len(src_test),len(tgt_train),len(tgt_test)))

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
                src_uid, src_iid = [], []
                src_rating = []
                #print('train type(src)',type(src))#('train type(src)', <type 'list'>)
                for (u, i), r in src:
                    srcu_vec.append(self.dict_u_list[u])
                    srci_vec.append(self.dict_i_list[i])
                    src_rating.append(r)
                    src_uid.append(u)
                    src_iid.append(i)
                
                srcu_vec = map(eval,srcu_vec)
                srci_vec = map(eval,srci_vec)
                srcu_vec = np.array(srcu_vec)
                srci_vec = np.array(srci_vec)
                src_rating = np.array(src_rating)
                src_uid = np.array(src_uid)
                src_iid = np.array(src_iid)

                tgtu_vec, tgti_vec = [], []
                tgt_uid, tgt_iid = [], []
                tgt_rating = []
                for (u, i), r in tgt:
                    tgtu_vec.append(self.dict_u_list[u])
                    tgti_vec.append(self.dict_i_list[i])
                    tgt_rating.append(r)
                    tgt_uid.append(u)
                    tgt_iid.append(i)

                tgtu_vec = map(eval,tgtu_vec)
                tgti_vec = map(eval,tgti_vec)
                tgtu_vec = np.array(tgtu_vec)
                tgti_vec = np.array(tgti_vec)
                tgt_rating = np.array(tgt_rating)
                tgt_uid = np.array(tgt_uid)
                tgt_iid = np.array(tgt_iid)

                yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating, 'uid':src_uid, 'iid':src_iid},
                       "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating, 'uid':tgt_uid, 'iid':tgt_iid},
                       'keep_prob':self.drop_keep}
                       
        elif train_or_test == "test":
            src = data["src"][train_or_test]
            tgt = data["tgt"][train_or_test]
            srcu_vec, srci_vec = [], []
            src_uid, src_iid = [], []
            src_rating = []
            #print('train type(src)',type(src))#('train type(src)', <type 'list'>)
            for (u, i), r in src:
                srcu_vec.append(self.dict_u_list[u])
                srci_vec.append(self.dict_i_list[i])
                src_rating.append(r)
                src_uid.append(u)
                src_iid.append(i)
            
            srcu_vec = map(eval,srcu_vec)
            srci_vec = map(eval,srci_vec)
            srcu_vec = np.array(srcu_vec)
            srci_vec = np.array(srci_vec)
            src_rating = np.array(src_rating)
            src_uid = np.array(src_uid)
            src_iid = np.array(src_iid)

            tgtu_vec, tgti_vec = [], []
            tgt_uid, tgt_iid = [], []
            tgt_rating = []
            for (u, i), r in tgt:
                tgtu_vec.append(self.dict_u_list[u])
                tgti_vec.append(self.dict_i_list[i])
                tgt_rating.append(r)
                tgt_uid.append(u)
                tgt_iid.append(i)

            tgtu_vec = map(eval,tgtu_vec)
            tgti_vec = map(eval,tgti_vec)
            tgtu_vec = np.array(tgtu_vec)
            tgti_vec = np.array(tgti_vec)
            tgt_rating = np.array(tgt_rating)
            tgt_uid = np.array(tgt_uid)
            tgt_iid = np.array(tgt_iid)

            yield {"src": {"user": srcu_vec, "item": srci_vec, "rating": src_rating, 'uid':src_uid, 'iid':src_iid},
                   "tgt": {"user": tgtu_vec, "item": tgti_vec, "rating": tgt_rating, 'uid':tgt_uid, 'iid':tgt_iid},
                   'keep_prob':1.0}
            
        else:
            raise Exception("train_or_test should in ['train', 'test']")    
