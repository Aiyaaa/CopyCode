# -*- coding: utf-8 -*-
#算的是交叠的dataframe里 user,item出现的次数
# nohup python /data1/home/jinyaru/DSNRec_1024/exam/preprocess/sele_data/form_data_for_allmethod_0123.py > /data1/home/jinyaru/DSNRec_1024/exam/preprocess/sele_data/form_data_for_allmethod_0123.log 2>&1 &
# python form_data_for_allmethod_0123.py --source_name Electronics --target_name Sports_and_Outdoors --u_less_list '[25]' --i_less_list '[5]'

import pandas as pd
import os
import sys
import click
from itertools import chain
from operator import itemgetter
from glob import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as io
from sklearn import preprocessing
import random
from sklearn.model_selection import KFold
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024/')) 
from preprocess.utils import readJson
from utils import pkload, pkdump


@click.group()
def cli():
    pass



def about_overlap(source_name,target_name,outputpath):
    json_path = os.path.join(outputpath, "transform/")  
    sourceDomain = json_path+'reviews_%s_5.json'%source_name
    targetDomain = json_path+'reviews_%s_5.json'%target_name
    '''生成src和tgt的交叠部分的所有样例，df_Src_over,df_Src_over ， ["user", "sourceItem",'overall','time']  ["user", "targetItem",'overall','time']

    Parameters
    ----------
    source_name : str
        源域名字

    target_name : str
        目标域名字
    outputpath ： 存储路径
    
    Returns
    -------
        df_Src_over : pd.DataFrame
        df_Src_over : pd.DataFrame
    '''

    Source = list(readJson(sourceDomain))
    SourceUserItem = [(d["reviewerID"], d["asin"], d["overall"], d["unixReviewTime"]) for d in Source]
    del Source
    print('load %s'%sourceDomain)
    Traget = list(readJson(targetDomain))
    TargetUserItem = [(d["reviewerID"], d["asin"], d["overall"], d["unixReviewTime"]) for d in Traget]
    del Traget
    print('load %s'%targetDomain)
    uiS = pd.DataFrame(data=SourceUserItem, columns=["user", "sourceItem", "overall",'time'])
    uiT = pd.DataFrame(data=TargetUserItem, columns=["user", "targetItem", "overall",'time'])

    """
    下面是新加的
    """
    arr_UOver = set(uiS['user'].unique())&set(uiT['user'].unique())
    df_Tgt_over = uiT[uiT['user'].isin(arr_UOver)]
    df_Src_over = uiS[uiS['user'].isin(arr_UOver)]
    a = list(df_Tgt_over['user'].unique())
    a.sort()
    b = list(df_Src_over['user'].unique())
    b.sort()
    print('if user in src and tgt is same:',a==b)
    print("%s and %s, overlap user count is %d"%(source_name,target_name, len(arr_UOver)))
    print('')
    return df_Src_over,df_Tgt_over

def selectDf(u_less,i_less,df_Src_over,df_Tgt_over):
    '''选出df_Src_over,df_Tgt_over中用户评论数超过u_less及商品评论数超过i_less的行，

    Parameters
    ----------
    u_less : int
        用户最少评论数

    i_less : int
        item最少评论数
    
    Returns
    -------
        df_Srcresult_use_select  : pd.DataFrame
        df_Tgtresult_use_select : pd.DataFrame
    '''
    print('u_less:',u_less,'i_less:',i_less)
    df_SrcUidCount = df_Src_over[["user",'overall']].groupby("user").count().reset_index()
    df_SrcUidCount = df_SrcUidCount.rename(columns={'overall':'CountUid'})
    df_SrcIidCount = df_Src_over[["sourceItem",'overall']].groupby("sourceItem").count().reset_index()
    df_SrcIidCount = df_SrcIidCount.rename(columns={'overall':'CountIid'})
    df_Srcresult = pd.merge(df_Src_over, df_SrcUidCount, how='left', on=['user'])
    df_Srcresult_use = pd.merge(df_Srcresult, df_SrcIidCount, how='left', on=['sourceItem'])
    
    df_TgtUidCount = df_Tgt_over[["user",'overall']].groupby("user").count().reset_index()
    df_TgtUidCount = df_TgtUidCount.rename(columns={'overall':'CountUid'})
    df_TgtIidCount = df_Tgt_over[["targetItem",'overall']].groupby("targetItem").count().reset_index()
    df_TgtIidCount = df_TgtIidCount.rename(columns={'overall':'CountIid'})
    df_Tgtresult = pd.merge(df_Tgt_over, df_TgtUidCount, how='left', on=['user'])
    df_Tgtresult_use = pd.merge(df_Tgtresult, df_TgtIidCount, how='left', on=['targetItem'])
    
    a = list(df_Tgtresult_use['user'].unique())
    a.sort()
    b = list(df_Srcresult_use['user'].unique())
    b.sort()
    print('if user in src and tgt is same:',a==b)
    
    uid_src_sele = df_Srcresult_use[(df_Srcresult_use['CountUid']>=u_less)&(df_Srcresult_use['CountIid']>=i_less)]['user'].unique()
    uid_tgt_sele = df_Tgtresult_use[(df_Tgtresult_use['CountUid']>=u_less)&(df_Tgtresult_use['CountIid']>=i_less)]['user'].unique()
    sele_uid_list = list(set(uid_src_sele)&set(uid_tgt_sele))
    df_Srcresult_use_select = df_Srcresult_use[df_Srcresult_use['user'].isin(sele_uid_list)]
    df_Tgtresult_use_select = df_Tgtresult_use[df_Tgtresult_use['user'].isin(sele_uid_list)]
    len_select_src = len(df_Srcresult_use_select)
    print('len(df_Srcresult):',len(df_Srcresult),'len_select_src:',len_select_src)
    len_select_tgt = len(df_Tgtresult_use_select)
    print('len(df_Tgtresult_use):',len(df_Tgtresult_use),'len_select_tgt:',len_select_tgt)
      
    df_Srcresult_use_select = df_Srcresult_use_select.reset_index(drop=True)
    df_Srcresult_use_select = df_Srcresult_use_select.rename(columns={'overall':'rating','sourceItem':'iid','user':'uid'})
    print('df_Srcresult_use_select',df_Srcresult_use_select.columns.values.tolist())
    df_Tgtresult_use_select = df_Tgtresult_use_select.reset_index(drop=True)
    df_Tgtresult_use_select = df_Tgtresult_use_select.rename(columns={'overall':'rating','targetItem':'iid','user':'uid'})
    print('df_Tgtresult_use_select',df_Tgtresult_use_select.columns.values.tolist())    
    
    a = list(df_Tgtresult_use['user'].unique())
    a.sort()
    b = list(df_Srcresult_use['user'].unique())
    b.sort()
    print('if user in src and tgt is same:',a==b)
    
    return df_Srcresult_use_select, df_Tgtresult_use_select

def save_w_src_tgt(df_SeleSrc,df_SeleTgt,u_less,i_less,if_time,outputpath,source_name,target_name):
    '''作用
    df_SeleSrc： 源域选出的数据，['uid','iid','rating','time']
    df_SeleTgt： 目标域选出的数据，['uid','iid','rating','time']
    '''
    if if_time:
        df_SeleSrc = df_SeleSrc.sort_values(by='time')
        df_SeleTgt = df_SeleTgt.sort_values(by='time')
        df_SeleSrc = df_SeleSrc.reset_index(drop=True)
        df_SeleTgt = df_SeleTgt.reset_index(drop=True)
    def GetCommentIndex_SU(uid):
        arr_index = np.array(df_SeleSrc[df_SeleSrc.loc[:,'uid']==uid].index)+1
        arr_index = list(arr_index)
        arr_index.sort()
        return str(arr_index)
    
    def GetCommentIndex_SI(iid):
        arr_index = np.array(df_SeleSrc[df_SeleSrc.loc[:,'iid']==iid].index)+1
        arr_index = list(arr_index)
        arr_index.sort()
        return str(arr_index)

    # def GetCommentIndex_TU(uid):
        # arr_index = np.array(df_SeleTgt[df_SeleTgt.loc[:,'uid']==uid].index)+1+len(df_SeleSrc)
        # return str(list(arr_index))
    def GetCommentIndex_TI(iid):
        arr_index = np.array(df_SeleTgt[df_SeleTgt.loc[:,'iid']==iid].index)+1+len(df_SeleSrc)
        arr_index = list(arr_index)
        arr_index.sort()
        return str(arr_index)
    
    df_SrcUlist = pd.DataFrame()
    df_SrcIlist = pd.DataFrame()
    # df_TgtUlist = pd.DataFrame()
    df_TgtIlist = pd.DataFrame()
    SrcUList = df_SeleSrc['uid'].unique()
    SrcIList = df_SeleSrc['iid'].unique()
    TgtUList = df_SeleTgt['uid'].unique()
    TgtIList = df_SeleTgt['iid'].unique()    
    df_SrcUlist['uid'] = SrcUList
    df_SrcIlist['iid'] = SrcIList
    # df_TgtUlist['uid'] = TgtUList
    df_TgtIlist['iid'] = TgtIList
    df_SrcUlist['u_list'] = map(GetCommentIndex_SU,SrcUList)
    df_SrcIlist['i_list'] = map(GetCommentIndex_SI,SrcIList)
    # df_TgtUlist['u_list'] = map(GetCommentIndex_TU,TgtUList)
    df_TgtIlist['i_list'] = map(GetCommentIndex_TI,TgtIList)
    df_SeleSrc = pd.merge(df_SeleSrc, df_SrcUlist, how='left', on=['uid'])
    df_SeleSrc = pd.merge(df_SeleSrc, df_SrcIlist, how='left', on=['iid'])
    df_SeleTgt = pd.merge(df_SeleTgt, df_SrcUlist, how='left', on=['uid'])
    df_SeleTgt = pd.merge(df_SeleTgt, df_TgtIlist, how='left', on=['iid'])
    SrcComment = outputpath + 'sentiRecOutput/%s.pk'%(source_name)
    src_Comment_Embedding = pkload(SrcComment)
    TgtComment = outputpath + 'sentiRecOutput/%s.pk'%(target_name)
    tgt_Comment_Embedding = pkload(TgtComment)
    
    def get_SrcW(key):
        return dic[key]
    srckey_list = zip(df_SeleSrc['uid'].to_list(),df_SeleSrc['iid'].to_list(),df_SeleSrc['time'].to_list())        
    tgtkey_list = zip(df_SeleTgt['uid'].to_list(),df_SeleTgt['iid'].to_list(),df_SeleTgt['time'].to_list()) 
    w = []
    w.append([0]*50)
    for key in srckey_list:
        w.append(src_Comment_Embedding[key])
    for key in tgtkey_list:
        w.append(tgt_Comment_Embedding[key])
    w = np.array(w)
    print('w.shape:',w.shape)   
    print('---------------------------------------------------over-----------------------------------------------------')
    print('')
    print('')
      
    out_path = outputpath+'sele_data/%s_%s/uThan%s_iThan%s/'%(source_name,target_name,u_less,i_less)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    np.save(out_path+'CommentArr.npy',w)
    df_SeleSrc.to_csv(out_path+'Src.csv',index=False)
    df_SeleTgt.to_csv(out_path+'Tgt.csv',index=False)

def form_train_test_for_LSCD_CoNet(df_src_train,df_src_test,df_tgt_train,df_tgt_test):
    '''
    form_train_test_for_CoNet
    '''
    a = list(df_src_train['uid'].unique())
    c = list(df_src_test['uid'].unique())
    b = list(df_tgt_train['uid'].unique())
    d = list(df_tgt_test['uid'].unique())
    a.sort()
    b.sort()
    c.sort()
    d.sort()
    print('len(a):%s, len(b):%s, if(a==b):%s'%(len(a),len(b),a==b))
    print('len(c):%s, len(d):%s, if(c==d):%s'%(len(c),len(d),c==d))
    
    le_uid = preprocessing.LabelEncoder()
    le_uid.fit(list(df_src_train['uid'].unique())+list(df_src_test['uid'].unique())+list(df_tgt_train['uid'].unique())+list(df_tgt_test['uid'].unique()))
    src_tr_uid_new = le_uid.transform(df_src_train['uid'].tolist())
    src_te_uid_new = le_uid.transform(df_src_test['uid'].tolist())
    tgt_tr_uid_new = le_uid.transform(df_tgt_train['uid'].tolist())
    tgt_te_uid_new = le_uid.transform(df_tgt_test['uid'].tolist())
    
    src_le_iid = preprocessing.LabelEncoder()
    src_le_iid.fit(list(df_src_train['iid'].unique())+list(df_src_test['iid'].unique()))
    src_tr_iid_new = src_le_iid.transform(df_src_train['iid'].tolist())
    src_te_iid_new = src_le_iid.transform(df_src_test['iid'].tolist())
    
    tgt_le_iid = preprocessing.LabelEncoder()
    tgt_le_iid.fit(list(df_tgt_train['iid'].unique())+list(df_tgt_test['iid'].unique()))
    tgt_tr_iid_new = tgt_le_iid.transform(df_tgt_train['iid'].tolist())
    tgt_te_iid_new = tgt_le_iid.transform(df_tgt_test['iid'].tolist())
 
    df_src_train['uid'] = src_tr_uid_new
    df_src_train['iid'] = src_tr_iid_new
    df_src_test['uid'] = src_te_uid_new
    df_src_test['iid'] = src_te_iid_new
    df_tgt_train['uid'] = tgt_tr_uid_new
    df_tgt_train['iid'] = tgt_tr_iid_new
    df_tgt_test['uid'] = tgt_te_uid_new
    df_tgt_test['iid'] = tgt_te_iid_new
    
    df_src_train_save = df_src_train.values
    df_src_test_save = df_src_test.values
    df_tgt_train_save = df_tgt_train.values
    df_tgt_test_save = df_tgt_test.values
    
    
    '''
    form_train_test_for_LSCD
    '''
    df_srcTrain = pd.DataFrame(df_src_train_save, columns = ['uid','iid','rating'])
    df_srcTest  = pd.DataFrame(df_src_test_save,  columns = ['uid','iid','rating'])
    df_tgtTrain = pd.DataFrame(df_tgt_train_save, columns = ['uid','iid','rating'])
    df_tgtTest  = pd.DataFrame(df_tgt_test_save,  columns = ['uid','iid','rating'])
    df_srcTrain['uid'] = df_srcTrain['uid'].values+1
    df_srcTrain['iid'] = df_srcTrain['iid'].values+1
    df_srcTest['uid'] = df_srcTest['uid'].values+1
    df_srcTest['iid'] = df_srcTest['iid'].values+1
    df_tgtTrain['uid'] = df_tgtTrain['uid'].values+1
    df_tgtTrain['iid'] = df_tgtTrain['iid'].values+1
    df_tgtTest['uid'] = df_tgtTest['uid'].values+1
    df_tgtTest['iid'] = df_tgtTest['iid'].values+1
    df_srcTrain['domain'] = 1
    df_srcTest['domain'] = 1
    df_tgtTrain['domain'] = 2
    df_tgtTest['domain'] = 2
    df_srcTrain = df_srcTrain.astype('uint16')
    df_srcTest = df_srcTest.astype('uint16')
    df_tgtTrain = df_tgtTrain.astype('uint16')
    df_tgtTest = df_tgtTest.astype('uint16')
    df_srcTrain['rating'] = df_srcTrain['rating'].astype('float64')
    df_srcTest['rating'] = df_srcTest['rating'].astype('float64')
    df_tgtTrain['rating'] = df_tgtTrain['rating'].astype('float64')
    df_tgtTest['rating'] = df_tgtTest['rating'].astype('float64')
    arr_srcTrain = df_srcTrain.values
    arr_srcTest = df_srcTest.values
    arr_tgtTrain = df_tgtTrain.values
    arr_tgtTest = df_tgtTest.values
    
    return df_src_train_save,df_src_test_save,df_tgt_train_save,df_tgt_test_save,arr_srcTrain,arr_srcTest,arr_tgtTrain,arr_tgtTest
    
def form_train_test(source_name,target_name,outputpath,u_less_list,i_less_list,fold):
    for i in range(len(u_less_list)):       
        u_less,i_less = u_less_list[i],i_less_list[i]
        print('u_less:%s, i_less:%s'%(u_less,i_less))
        print('start form data for our..................')
        out_path = outputpath+'sele_data/%s_%s/uThan%s_iThan%s/'%(source_name,target_name,u_less,i_less)
        df_src = pd.read_csv(out_path+'Src.csv')
        df_tgt = pd.read_csv(out_path+'Tgt.csv')
        uid_list = list(df_src['uid'].unique())
        
        uid_list = np.array(uid_list)
        kf_uid_list_tr, kf_uid_list_te = [],[]
        kf = KFold(n_splits=5, shuffle=False,random_state=2020)
        for train_index, test_index in kf.split(uid_list):
            kf_uid_list_tr.append(uid_list[train_index])
            kf_uid_list_te.append(uid_list[test_index])
        print('end form data for our..................')
        print('')
        
        for i in range(fold):
            df_src_train = df_src[df_src['uid'].isin(kf_uid_list_tr[i])]
            df_src_test = df_src[df_src['uid'].isin(kf_uid_list_te[i])]
            df_tgt_train = df_tgt[df_tgt['uid'].isin(kf_uid_list_tr[i])]
            df_tgt_test = df_tgt[df_tgt['uid'].isin(kf_uid_list_te[i])]
            df_src_train = df_src_train.sample(frac=1.0)
            df_src_train = df_src_train.reset_index(drop=True)
            df_src_test = df_src_test.sample(frac=1.0)
            df_src_test = df_src_test.reset_index(drop=True)
            df_tgt_train = df_tgt_train.sample(frac=1.0)
            df_tgt_train = df_tgt_train.reset_index(drop=True)
            df_tgt_test = df_tgt_test.sample(frac=1.0)
            df_tgt_test = df_tgt_test.reset_index(drop=True)
            save_path = out_path+'fold_%s/'%i
            
            a = list(df_src_train['uid'].unique())
            a.sort()
            b = list(df_tgt_train['uid'].unique())
            b.sort()
            c = list(df_src_test['uid'].unique())
            c.sort()
            d = list(df_tgt_test['uid'].unique())
            d.sort()
            print('if train user in src and tgt is same:',a==b)
            print('if test user in src and tgt is same:',c==d)
    
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df_src_train.to_csv(save_path+'src_train.csv',index=False)
            df_src_test.to_csv(save_path+'src_test.csv',index=False)
            df_tgt_train.to_csv(save_path+'tgt_train.csv',index=False)
            df_tgt_test.to_csv(save_path+'tgt_test.csv',index=False)
            
            
            print('start form data for mf..................')
            save_path_mf = outputpath+'/sele_data/use_for_mf0107/%s_%s/uThan%s_iThan%s/fold_%s/'%(source_name,target_name,u_less,i_less,i)
            if not os.path.exists(save_path_mf):
                os.makedirs(save_path_mf)
            df_src_train[['uid','iid','rating']].to_csv(save_path_mf+'reviews_%s_5_train.csv'%source_name,index=False,header=False)
            df_src_test[['uid','iid','rating']].to_csv(save_path_mf+'reviews_%s_5_test.csv'%source_name,index=False,header=False)
            df_tgt_train[['uid','iid','rating']].to_csv(save_path_mf+'reviews_%s_5_train.csv'%target_name,index=False,header=False)
            df_tgt_test[['uid','iid','rating']].to_csv(save_path_mf+'reviews_%s_5_test.csv'%target_name,index=False,header=False)
            print('end form data for mf..................')
            print('')
            
            print('start form data for CoNet and LSCD.................')
            save_path_CoNet = outputpath+'/sele_data/use_for_CoNet0123/%s_%s/uThan%s_iThan%s/fold_%s/'%(source_name,target_name,u_less,i_less,i)
            save_path_LSCD = outputpath+'/sele_data/use_for_LSCD0115/mat_file/%s_%s_uThan%s_iThan%s_fold%s/'%(source_name,target_name,u_less,i_less,i)
            if not os.path.exists(save_path_CoNet):
                os.makedirs(save_path_CoNet)
            if not os.path.exists(save_path_LSCD):
                os.makedirs(save_path_LSCD)
            df_src_train = df_src_train[['uid','iid','rating']]
            df_src_test = df_src_test[['uid','iid','rating']]
            df_tgt_train = df_tgt_train[['uid','iid','rating']]
            df_tgt_test = df_tgt_test[['uid','iid','rating']]
            df_src_train_save,df_src_test_save,df_tgt_train_save,df_tgt_test_save,arr_srcTrain,arr_srcTest,arr_tgtTrain,arr_tgtTest = form_train_test_for_LSCD_CoNet(df_src_train,df_src_test,df_tgt_train,df_tgt_test)
            np.savetxt(save_path_CoNet+'/%s_train.txt'%(source_name),df_src_train_save, delimiter=",")
            np.savetxt(save_path_CoNet+'/%s_test.txt'%(source_name),df_src_test_save, delimiter=",")
            np.savetxt(save_path_CoNet+'/%s_train.txt'%(target_name),df_tgt_train_save, delimiter=",")
            np.savetxt(save_path_CoNet+'/%s_test.txt'%(target_name),df_tgt_test_save, delimiter=",")
            io.savemat(save_path_LSCD+'/src_train.mat', {'src_train': arr_srcTrain})
            io.savemat(save_path_LSCD+'/src_test.mat', {'src_test': arr_srcTest})
            io.savemat(save_path_LSCD+'/tgt_train.mat', {'tgt_train': arr_tgtTrain})
            io.savemat(save_path_LSCD+'/tgt_test.mat', {'tgt_test': arr_tgtTest})
            print('end form data for CoNet and LSCD..................')
            print('')
            
            print('fold %s, len(src_tr):%s, len(src_te):%s, len(tgt_tr):%s, len(tgt_te):%s'%(i,len(df_src_train),len(df_src_test),len(df_tgt_train),len(df_tgt_test)))
        print('')
        

    

  
    
@cli.command()
@click.option("--source_name", default="Musical_Instruments", help=u"源领域名称")
@click.option("--target_name", default="Automotive", help=u"目标领域名称")
@click.option("--outputpath", default="/data1/home/jinyaru/DSNRec_1024/exam/preprocess/", help=u"数据存储路径")
@click.option("--u_less_list", default='[15,20]', help=u"user至少多少条评论才保留")
@click.option("--i_less_list", default='[15,20]', help=u"item至少多少条评论才保留")
@click.option("--if_time", default=True, help=u"是否按时序排列评论")
@click.option("--fold", default=5, help=u"交叉验证的折数")
# print('source_name',source_name,'target_name:',target_name)

def selec_data(source_name,target_name,outputpath,u_less_list,i_less_list,if_time,fold): 
    print('---------------------------------------------------start-----------------------------------------------------')
    df_Src_over,df_Tgt_over = about_overlap(source_name,target_name,outputpath)
    u_less_list,i_less_list = map(int,eval(u_less_list)),map(int,eval(i_less_list))
    for i in range(len(u_less_list)):
        u_less,i_less = u_less_list[i],i_less_list[i]
        df_SeleSrc,df_SeleTgt = selectDf(u_less,i_less,df_Src_over,df_Tgt_over)
        save_w_src_tgt(df_SeleSrc,df_SeleTgt,u_less,i_less,if_time,outputpath,source_name,target_name)
        form_train_test(source_name,target_name,outputpath,u_less_list,i_less_list,fold)
    print('---------------------------------------------------over-----------------------------------------------------')
selec_data()   