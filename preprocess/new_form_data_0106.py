# -*- coding: utf-8 -*-
#算的是交叠的dataframe里 user,item出现的次数
import pandas as pd

import os
import sys
import click
from itertools import chain
from operator import itemgetter
from glob import glob
import numpy as np
import pandas as pd

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
  
    
@cli.command()
@click.option("--source_name", default="Musical_Instruments", help=u"源领域名称")
@click.option("--target_name", default="Automotive", help=u"目标领域名称")
@click.option("--outputpath", default="/data1/home/jinyaru/DSNRec_1103/preprocess/", help=u"数据存储路径")
@click.option("--u_less_list", default='[15,20]', help=u"user至少多少条评论才保留")
@click.option("--i_less_list", default='[15,20]', help=u"item至少多少条评论才保留")
@click.option("--if_time", default=True, help=u"是否按时序排列评论")
# print('source_name',source_name,'target_name:',target_name)

def selec_data(source_name,target_name,outputpath,u_less_list,i_less_list,if_time): 
    print('---------------------------------------------------start-----------------------------------------------------')
    df_Src_over,df_Tgt_over = about_overlap(source_name,target_name,outputpath)
    u_less_list,i_less_list = map(int,eval(u_less_list)),map(int,eval(i_less_list))
    for i in range(len(u_less_list)):
        u_less,i_less = u_less_list[i],i_less_list[i]
        df_SeleSrc,df_SeleTgt = selectDf(u_less,i_less,df_Src_over,df_Tgt_over)
        save_w_src_tgt(df_SeleSrc,df_SeleTgt,u_less,i_less,if_time,outputpath,source_name,target_name)
selec_data()   