# -*- coding: utf-8 -*-
'''
把src_df和tgt_df按user分成train和test
分成5折试试
'''

import pandas as pd

import os
import sys
import click
from itertools import chain
from operator import itemgetter
from glob import glob
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold


@click.group()
def cli():
    pass
   
@cli.command()
@click.option("--source_name", default="Movies_and_TV", help=u"源领域名称")
@click.option("--target_name", default="Electronics", help=u"目标领域名称")
@click.option("--outputpath", default="/data1/home/jinyaru/DSNRec_1103/preprocess/", help=u"数据存储路径")
@click.option("--u_less_list", default='[20,30,40]', help=u"user至少多少条评论才保留")
@click.option("--i_less_list", default='[20,30,40]', help=u"item至少多少条评论才保留")
@click.option("--fold", default=5, help=u"交叉验证的折数")

def form_train_test(source_name,target_name,outputpath,u_less_list,i_less_list,fold):
    u_less_list,i_less_list = map(int,eval(u_less_list)),map(int,eval(i_less_list))
    for i in range(len(u_less_list)):       
        u_less,i_less = u_less_list[i],i_less_list[i]
        print('u_less:%s, i_less:%s'%(u_less,i_less))
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
            
            df_src_train[['uid','iid','rating']].to_csv(save_path+'reviews_%s_5_train.csv'%source_name,index=False,header=False)
            df_src_test[['uid','iid','rating']].to_csv(save_path+'reviews_%s_5_test.csv'%source_name,index=False,header=False)
            df_tgt_train[['uid','iid','rating']].to_csv(save_path+'reviews_%s_5_train.csv'%target_name,index=False,header=False)
            df_tgt_test[['uid','iid','rating']].to_csv(save_path+'reviews_%s_5_test.csv'%target_name,index=False,header=False)
            print('fold %s, len(src_tr):%s, len(src_te):%s, len(tgt_tr):%s, len(tgt_te):%s'%(i,len(df_src_train),len(df_src_test),len(df_tgt_train),len(df_tgt_test)))
        print('')
form_train_test()         