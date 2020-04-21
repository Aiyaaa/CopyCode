# -*- coding: utf8 -*-
import os
import sys
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024/')) 
from utils import *
import click
import numpy as np
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from glob import glob

from preprocess.utils import buildVoca, transReview
from preprocess.utils import readJson
import config

def mergeUserItem(filename,num_comment,start_index):
    u'''作用：从json文件中提取出user,item,评论三个词典, 

    参数：
    filename为json文件
    num_comment为每个用户和item评论截断的阈值
    start_index为该领域评论编号的起始值

    函数输出：
        三个词典user, item, sent_vec_dic
        user: e.g --user[u_index]=u_comm_list
              key为user的编号，value为该user按时间先后的评论，截断过的
        item: e.g --item[i_index]=i_comm_list
              key为item的编号，value为该item按时间先后的评论，截断过的
        sent_vec_dic: e.g --sent_vec_dic[c_index]=comm_arr
              key为评论的编号，value为该编号对应的评论向量
    
    '''
    data = pkload(filename)
    print('load %s'%filename)
    user = defaultdict(list)
    item = defaultdict(list)
    user_time = defaultdict(list)
    item_time = defaultdict(list)
    sent_vec_dic = {}
    index = start_index
    

    all_data_items=sorted(data.items(),key=lambda x:x[0][2],reverse=True)

    #for (u, i, time), vector in data.items():
    for (u, i, time), vector in all_data_items:
        if len(user[u])<num_comment or len(item[i])<num_comment:
            user_time[u].insert(0,time) #user_time[u].append(time)
            item_time[i].insert(0,time)#item_time[i].append(time)
            user[u].insert(0,index) #user[u].append(index)
            item[i].insert(0,index) #item[i].append(index)
            #print(type(vector),type(vector[0])) #(<type 'numpy.ndarray'>, <type 'numpy.float32'>)
            sent_vec_dic[index] = list(vector)
            index = index+1
        else:
            #print('被截断了')
            nouse_num = 1
    del data
    print('del data')
    
    user = dict(user)
    item = dict(item)
    user_time = dict(user_time)
    item_time = dict(item_time)
    def get_num_of_comment(arr):
        return len(arr)
    u_len = list(map(get_num_of_comment,list(user.values())))
    u_len.sort()
    bn = os.path.basename(filename)
    bn = bn[:-3]
    #print('domain:',bn,'评论数:%s'%len(sent_vec_dic))
    print('domain:',bn)
    print('num of samples:%s'%len(sent_vec_dic))
    print('num of user:%s'%len(u_len),'mean comments:%s'%round(np.mean(np.array(u_len)),0),'min comments:%s'%(u_len[0]),'most comments:%s'%(u_len[-1]))
    # domain:Musical_Instruments ('num of user:', 1429, 'mean comments:', 7.1805458362491255, [5, 5], [38, 42])
    i_len = list(map(get_num_of_comment,list(item.values())))
    i_len.sort()
    print('num of item:%s'%len(i_len),'mean comments:%s'%round(np.mean(np.array(i_len)),0),'min comments:%s'%(i_len[0]),'most comments:%s'%(i_len[-1]))
    # domain:Musical_Instruments ('num of item:', 900, 'mean comments:', 11.401111111111112, [5, 5], [143, 163])


    ### 以下按时间把评论排序
    def reget_u_comIndexList(u):
        u_index = user[u]
        u_time = user_time[u] 
        tmp=[(time, index) for time, index in zip(u_time,u_index)] #先转化成元组
        tmp.sort() #按照分数排序
        index_list = [index for time, index in tmp] #将排好序的分数姓名的元组分开
        if len(index_list)>num_comment:
            index_list=index_list[-num_comment:]
        return index_list
    def reget_i_comIndexList(i):
        i_index = item[i]
        i_time = item_time[i] 
        tmp=[(time, index) for time, index in zip(i_time,i_index)] #先转化成元组
        tmp.sort() #按照分数排序
        index_list = [index for time, index in tmp] #将排好序的分数姓名的元组分开
        if len(index_list)>num_comment:
            index_list=index_list[-num_comment:]
        return index_list
    u_index_list = list(user.keys())
    u_commentList_list = list(map(reget_u_comIndexList,u_index_list))
    user = dict(zip(u_index_list,u_commentList_list))
    i_index_list = list(item.keys())
    i_commentList_list = list(map(reget_i_comIndexList,i_index_list))
    item = dict(zip(i_index_list,i_commentList_list))
    """
    for u in user:
        u_index = user[u]
        u_time = user_time[u] 
        tmp=[(time, index) for time, index in zip(u_time,u_index)] #先转化成元组
        tmp.sort() #按照分数排序
        index_list = [index for time, index in tmp] #将排好序的分数姓名的元组分开
        if len(index_list)>num_comment:
            index_list=index_list[-num_comment:]
#         elif len(index_list)<num_comment:
#             index_list = [0]*(num_comment-len(index_list)) + index_list
        user[u] = index_list
    for i in item:
        i_index = item[i]
        i_time = item_time[i] 
        tmp=[(time, index) for time, index in zip(i_time,i_index)] #先转化成元组
        tmp.sort() #按照分数排序
        index_list = [index for time, index in tmp] #将排好序的分数姓名的元组分开
        if len(index_list)>num_comment:
            index_list=index_list[-num_comment:]
#         elif len(index_list)<num_comment:
#             index_list = [0]*(num_comment-len(index_list)) + index_list
        item[i] = index_list
    """
    return user, item, sent_vec_dic



@click.group()
def cli():
    pass

@cli.command()
@click.option("--mode",default="DEBUG", type=click.Choice(["DEBUG", "DEFAULT", "DEVELOP"]))
@click.option("--data_dir", default="/data1/home/jinyaru/DSNRec_1024/exam/", help=u"输入数据文件夹路径")
@click.option("--source_domain", default="Musical_Instruments", help=u"源领域")
@click.option("--target_domain", default="Automotive", help=u"目标领域")
@click.option("--output_dir", default="uirepresent", help=u"输出路径")
@click.option("--num_comment", default=500, help=u"截断评论数")
def mergeUI(mode,data_dir,source_domain,target_domain, output_dir,num_comment):
    u'''作用：为source_domain和target_domain生成可用来跨领域推荐的词典并存储 

    参数：
    source_domain：源领域名称
    target_domain：目标领域名称
    output_dir：词典的存储位置
    num_comment：为每个用户和item评论截断的阈值
    
    '''
   
    runConfig = config.configs[mode]("mergeUI_")
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    
    #pattern = "%s/preprocess/sentiRecOutput/%s.pk"
    out_dir = "%s/preprocess/%s/source_%s/%s_%s"%(data_dir,output_dir,source_domain,source_domain,target_domain)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 
    patterns=[]
    if source_domain==target_domain:
        patterns.append("%s/preprocess/sentiRecOutput/%s.pk"%(data_dir,source_domain))
        start_index = 1
    else:
        patterns.append("%s/preprocess/sentiRecOutput/%s.pk"%(data_dir,target_domain))
        if source_domain=='Books':
            start_index = 1+8898041
        elif source_domain=='Tools_and_Home_Improvement':
            start_index = 1+134476
        elif source_domain=='Video_Games':
            start_index = 1+231780
        elif source_domain=='Automotive':
            start_index = 1+20473
        elif source_domain=='Kindle':
            start_index = 1+982619
        elif source_domain=='Musical_Instruments':
            start_index = 1+10261
        elif source_domain=='Toys_and_Games':
            start_index = 1+167597
        elif source_domain=='Beauty':
            start_index = 1+198502
        elif source_domain=='Cell_Phones_and_Accessories':
            start_index = 1+194439
        elif source_domain=='Clothing_Shoes_and_Jewelry':
            start_index = 1+278677
        elif source_domain=='Digital_Music':
            start_index = 1+64706
        elif source_domain=='Grocery_and_Gourmet_Food':
            start_index = 1+151254
        elif source_domain=='Office_Products':
            start_index = 1+53258
        elif source_domain=='Sports_and_Outdoors':
            start_index = 1+296337
    print('target_domain:',target_domain,'start_index',start_index)
    for fn in patterns:
        logger.debug("Dealing with %s", fn)
        user, item, sent_vec_dic = mergeUserItem(fn,num_comment,start_index)
#         print('start_index:',start_index)
#         print('end_index:',start_index+len(sent_vec_dic)-1)
        start_index = start_index+len(sent_vec_dic)
        bn = os.path.basename(fn)
        #print("os.path.join(out_dir, 'comment_'+bn):",os.path.join(out_dir, 'comment_'+bn))
        #print('os.path.join(out_dir, "item_"+bn):',os.path.join(out_dir, "item_"+bn))
        #("os.path.join(out_dir, 'comment_'+bn):", '/preprocess/uirepresent/comment_Musical_Instruments.pk')
        #('os.path.join(out_dir, "item_"+bn):', '/preprocess/uirepresent/item_Musical_Instruments.pk')
        pkdump(user, os.path.join(out_dir, "user_"+bn))
        del user
        pkdump(item, os.path.join(out_dir, "item_"+bn))
        del item
        pkdump(sent_vec_dic, os.path.join(out_dir, 'comment_'+bn))
        logger.debug("Dealing end  %s", fn)
        del sent_vec_dic
    print('dump over')
    #rating = "%s/preprocess/transform/*.json"
    key_getter = itemgetter("reviewerID", "asin")
    value_getter = itemgetter("overall", "unixReviewTime")
    rating_patterns = []
    for d in [target_domain]:##
        rating_patterns.append("%s/preprocess/transform/reviews_%s_5.json"%(data_dir,d))
    for fn in rating_patterns:        
        out_name = os.path.join("%s/preprocess/transform/"%data_dir,
                                os.path.basename(os.path.splitext(fn)[0])+".pk")
        out_name = out_name.replace("reviews", "rating_time")        
        if not os.path.exists(out_name): 
            print('save file:',out_name)
            data = {key_getter(d): value_getter(d) for d in readJson(fn)}
            pkdump(data, out_name)
            del data
mergeUI()
