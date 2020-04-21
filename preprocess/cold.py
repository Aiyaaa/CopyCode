# -*- coding: utf-8 -*-

import pandas as pd

import os
import sys
import click
from itertools import chain
from operator import itemgetter
from glob import glob

sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024')) 
from preprocess.utils import readJson
from utils import pkload, pkdump

@click.group()
def cli():
    pass


# print('source_name',source_name,'target_name:',target_name)
def generateColdUser(sourceDomain, targetDomain):
    '''生成冷用户

    source domain 冷用户定义如下:
        在target domain中有记录而在source domain中没有记录的用户
    target domain 冷用户定义类似

    Parameters
    ----------
    sourceDomain : str
        sourceDomain json数据文件路径

    targetDomain : str
        targetDomain json数据文件路径

    Returns
    -------
        coldUserSource : pd.DataFrame
    '''

    Source = list(readJson(sourceDomain))
    SourceUserItem = [(d["reviewerID"], d["asin"]) for d in Source]
    del Source
    print('load %s'%sourceDomain)
    Traget = list(readJson(targetDomain))
    TargetUserItem = [(d["reviewerID"], d["asin"]) for d in Traget]
    del Traget
    print('load %s'%targetDomain)
    uiS = pd.DataFrame(data=SourceUserItem, columns=["user", "sourceItem"])
    uiT = pd.DataFrame(data=TargetUserItem, columns=["user", "targetItem"])
    uS = uiS.groupby("user").count()
    uT = uiT.groupby("user").count()
    uBoth = pd.concat([uS, uT], axis=1).fillna(0)
    # coldUserSource 的user 来自于 Target domain, 在Source Domain中记录数量为0
    coldUserSource = uBoth.query("sourceItem == 0")
    # coldUserTarget 的user 来自于 Source Domain, 在Target Domain 中记录数量为0
    coldUserTarget = uBoth.query("targetItem == 0")
    overlapUser = uBoth.query("sourceItem != 0 and targetItem != 0")
    return list(coldUserTarget.index), list(coldUserSource.index), list(overlapUser.index)

@cli.command()
@click.option("--source_name", default="Musical_Instruments", help=u"源领域名称")
@click.option("--target_name", default="Automotive", help=u"目标领域名称")
@click.option("--outputpath", default="/data1/home/jinyaru/DSNRec_1024/exam/preprocess/", help=u"冷用户信息的输出路径")
def getColdUser(source_name,target_name,outputpath):
    json_path = os.path.join(outputpath, "transform/")  
    SOURCE = json_path+'reviews_%s_5.json'%source_name
    TARGET = json_path+'reviews_%s_5.json'%target_name
    ColdSU, ColdTU, OverLapU = generateColdUser(SOURCE, TARGET)      
#     ColdSU = [user[u] for u in ColdSU]
#     ColdTU = [user[u] for u in ColdTU]
#     OverLapU = [user[u] for u in OverLapU]
    print("cold user count in %s is %d"%(source_name,len(ColdSU)))
    print("cold user count in %s is %d"%(target_name,len(ColdTU)))
    print("%s and %s, overlap user count is %d"%(source_name,target_name, len(OverLapU)))

    ColdOutputPath = os.path.join(outputpath, "cold/%s_%s"%(source_name,target_name))
    if not os.path.exists(ColdOutputPath):
        os.mkdir(ColdOutputPath)

    pkdump(ColdSU, os.path.join(ColdOutputPath, "%s_ColdUser.pk"%(source_name)))
    pkdump(ColdTU, os.path.join(ColdOutputPath, "%s_ColdUser.pk"%(target_name)))
    pkdump(OverLapU, os.path.join(ColdOutputPath, "overlapUser.pk"))
    

getColdUser()
