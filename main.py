# -*- coding: utf8 -*-
#python preprocess generatevoca --mode DEBUG --data_dir exam/ 


import os
import sys
import click
from itertools import chain
from operator import itemgetter
from glob import glob
import sys
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1024')) 
from preprocess.utils import buildVoca, transReview
from preprocess.utils import readJson
from utils import pkdump, recordTime, pkload
import config
from preprocess.cold import generateColdUser
from preprocess.sentiOutputMergeUserItem import mergeUserItem



def csv_form(mode, data_dir, fields, output_dir):
    runConfig = config.configs[mode]("csv_format_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")

    if fields == "*":
        fields = "reviewerID,asin,overall"

    getter = itemgetter(*fields.split(","))

    transform = "%s/preprocess/transform" % data_dir
    output_dir = "%s/preprocess/%s" % (data_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cold_dir = "%s/preprocess/cold/overlapUser.pk" % data_dir
    cold_user = pkload(cold_dir)

    for fn in os.listdir(transform):
        @recordTime
        def transCSV(fn):
            inf = os.path.join(transform, fn)
            ext = os.path.splitext(fn)[-1]
            out_train = os.path.join(output_dir, fn).replace(ext, "_train.csv")
            out_test = os.path.join(output_dir, fn).replace(ext, "_test.csv")
            data = [getter(d) for d in readJson(inf)]
            train, test = [], []
            user_dic = {}
            for d in data:
                ###
                if d[0] in user_dic:
                    user_dic[d[0]]=user_dic[d[0]]+1
                else:
                    user_dic[d[0]]=1
                ###
                if (d[0] in cold_user) & (user_dic[d[0]]>4):
                    test.append(",".join(map(str, d)))
                else:
                    train.append(",".join(map(str, d)))
            
            with open(out_train, "w") as f:
                f.write("\n".join(train))

            with open(out_test, "w") as f:
                f.write("\n".join(test))

        transCSV(fn)


def generateVoca(mode, sub_output_path, fields, data_dir):
    u'''提取所有json文件的词典, 并将所有的review转换成为字典

    json文件的查找路径为当前路径之下的data/source

    词典输出路径为: data/preprocess/${sub_output_path}

    reviewText转换的文件输出在data/preprocess/transform/

    fields为一个逗号分隔的list,合法值如下:
        asin,helpful,overall,reviewText,
        reviewTime,reviewerID,reviewerName,
        summary,unixReviewTime
    选项不提供默认全部保留
    e.g --fields asin,reviewerID,overall,reviewText

    '''
    runConfig = config.configs[mode]("generateVoca_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    runConfig.setSourcePath(data_dir)
    runConfig.checkValid()
    logger = runConfig.getLogger()
    logger.info("the command line is %s", " ".join(sys.argv))
    inputPath = runConfig.PREPROCESS_CONFIG["source_path"]
    outputPath = runConfig.PREPROCESS_CONFIG["output_path"]

    '''从source path下所有json文件中提取出词典'''
    vocas = []
    users = []
    items = []
    if not os.path.exists(os.path.join(outputPath, sub_output_path)):
        os.mkdir(os.path.join(outputPath, sub_output_path))

    for filename in os.listdir(inputPath):
        @recordTime
        def buildMapper(filename):
            voca, user, item = buildVoca(os.path.join(inputPath, filename))
            vocas.append(voca)
            users.append(user)
            items.append(item)
            outputDir = os.path.join(outputPath, sub_output_path)
            fnPrefix = filename.rsplit(".", 1)[0]
            for suffix, data in zip((".pk", "_user.pk", "_item.pk"),
                                    (voca, user, item)):
                outputName = os.path.join(outputDir, fnPrefix + suffix)
                pkdump(data, outputName)

        buildMapper(filename)

    mappers = []
    for suffix, data in zip((".pk", "_user.pk", "_item.pk"),
                            (vocas, users, items)):
        dumpData = set()
        for d in data:
            dumpData.update(set(d.values()))
        dumpData = {i: content for i, content in enumerate(dumpData)}
        pkdump(dumpData, os.path.join(outputPath, "vocab", "allDomain"+suffix))
        mappers.append(dumpData)

    '''利用词典将原始文本得reviewText转换成文id'''
    transOutputPath = os.path.join(outputPath, "transform")
    if not os.path.exists(transOutputPath):
        os.mkdir(transOutputPath)

    voca, user, item = mappers
    voca = {word: i for i, word in voca.items()}
    user = {u: i for i, u in user.items()}
    item = {it: i for i, it in item.items()}
    mapper = {"vocab": voca, "user": user, "item": item}
    fields = None if fields == "" else fields.split(",")
    for filename in os.listdir(inputPath):
        @recordTime
        def trans(filename):
            transReview(os.path.join(inputPath, filename),
                        mapper, os.path.join(transOutputPath, filename),
                        fields)
        trans(filename)

    @recordTime
    def getColdUser():
        files = [os.path.join(inputPath, f) for f in os.listdir(inputPath)]
        SOURCE, TARGET = files[:2]
        ColdSU, ColdTU, OverLapU = generateColdUser(SOURCE, TARGET)
        ColdSU = [user[u] for u in ColdSU]
        ColdTU = [user[u] for u in ColdTU]
        OverLapU = [user[u] for u in OverLapU]
        logger.info("cold user count in %s is %d",
                    os.path.basename(SOURCE),
                    len(ColdSU))
        logger.info("cold user count in %s is %d",
                    os.path.basename(TARGET),
                    len(ColdTU))
        logger.info("overlap user count is %d", len(OverLapU))

        ColdOutputPath = os.path.join(outputPath, "cold")
        if not os.path.exists(ColdOutputPath):
            os.mkdir(ColdOutputPath)

        ColdSO = os.path.join(ColdOutputPath, os.path.basename(SOURCE))
        ColdTO = os.path.join(ColdOutputPath, os.path.basename(TARGET))
        pkdump(ColdSU, ColdSO.rsplit(".", 1)[0]+".pk")
        pkdump(ColdTU, ColdTO.rsplit(".", 1)[0]+".pk")
        pkdump(OverLapU, os.path.join(ColdOutputPath, "overlapUser.pk"))

    #getColdUser()





def mergeUI(mode, data_dir, domain, output_dir):
    runConfig = config.configs[mode]("mergeUI_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    pattern = "%s/preprocess/sentiRecOutput/%s"

    patterns = [glob(pattern % (data_dir, d)) for d in domain.split(",")]

    out_dir = "%s/preprocess/%s" % (data_dir, output_dir)
    for fn in chain(*patterns):
        logger.debug("Dealing with %s", fn)
        user, item = mergeUserItem(fn)
        bn = os.path.basename(fn)
        pkdump(user, os.path.join(out_dir, "user_"+bn))
        pkdump(item, os.path.join(out_dir, "item_"+bn))
        logger.debug("Dealing end  %s", fn)

    rating = "%s/preprocess/transform/*.json"
    key_getter = itemgetter("reviewerID", "asin")
    value_getter = itemgetter("overall", "unixReviewTime")
    for fn in glob(rating % data_dir):
        data = {key_getter(d): value_getter(d) for d in readJson(fn)}
        out_name = os.path.join(out_dir,
                                os.path.basename(os.path.splitext(fn)[0])+".pk")
        out_name = out_name.replace("reviews", "rating_time")
        pkdump(data, out_name)



        

def extractInfo(mode, data_dir):
    u'''从转换后的用户数据提取信息'''
    runConfig = config.configs[mode]("extractInfo_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    runConfig.setSourcePath(data_dir)
    runConfig.checkValid()
    logger = runConfig.getLogger()
    logger.info("the command line is %s", " ".join(sys.argv))

