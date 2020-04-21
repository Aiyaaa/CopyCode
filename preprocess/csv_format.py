# -*- coding: utf8 -*-

import os
import sys
import click

from itertools import chain
from operator import itemgetter
from glob import glob

import sys
sys.path.append(os.path.abspath('/data1/home/jinyaru/DSNRec_1103')) 
from preprocess.utils import readJson
from utils import recordTime, pkload
import config


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data_dir", default="/data1/home/jinyaru/DSNRec_1024/exam/", help=u"数据路径")
@click.option("--fields", default="*", help=u"逗号分割, 指定输出字段")
@click.option("--output_dir", default="csv_format", help=u"输出路径")
@click.option("--src_domain", default="Kindle", help=u"源领域名称")
@click.option("--tgt_domain", default="Automotive", help=u"目标领域名称")

def csv_form(data_dir, fields, output_dir, src_domain, tgt_domain):
    if fields == "*":
        fields = "reviewerID,asin,overall"

    getter = itemgetter(*fields.split(","))

    transform = "%s/preprocess/transform" % data_dir
    output_dir = "%s/preprocess/%s/%s_%s/" % (data_dir, output_dir, src_domain, tgt_domain)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cold_dir = "%s/preprocess/cold/%s_%s/overlapUser.pk" % (data_dir,src_domain,tgt_domain)
    cold_user = pkload(cold_dir)

    need_transform = ['reviews_%s_5.json'%src_domain,'reviews_%s_5.json'%tgt_domain]
    for fn in need_transform:
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
                if d[0] in cold_user: 
                    test.append(",".join(map(str, d)))
                else:
                    train.append(",".join(map(str, d)))

            with open(out_train, "w") as f:
                f.write("\n".join(train))

            with open(out_test, "w") as f:
                f.write("\n".join(test))

        transCSV(fn)
csv_form()
