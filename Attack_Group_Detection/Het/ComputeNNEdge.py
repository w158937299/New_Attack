import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import Parser
import Bulidata
import time
from tqdm import tqdm
import getData
import networkx
import datetime
import dateutil
import math
import scipy.stats
from dateutil.relativedelta import relativedelta


def date_minus_month(gk_date):
    d = datetime.datetime.strptime(str(gk_date), "%Y-%m-%d")
    date_mm = d + relativedelta(months=1)
    date_mm = datetime.datetime.strftime(date_mm, "%Y-%m-%d")
    return date_mm

def date_compare(date1, date2, fmt='%Y-%m-%d') -> bool:
    """
    比较两个真实日期之间的大小，date1 > date2 则返回True
    :param date1:
    :param date2:
    :param fmt:
    :return:
    """

    d1 = datetime.datetime.strptime(str(date1), fmt)

    d2 = datetime.datetime.strptime(str(date2), fmt)

    return d1 > d2

if __name__ == '__main__':
    args = Parser.Define_Params()
    file_path = '.' + args.file_path
    save_uuweight_path = args.UUedge_weight
    UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
    startTime = min(Times)
    endTime = max(Times)
    TimeSeque = []
    startTime = str(startTime)[0:10]
    endTime = str(endTime)[0:10]

    while startTime <= endTime:
        TimeSeque.append(startTime)
        startTime = date_minus_month(startTime)
    TimeSeque.append(endTime)

    ttindex = {}
    for i in range(len(TimeSeque) - 1):
        index = 't' + str(TimeSeque[i]) + 't' + str(TimeSeque[i + 1])
        ttindex.setdefault(index, 0)

    ttratindex = {}
    for i in range(len(TimeSeque) - 1):
        index = 't' + str(TimeSeque[i]) + 't' + str(TimeSeque[i + 1])
        ratindex = {}
        for i in range(5):
            ratindex.setdefault(i + 1, 0)
        # ttratindex.setdefault(index, ratindex)
        ttratindex[index] = ratindex


    itemRTDD = []
    itemURTI = []
    finish = 0
    for item in ItemToUser.keys():
        suser = len(ItemToUser[item])
        finishlen = len(ItemToUser.keys())
        if finish % 100 == 0:
            print('the all finishlen is %d , now run the %dth'%(finishlen,finish))

        for key in ttindex:
            ttindex[key] = 0
        for key1 in ttratindex:
            for value in ttratindex[key1]:
                ttratindex[key1][value] = 0

        for user in ItemToUser[item]:
            tempidx = 'u' + str(user) + 'i' + str(item)
            now_time = newUItoRT[tempidx][1]
            now_rat = newUItoRT[tempidx][0]
            for i in range(1, len(TimeSeque)):
                if date_compare(TimeSeque[i], now_time):
                    ttidx = 't' + str(TimeSeque[i - 1]) + 't' + str(TimeSeque[i])
                    count = ttindex[ttidx]
                    count = count + 1
                    ttindex[ttidx] = count

                    count1 = ttratindex[ttidx][int(now_rat)]
                    count1 = count1 + 1
                    ttratindex[ttidx][int(now_rat)] = count1
                    break



        itemRTDD.append(ttindex)
        itemURTI.append(ttratindex)
        finish = finish + 1

    next_RTDD = {}
    item_next_RTDD = []
    for itemindex in range(len(itemRTDD)):
        all = 0
        for tt in itemRTDD[itemindex]:
            all = all + itemRTDD[itemindex][tt]
        for tt in itemRTDD[itemindex]:
            temp = itemRTDD[itemindex][tt]/all
            next_RTDD.setdefault(tt, temp)
        item_next_RTDD.append(next_RTDD)

    next_URII = {}
    item_next_URII = []
    for itemindex in range(len(itemURTI)):
        # all1:所有的时间间隔
        # whole1:一段时间间隔内的全部数目
        all1 = 0
        whole1 = {}
        for tt in itemURTI[itemindex]:
            temp = 0
            for rat in itemURTI[itemindex][tt]:
                tempcount = itemURTI[itemindex][tt][rat]
                all1 = all1 + tempcount
                temp = temp + tempcount
            whole1.setdefault(tt, temp)

        for tt in itemURTI[itemindex]:
            # 单一时间间隔的比例
            ratio = [0,0,0,0,0]
            # 所有时间内的比例
            o_ratio = [0,0,0,0,0]
            for rat in itemURTI[itemindex][tt]:
                ratio[int(rat) - 1] = itemURTI[itemindex][tt][rat] / whole1[tt]
                o_ratio[int(rat) - 1] = itemURTI[itemindex][tt][rat] / all1
            begin_P1 = 0
            P2 = 0
            for i in range(len(ratio)):
                if ratio[i] != 0:
                    begin_P1 = begin_P1 + ratio[i] * math.log2(ratio[i])
                    P2 = P2 + math.pow((ratio[i] - o_ratio[i]), 2)/o_ratio[i]
            P1 = 1/(1 - begin_P1)
            last = P1 * P2
            next_URII.setdefault(tt, last)
        item_next_URII.append(next_URII)
    uuedge = {}
    for i in range(len(item_next_URII)):
        temp = []
        for tt in item_next_URII[i]:
            des = item_next_URII[i][tt] * item_next_RTDD[i][tt]
            temp.append(des)
      


    print(item_next_RTDD)



























