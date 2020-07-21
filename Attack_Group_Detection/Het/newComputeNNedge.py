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
import sys



def date_minus_month(gk_date):
    d = datetime.datetime.strptime(str(gk_date), "%Y-%m-%d")
    date_mm = d + relativedelta(months=1)
    date_mm = datetime.datetime.strftime(date_mm, "%Y-%m-%d")
    return date_mm

def intersect(a, b):
    return list(set(a) & set(b))



def time_del(time1, time2, type="day"):
    """
    计算时间差
    :param time1: 较小的时间（datetime类型）
    :param time2: 较大的时间（datetime类型）
    :param type: 返回结果的时间类型（暂时就是返回相差天数）
    :return: 相差的天数
    """
    day1 = time.strptime(str(time1), '%Y-%m-%d')
    day2 = time.strptime(str(time2), '%Y-%m-%d')
    if type == 'day':
        day_num = (int(time.mktime(day2)) - int(time.mktime(day1))) / (
            24 * 60 * 60)
    return abs(int(day_num))

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

if __name__ == '__main__':
    args = Parser.Define_Params()
    file_path = '.' + args.file_path

    UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
    startTime = min(Times)
    endTime = max(Times)

    startTime = str(startTime)[0:10]
    endTime = str(endTime)[0:10]
    iu_neigh = args.iu_neigh

    # with open(iu_neigh, 'a') as file:
    #     for item in ItemToUser:
    #         file.write(item + '  ')
    #         file.write(str(ItemToUser[item])+'\n')



    TimeSeque = []
    while startTime <= endTime:
        startTime = date_minus_month(startTime)
        TimeSeque.append(startTime)

    # 计算用户对所有的项目的平均评分也是一个怀疑度
    # UserAveRat = {}
    # for user in UserToItem:
    #     temprat = 0
    #     for item in UserToItem[user]:
    #         ui = 'u' + user + 'i' + item
    #         temprat = temprat + int(newUItoRT[ui][0])
    #     temprat = temprat/len(UserToItem[user])
    #     UserAveRat.setdefault(user, temprat)
    # print(UserAveRat)

    # 计算项目的平均评分
    # ItemAveRat = {}
    # print(ItemToUser)
    # for item in ItemToUser:
    #     temprat = 0
    #     for user in ItemToUser[item]:
    #         ui = 'u' + user + 'i' + item
    #         temprat = temprat + int(newUItoRT[ui][0])
    #     temprat = temprat/len(ItemToUser[item])
    #     ItemAveRat.setdefault(item, temprat)

    # UIweight = {}
    # # 计算用户和项目之间的权重
    # for user in UserToItem:
    #     for item in UserToItem[user]:
    #         ui = 'u' + user + 'i' + item
    #         tempw = format(1/len(UserToItem[user]),'.4f')
    #         UIweight.setdefault(ui, tempw)
    #



    # save_uiweight_path = args.UIedge_weight
    # with open(save_uiweight_path, 'a') as file:
    #     for items in UIweight.keys():
    #         file.write(items + ' ')
    #         file.write(str(UIweight[items]) + '\n')


    UUweight = {}
    # 计算用户和用户之间的权重,这样太慢可以从项目中入手，都试试。
    # for i in range(len(UserToItem.keys())):
    #     for j in range(len(UserToItem.keys())):
    uuindexs = []
    sum = 0

    finish = 0
    time_thresold = 20
    rat_thresold = 2
    # print(len(UserToItem)*len(UserToItem))
    # exit(0)

    file_path = './edges/uuedge1.txt'


    for item in ItemToUser.keys():

        suser = len(ItemToUser[item])
        finishlen = len(ItemToUser.keys())
        finish = finish + 1
        if finish % 100 == 0:
            print('the all finishlen is %d , now run the %dth'%(finishlen,finish))

        for i in range(suser):
            for j in range(i + 1, suser):
                uuindex = 'u' + ItemToUser[item][i] + 'u' + ItemToUser[item][j]
                # 不能在这直接设置alias表，如果有两个相同的用户会覆盖的后面找不到key
                if uuindex not in uuindexs:
                    uuindexs.append(uuindex)
                else:
                    continue
                user1 = ItemToUser[item][i]
                user2 = ItemToUser[item][j]
                itemis = UserToItem[user1]
                itemjs = UserToItem[user2]
                common_items = intersect(itemis, itemjs)
                bing_items = list(set(itemis).union(set(itemjs)))
                temp = 0
                for item1 in common_items:
                    uuiindex = uuindex + 'i' + str(item1)

                    uii = 'u' + user1 + 'i' + str(item1)
                    uij = 'u' + user2 + 'i' + str(item1)
                    uiiRat = int(newUItoRT[uii][0])
                    uijRat = int(newUItoRT[uij][0])
                    uiiTime = newUItoRT[uii][1]
                    uijTime = newUItoRT[uij][1]
                    days = time_del(uiiTime, uijTime)
                    rats = math.fabs((uiiRat - uijRat))
                    if days < time_thresold:
                        time_ratio = days/time_thresold
                    else:
                        time_ratio = 0
                    if rats < rat_thresold:
                        rat_ratio = rats / rat_thresold
                    else:
                        rat_ratio = 0
                    des = (1 - time_ratio)/2 + (1 - rat_ratio)/2
                    temp = temp + des
                corres = (len(common_items) / len(bing_items)) * temp


                with open(file_path, 'a') as file:
                    file.write(str(uuindex) + ' ' + str(corres) + '\n')





    # uuflag = {}
    # uuEdge = {}
    # for uui in relev:
    #     user1 = str(uui).split('u')[1]
    #     user2 = str(uui).split('u')[-1].split('i')[0]
    #     uuindex = 'u' + user1 + 'u' + user2
    #     if uuindex not in uuflag:
    #         uuflag.setdefault(uuindex, 1)
    #     else:
    #         continue
    #     item = str(uui).split('u')[-1].split('i')[1]
    #
    #     itemis = UserToItem[user1]
    #     itemjs = UserToItem[user2]
    #     common_items = intersect(itemis, itemjs)
    #     bing_items = list(set(itemis).union(set(itemjs)))
    #     temp = 0
    #     for oitem in common_items:
    #         uuiindex = uuindex + 'i' + str(oitem)
    #         temp = temp + relev[uuiindex]
    #     corres = (len(common_items)/ len(bing_items)) * temp
    #     file_path = './edges/uuedge1.txt'
    #
    #     with open(file_path, 'a') as file:
    #         file.write(str(uuindex) + ' ' + str(corres) + '\n')
        # Edge = 2/(1 + math.exp())












