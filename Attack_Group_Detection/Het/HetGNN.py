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
    save_uuweight_path = args.UUedge_weight
    UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
    startTime = min(Times)
    endTime = max(Times)

    startTime = str(startTime)[0:10]
    endTime = str(endTime)[0:10]
    iu_neigh = args.iu_neigh
    print(ItemToUser)
    with open(iu_neigh, 'a') as file:
        for item in ItemToUser:
            file.write(item + '  ')
            file.write(str(ItemToUser[item])+'\n')



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

    UIweight = {}
    # 计算用户和项目之间的权重
    for user in UserToItem:
        for item in UserToItem[user]:
            ui = 'u' + user + 'i' + item
            tempw = format(1/len(UserToItem[user]),'.4f')
            UIweight.setdefault(ui, tempw)




    save_uiweight_path = args.UIedge_weight
    with open(save_uiweight_path, 'a') as file:
        for items in UIweight.keys():
            file.write(items + ' ')
            file.write(str(UIweight[items]) + '\n')


    UUweight = {}
    # 计算用户和用户之间的权重,这样太慢可以从项目中入手，都试试。
    # for i in range(len(UserToItem.keys())):
    #     for j in range(len(UserToItem.keys())):
    uuindexs = []
    sum = 0

    finish = 0

    # print(len(UserToItem)*len(UserToItem))
    # exit(0)

    for item in ItemToUser.keys():
        suser = len(ItemToUser[item])
        finishlen = len(ItemToUser.keys())
        if finish % 100 == 0:
            print('the all finishlen is %d , now run the %dth'%(finishlen,finish))

        for i in range(suser):
            for j in range(i + 1, suser):
                sum = sum + 1
                uuindex = 'u' + ItemToUser[item][i] + 'u' + ItemToUser[item][j]
                if uuindex not in uuindexs:
                    uuindexs.append(uuindex)
                else:
                    continue
                itemis = UserToItem[ItemToUser[item][i]]
                itemjs = UserToItem[ItemToUser[item][j]]
                common_items = intersect(itemis, itemjs)
                uiis = []
                uijs = []
                uiirats = []
                uijrats = []
                # uiitimes = []
                # uijtimes = []
                timedays = 0
                for single_item in common_items:

                    uii = 'u' + ItemToUser[item][i] + 'i' + single_item
                    uiis.append(uii)

                    uij = 'u' + ItemToUser[item][j] + 'i' + single_item
                    uijs.append(uij)
                for k in range(len(uiis)):
                    newuii = uiis[k]
                    newuij = uijs[k]
                    uiiRat = newUItoRT[newuii][0]
                    uijRat = newUItoRT[newuij][0]
                    uiiTime = newUItoRT[newuii][1]
                    uijTime = newUItoRT[newuij][1]
                    # uiitimes.append(uiiTime)
                    # uijtimes.append(uijTime)
                    uiirats.append(uiiRat)
                    uijrats.append(uijRat)
                    timedays = timedays + time_del(uiiTime, uijTime)
                # print(len(uiis))
                # print(len(common_items))
                # 这两个权重采用同一种度量方式是不是更好一些
                # 时间的权重, 越相似则权重越大，0-1
                ave_time = timedays/len(common_items)
                Similar_time = 1 - math.tanh(ave_time/12)
                Similar_time = float(format(Similar_time, '.4f'))
                # 评分的权重
                p = np.array(list(map(int, uiirats)))
                q = np.array(list(map(int, uijrats)))
                Similar_rat = float(format(12 * JS_divergence(p, q),'.4f'))
                if Similar_rat > 1:
                    Similar_rat = 1
                # print('time   ', Similar_time)
                # print('rat',  Similar_rat)
                ijweight = float(format((Similar_time + Similar_rat)/2, '.4f'))

                if ijweight != 0:
                    UUweight.setdefault(uuindex, ijweight)
        finish = finish + 1


    with open(save_uuweight_path, 'a') as f:
        for items in UUweight.keys():
            f.write(items + ' ')
            f.write(str(UUweight[items]) + '\n')
    # with open(save_uuweight) as f:
    #     f.writelines()

    print(sum)

                # timedays = time_del(uiiTime, uijTime)
                # 最后看看差一分效果会不会更好点
                # if timedays <= 30 and uiiRat == uijRat:
                #     newuiis.append(newuii)
                #     newuijs.append(newuij)








    # 计算每个项目对应的不同用户,他们对该用户评论的时间和评分来划分群体
    # for item in ItemToUser.keys():
    #     temp = {}
    #     for user in ItemToUser[item]:
    #
    #         UI = 'u' + user + 'i' + item
    #         temp[UI] = newUItoRT[UI]
    #     timedict = {}
    #     ratdict = {}
    #     for index in range(len(temp.values())):
    #
    #         tindex = list(temp.values())[index][1]
    #         ui = list(temp.keys())[index]
    #
    #         for i in range(len(TimeSeque) - 1):
    #
    #             if tindex > TimeSeque[i] and tindex < TimeSeque[i + 1]:
    #                 if TimeSeque[i] not in timedict.keys():
    #                     timedict.setdefault(TimeSeque[i], [ui])
    #                 else:
    #                     temp1 = timedict[TimeSeque[i]]
    #                     temp1.append(ui)
    #                     timedict[TimeSeque[i]] = temp1
    #
    #         # if list(temp.values())[index][0] == list(temp.values())[index + 1][0]:
    #
    #         if list(temp.values())[index][0] not in ratdict.keys():
    #             ratdict.setdefault(list(temp.values())[index][0], [ui])
    #         else:
    #             temp1 = ratdict[list(temp.values())[index][0]]
    #             temp1.append(ui)
    #             ratdict[list(temp.values())[index][0]] = temp1
    #     # agroup = [list(filter(lambda x: x in list(ratdict.values()), sublist)) for sublist in list(timedict.values())]
    #
    #     tempUser = []
    #     for ratUser in list(ratdict.values()):
    #         for timeUser in list(timedict.values()):
    #             if len(intersect(ratUser, timeUser)) != 0:
    #                 tempUser.append(intersect(ratUser, timeUser))
    #     print(list(ratdict.values()))
    #     print(list(timedict.values()))
    #     print(tempUser)
    #     exit(0)


