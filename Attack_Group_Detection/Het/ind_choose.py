import sys

import numpy as np
import random
import time
import datetime
import dateutil
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
import networkx
import datetime
import dateutil
import math
import scipy.stats
from dateutil.relativedelta import relativedelta

import random
import getData
import math

args = Parser.Define_Params()
file_path = '.' + args.file_path
save_uuweight_path = args.UUedge_weight
UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
format_uu_edge_filepath = args.format_UUedge_weight
format_uu_edge_filepath = './edges/format_UUedge1.txt'

uu_edge_weight = {}
with open(format_uu_edge_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('  ')
        uu_edge_weight.setdefault(line[0], line[1])

def timeformat_to_timestamp(timeformat=None,format = '%Y-%m-%d %H:%M:%S'):
    # try:
    if timeformat:
        time_tuple = time.strptime(timeformat,format)
        res = time.mktime(time_tuple) #转成时间戳
    else:
        res = time.time()        #获取当前时间戳
    return int(res)


# new_time = timeformat_to_timestamp('2012-06-04', format='%Y-%m-%d')
# new_time1 = timeformat_to_timestamp('2012-06-02', format='%Y-%m-%d')
# new = (new_time + new_time1) / 2
# new = time.strftime("%Y-%m-%d", time.localtime(new))
# print(new)
# exit(0)
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


def computeLG(settys,  CustToProd):
    ProdTimes = {}
    DestProd = {}
    LG = {}
    Pg = []
    # 双联通群组
    # rg：计算在每个双联通群组中，所有用户评论的产品的次数超过2的数量
    # pg：群组数量
    for setty in settys:
        if str(setty)[0] == '[':
            setty = str(setty)[1:]
        if str(setty)[-1] == ']':
            setty = str(setty)[:-1]
        for prod in CustToProd[str(setty)]:
            if prod not in ProdTimes.keys():
                ProdTimes.setdefault(prod, 1)
            else:
                ProdTimes[prod] = ProdTimes[prod] + 1
    for prod in ProdTimes:
        if ProdTimes[prod] >= 2:
            DestProd.setdefault(prod, ProdTimes[prod])
    # numbi = len(settys)
    Pg = DestProd.keys()
    rg = len(DestProd.keys())
    pg = len(settys)
    lg = 1.0/(1.0 + math.exp(3.0 - rg - pg))
    if math.isnan(lg):
        lg = 0
    # 等会再传标记。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
    # LG.setdefault(settys, lg)
    return lg, Pg

def computeNT(settys, LG):
    Weights = 0.0
    for CustI in settys:
        for CustJ in settys:
            tempidx = 'u' + str(CustI) + 'u' +str(CustJ)
            if tempidx in uu_edge_weight.keys():
                Weights = Weights + float(uu_edge_weight[tempidx])
    temp1 = len(settys)
    temp2 = len(settys) - 1
    temp = temp1 * temp2 / 2
    print(temp1)
    nt = Weights/temp
    nt = nt * LG
    return nt

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
# a = '2019-10-8'
# b = '2019-11-29'
# print(date_compare(a, b))
# exit(0)

# Pg:在该群组中评论一个产品次数超过两次的产品
def computeTW(settys, Pg, LG):
    DestProduct = Pg
    Tw = 0
    visited = True

    for Product in DestProduct:
        Start_time = 0
        End_time = 0
        for Cust in settys:
            tempidx = 'u' + str(Cust) + 'i' + str(Product)
            if tempidx in newUItoRT.keys():
                if Start_time == 0:
                    Start_time = newUItoRT[tempidx][1]
                if End_time == 0:
                    End_time = newUItoRT[tempidx][1]
                # if newUItoRT[tempidx][1] > Start_time:
                if date_compare(newUItoRT[tempidx][1], Start_time):
                    Start_time = newUItoRT[tempidx][1]
                # if int(newUPTime[tempidx].split('-')[0]) < End_time:
                if date_compare(newUItoRT[tempidx][1], End_time) == False:
                    End_time = newUItoRT[tempidx][1]

        timeDiff = time_del(End_time, Start_time)
        # print(Start_time,End_time,timeDiff)
        if timeDiff <= 10:
            Tw = Tw + (1.0 - timeDiff/10)

    pg = len(DestProduct)
    Tw = Tw/pg
    Tw = Tw * LG
    if math.isnan(Tw):
        Tw = 0.0
    return Tw

# Pg目标产品列表 LG公式中普遍乘的惩罚系数，settys：该双联通群组
def computeRV(settys, Pg, LG):
    DX = 0.0
    for DestProduct in Pg:
        EXX = 0.0
        EX = 0.0
        NumOfCustReviewP = 0.0
        for Cust in settys:
            tempidx = 'u' + str(Cust) + 'i' + str(DestProduct)
            if tempidx in newUItoRT.keys():
                EX = EX + int(newUItoRT[tempidx][0])
                EXX = EXX + int(newUItoRT[tempidx][0]) * int(newUItoRT[tempidx][0])
                NumOfCustReviewP = NumOfCustReviewP + 1
        EX = EX / NumOfCustReviewP
        EXX = EXX / NumOfCustReviewP
        DX = DX + (EXX - EX * EX)
    pg = len(Pg)
    #                                          标准差
    rv = 2.0 * (1.0 - (1.0 / (1.0 + math.exp(-(DX / pg))))) * LG
    if math.isnan(rv):
        rv = 0.0
    return rv

def computeRR(settys, Pg):
    rr = 0.0
    for DestProduct in Pg:
        NumOfCustReviewP = 0.0
        for cust in settys:
            tempidx = 'u' + str(cust) + 'i' + str(DestProduct)
            if tempidx in newUItoRT.keys():
                NumOfCustReviewP = NumOfCustReviewP + 1
        temprr = NumOfCustReviewP/len(ItemToUser[DestProduct])
        if temprr > rr:
            rr = temprr
    if math.isnan(rr):
        rr = 0.0

    return rr

def Find_index(list, name):
    count = 0
    for i in list:
        if i == name:
            count = count + 1

    return count

def compute_SGRD(group):
    # group = ['156', '945', '1826', '2208', '2275', '2507', '2816', '3224', '3706', '4116', '4439', '4590', '4619',
#             '4768']

    rats = []
    rat = ['1', '2', '3', '4', '5']
    all_item = []
    for user in group:
        for item in UserToItem[user]:
            all_item.append(item)
            tempidx = 'u' + user + 'i' + item
            rats.append(newUItoRT[tempidx][0])

    count = []
    whole = 0
    for i in rat:
        num = Find_index(rats, i)
        whole = whole + num
        count.append(num)
    Entropy = 0
    for i in range(5):

        if count[i] != 0:
            Entropy = Entropy + (count[i]/whole) * np.log2(count[i]/whole)

    P1 = 1/(1 - Entropy)

    chazhi = 0
    for item in all_item:
        g_rat = 0
        g_rat_num = 0
        a_rat = 0
        a_rat_num = 0
        for user in group:
            tempidx = 'u' + user + 'i' + item
            if tempidx in newUItoRT:
                rat = int(newUItoRT[tempidx][0])
                g_rat = g_rat + rat
                g_rat_num = g_rat_num + 1
        for all_user in ItemToUser[item]:
            tempidx1 = 'u' + all_user + 'i' + item
            rat = int(newUItoRT[tempidx1][0])
            a_rat = a_rat + rat
            a_rat_num = a_rat_num + 1
        g_ave = g_rat/g_rat_num
        a_ave = a_rat/a_rat_num
        # 这里除的那个|P|是群组中对该项目进行评价的数目么
        chazhi = chazhi + math.fabs(g_ave - a_ave)/g_rat_num
        # chazhi = chazhi + math.fabs(g_ave - a_ave)

    P2 = chazhi/len(all_item)


    SGRD = (P1 + P2)/2

    return SGRD
# 指标漏洞
def compute_MTWSD(group, thresold = 30):
    # group = ['156', '945', '1826', '2208', '2275', '2507', '2816', '3224', '3706', '4116', '4439', '4590', '4619',
    #         '4768']
    all_items = []
    TW = []
    for user in group:
        for item in UserToItem[str(user)]:
            all_items.append(item)

    all_items = list(set(all_items))
    for item in all_items:
        sigh = 0
        F_T = '2019-10-23'
        L_T = '1980-11-23'
        for user in group:
            tempidx = 'u' + str(user) + 'i' + str(item)
            if tempidx in newUItoRT:
                sigh = sigh + 1
                Time = newUItoRT[tempidx][1]
                if date_compare(F_T, Time):
                    F_T = Time
                if date_compare(L_T, Time) == False:
                    L_T = Time


        if time_del(L_T, F_T) < thresold and sigh > 1:
            temp_tw = 1 - (time_del(L_T, F_T))/thresold
        else:
            temp_tw = 0
        TW.append(temp_tw)
    all = 0
    for i in TW:
        all = all + i

    MTWSD = all/(len(TW))
    # TW = []
    # for user in group:
    #     F_T = '2020-3-25'
    #     L_T = '1920-3-25'
    #
    #     for item in UserToItem[user]:
    #         tempidx = 'u' + user + 'i' + item
    #         Time = newUItoRT[tempidx][1]
    #
    #         if date_compare(F_T, Time):
    #             F_T = Time
    #         if date_compare(L_T, Time) == False:
    #             L_T = Time
    #
    #     if time_del(L_T, F_T) < 30:
    #         temp_tw = 1 - (time_del(L_T, F_T))/thresold
    #     else:
    #         temp_tw = 0
    #     TW.append(temp_tw)
    # all = 0
    # for i in TW:
    #     all = all + i
    # MTWSD = all/(len(TW))

    return MTWSD

def compute_GURAC(group):
    # group = ['156', '945', '1826', '2208', '2275', '2507', '2816', '3224', '3706', '4116', '4439', '4590', '4619',
   #          '4768']

    Smax = 5
    User_rat_ave = []
    for user in group:
        temp = 0
        for item in UserToItem[user]:
            tempidx = 'u' + user + 'i' + item
            rat = int(newUItoRT[tempidx][0])
            temp = temp + rat
        User_rat_ave.append(temp/len(UserToItem[user]))
    count = 0
    for num in User_rat_ave:
        count = count + num
    Sg_ave = count/len(User_rat_ave)
    Fangcha = 0
    for num1 in User_rat_ave:
        Fangcha = Fangcha + math.pow((num1 - Sg_ave), 2)
    GURAC = Sg_ave/(Smax * (1 + math.sqrt(1/(len(User_rat_ave) - 1)*(Fangcha))))

    return GURAC


def compute_GERR(group, threold = 360):
    # group = ['156', '945', '1826', '2208', '2275', '2507', '2816', '3224', '3706', '4116', '4439', '4590', '4619',
    #         '4768']
    ERR = []
    all_items = []
    for user in group:

        for item in UserToItem[str(user)]:
            all_items.append(item)

    all_items = list(set(all_items))
    for item in all_items:
        L_TGP = '1980-11-23'
        E_P = '2019-10-23'
        for user in group:
            tempidx = 'u' + str(user) + 'i' + str(item)
            if tempidx in newUItoRT:
                Time = newUItoRT[tempidx][1]

                if date_compare(L_TGP, Time) == False:
                    L_TGP = Time

        for user1 in ItemToUser[item]:
            tempidx1 = 'u' + user1 + 'i' + item
            Time1 = newUItoRT[tempidx1][1]

            if date_compare(E_P, Time1):
                E_P = Time1

        if time_del(L_TGP, E_P) < threold:
            ERR.append(1 - (time_del(L_TGP, E_P)/threold))
        else:
            ERR.append(0)
    des = 0
    for i in ERR:
        des = des + i

    GERR = des/(len(ERR))

    return GERR

                # 大 小 小
# group = [['2078', '2778', '2909', '2978', '3235', '3487', '3690', '4198', '4526', '4636'], ['495', '635', '2204', '2777', '3306']
#          ,['4839', '4849', '4864', '4869', '4891', '4899'],['3331', '3506', '3518', '4683'],
#          ['343', '660', '782', '1560', '2195', '3723'],['917', '1079', '1361', '2191'],['1413', '1424', '2016'],
#          ['3644', '4892'],['4538', '4740']]
#
# for i in group:
#     print(compute_GURAC(i))
#     print(compute_GERR(i))
#     print(compute_MTWSD(i))
#     print(compute_SGRD(i))
#     print('###############')
# exit(0)

def ditance_gegree(group):
    # group = ['156', '945', '1826', '2208', '2275', '2507', '2816', '3224', '3706', '4116', '4439', '4590', '4619', '4768']
    jiaotemp = UserToItem[str(group[0])]
    bingtemp = []

    for user in group:
        bingtemp = list(set(bingtemp).union(set(UserToItem[str(user)])))
        jiaotemp = list(set(jiaotemp).intersection(set(UserToItem[str(user)])))
        # print(UserToItem[str(user)])
    des = 0
    aa = 0
    if len(jiaotemp) >= 2:
        for item in jiaotemp:
            agroup_time = []
            agroup_rat = []
            rat_sum = 0
            time_sum = 0
            for user in group:
                ui = 'u' + str(user) + 'i' + str(item)
                RT = newUItoRT[ui]
                agroup_time.append(RT[1])
                agroup_rat.append(RT[0])
                rat_sum = rat_sum + int(RT[0])
                time_sum = time_sum + timeformat_to_timestamp(RT[1], format='%Y-%m-%d')


                # new_time1 = timeformat_to_timestamp('2012-06-02', format='%Y-%m-%d')
                # new = (new_time + new_time1) / 2
                # new = time.strftime("%Y-%m-%d", time.localtime(new))
                # new_time = timeformat_to_timestamp(RT[1], format='%Y-%m-%d')
            rat_sum = rat_sum / len(group)
            time_sum = time_sum / len(group)
            advance_time = new = time.strftime("%Y-%m-%d", time.localtime(time_sum))
            for i in range(len(agroup_rat)):

                if math.fabs(float(agroup_rat[i]) - rat_sum) < 0.5 and time_del(advance_time, agroup_time[i]) < 30:
                    aa = aa + 1
                elif 0.5 < math.fabs(float(agroup_rat[i]) - rat_sum) < 1 and time_del(advance_time, agroup_time[i]) < 30:
                    aa = aa + 0.5
        des = aa / (len(jiaotemp) * len(group))
        # lg, Pg = computeLG(group, UserToItem)
        # NT = computeNT(group, lg)
        # print(NT)


    else:
        ab = 0
        num = 0
        des1 = 0

        for user in group:
            # for item in UserToItem[str(user)]:
            num = num + len(UserToItem[str(user)])
            #     rat = 0
            #     for users in ItemToUser[item]:
            #         tempidx = 'u' + users + 'i' + item
            #         rat = int(newUItoRT[tempidx][0]) + rat
            #     ave_rat = rat / len(ItemToUser[item])
            #     tempidx1 =  'u' + str(user) + 'i' + str(item)
            #     single_rat = newUItoRT[tempidx1][0]
            #     if math.fabs(ave_rat - float(single_rat)) < 0.5:
            #         ab = ab + 0.5
            #     elif 0.5 < math.fabs(ave_rat - float(single_rat)) < 1.0:
            #         ab = ab + 0.25
            for user1 in group:
                if user != user1:
                    jiaotemp = list(set(UserToItem[str(user)]).intersection(set(UserToItem[str(user1)])))
                    for item in jiaotemp:
                        tempidx =  'u' + str(user) + 'i' + str(item)
                        tempidx1 =  'u' + str(user1) + 'i' + str(item)
                        userRat = int(newUItoRT[tempidx][0])
                        user1Rat = int(newUItoRT[tempidx1][0])
                        userTime = newUItoRT[tempidx][1]
                        user1Time = newUItoRT[tempidx1][1]
                        if user1Rat == userRat and time_del(userTime, user1Time) < 30:
                            des1 = des1 + 2.0




        # des = 0.25*(ab/num) + 0.75*(des1/num)
        des = des1/num
        if des > 1.0:
            des = 1.0

        if des == 0:
            pass

    return des










    # 碰到这种分割一半都不行太难了吧
    #     num = len(UserToItem[group[0]])
    #     min_index = 0
    #
    #     sub_min_index = 0
    #     for i in range(len(group)):
    #         if num > len(UserToItem[group[i]]):
    #             num = len(UserToItem[group[i]])
    #             min_index = i
    #     min_store = group[min_index]
    #     del group[min_index]
    #
    #     num = len(UserToItem[group[0]])
    #     for i in range(len(group)):
    #         if num > len(UserToItem[group[i]]):
    #             num = len(UserToItem[group[i]])
    #             sub_min_index = i
    #     sub_min_store = group[sub_min_index]
    #     del group[sub_min_index]
    #     first_group = group[: len(group)//2]
    #     first_group.append(min_store)
    #     second_group = group[len(group)//2 :]
    #     second_group.append(sub_min_store)
    #     print(first_group)
    #     print(second_group)
    #     exit(0)
    #     for user in first_group:
    #         bingtemp = list(set(bingtemp).union(set(UserToItem[str(user)])))
    #         jiaotemp = list(set(jiaotemp).intersection(set(UserToItem[str(user)])))
    #     print(jiaotemp)
    #     aa = 0
    #     for item in jiaotemp:
    #         agroup_time = []
    #         agroup_rat = []
    #         rat_sum = 0
    #         time_sum = 0
    #         for user in first_group:
    #             ui = 'u' + user + 'i' + item
    #             RT = newUItoRT[ui]
    #             agroup_time.append(RT[1])
    #             agroup_rat.append(RT[0])
    #             rat_sum = rat_sum + int(RT[0])
    #             time_sum = time_sum + timeformat_to_timestamp(RT[1], format='%Y-%m-%d')
    #
    #
    #         rat_sum = rat_sum / len(first_group)
    #         time_sum = time_sum / len(first_group)
    #         advance_time = time.strftime("%Y-%m-%d", time.localtime(time_sum))
    #         for i in range(len(agroup_rat)):
    #
    #             if math.fabs(float(agroup_rat[i]) - rat_sum) < 0.5 and time_del(advance_time, agroup_time[i]) < 30:
    #                 aa = aa + 1
    #             elif 0.5 < math.fabs(float(agroup_rat[i]) - rat_sum) < 1 and time_del(advance_time, agroup_time[i]) < 30:
    #                 aa = aa + 0.5
    #     des1 = aa / (len(jiaotemp) * len(first_group))
    #
    #
    #     for user in second_group:
    #         bingtemp = list(set(bingtemp).union(set(UserToItem[str(user)])))
    #         jiaotemp = list(set(jiaotemp).intersection(set(UserToItem[str(user)])))
    #
    #     print(jiaotemp)
    #     aa = 0
    #     for item in jiaotemp:
    #         agroup_time = []
    #         agroup_rat = []
    #         rat_sum = 0
    #         time_sum = 0
    #         for user in second_group:
    #             ui = 'u' + user + 'i' + item
    #             RT = newUItoRT[ui]
    #             agroup_time.append(RT[1])
    #             agroup_rat.append(RT[0])
    #             rat_sum = rat_sum + int(RT[0])
    #             time_sum = time_sum + timeformat_to_timestamp(RT[1], format='%Y-%m-%d')
    #
    #         rat_sum = rat_sum / len(second_group)
    #         time_sum = time_sum / len(second_group)
    #         advance_time = time.strftime("%Y-%m-%d", time.localtime(time_sum))
    #         for i in range(len(agroup_rat)):
    #
    #             if math.fabs(float(agroup_rat[i]) - rat_sum) < 0.5 and time_del(advance_time, agroup_time[i]) < 30:
    #                 aa = aa + 1
    #             elif 0.5 < math.fabs(float(agroup_rat[i]) - rat_sum) < 1 and time_del(advance_time, agroup_time[i]) < 30:
    #                 aa = aa + 0.5
    #     des2 = aa / (len(jiaotemp) * len(second_group))
    #     des = (des1 + des2)/2
    # print(des)
    # exit(0)

def vision(spam_UserID, field1, field2, name):
    import matplotlib.pyplot as plt


    plt.scatter(spam_UserID[:len(spam_UserID)//2], field1, c='red', label='spam')
    plt.scatter(spam_UserID[len(spam_UserID)//2:], field2, c='blue', label='gene')
    plt.legend(loc='best')
    plt.title(name)
    plt.show()

# file_path = './edges/last_candidate_group.txt'
# file_path = './edges/last_candidate_group1.txt'
# file_path = './edges/last_testgroup.txt'
# file_path = './edges/last_candidate_group30.txt'
# file_path = './edges/last_candidate_group15.txt'
# file_path = './edges/last_candidate_group20.txt'
# file_path = './edges/last_candidate_group302.txt'
file_path = './edges/Candidate_group_att.txt'


candidate_group = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')[1: -1].split(', ')
        if len(line) > 1:
            for i in range(len(line)):

                line[i] = str(line[i][1: -1])
                if line[i][0] == "'":
                    line[i] = str(line[i][1: -1])
                    line[i] = line[i].strip()

            candidate_group.append(line)
# print(candidate_group)
# exit(0)


des_group = {}
if candidate_group[0][0][0] == 'u':
    for i in range(len(candidate_group)):
        for j in range(len(candidate_group[i])):
            candidate_group[i][j] = candidate_group[i][j][1:]


for group in candidate_group:
    des = (compute_GERR(group) + compute_MTWSD(group) + compute_SGRD(group) + compute_GURAC(group)) / 4

    if des not in des_group:
        des_group.setdefault(des, [group])
    else:
        temp = des_group[des]
        temp.append(group)
        des_group[des] = temp

# f = "./edges/AMSpamGroup.txt"
# f = "./edges/testAMSpamGroup.txt"
# f = "./edges/testAMSpamGroup1.txt"
# f = "./edges/testAMSpamGroup2.txt"
# f = "./edges/testAMSpamGroup3.txt"
# f = "./edges/AMSpamGroup30.txt"
# f = "./edges/AMSpamGroup15.txt"
# file_path = "./edges/AMSpamGroup302.txt"
file_path = "./edges/ScanAMSpamGroupatt.txt"
f = open(file_path, 'r+')
f.truncate()
f.close()

des_group1 = sorted(des_group.keys(), reverse=True)

for i in range(len(des_group)):
    j = des_group1[i]
    for items in des_group[j]:
        with open(file_path, "a") as file:
            file.write(str(j) +'  ' + str(items)+'\n')




# 垃圾群组检测
# print('垃圾群组检测####################################')
# file_path = './edges/SP.txt'
# spam_group = []
# spam_UserID = []
# sNT = []
# sTW = []
# sRV = []
# sRR = []
# i = 0
# with open(file_path, 'r') as f:
#     lines = f.readlines()
#
#     for line in lines:
#         i = i + 1
#         spam_UserID.append(i)
#
#         line = line.strip('\n').split('\t')[2: -1]
#         NT, TW, RV, RR = ditance_gegree(line)
#         sNT.append(NT)
#         sTW.append(TW)
#         sRV.append(RV)
#         sRR.append(RR)

# print('真实群组检测##########################################')
# file_path = './edges/gene_group.txt'
# gen_UserID = []
# gNT = []
# gTW = []
# gRV = []
# gRR = []
# j = 0
# with open(file_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#
#         if float(line.split('  ')[0]) < 0.8:
#             line = line.strip('\n').split('  ')[1][1:-1].split(', ')
#             j = j + 1
#             gen_UserID.append(j)
#             NT, TW, RV, RR = ditance_gegree(line)
#             gNT.append(NT)
#             gTW.append(TW)
#             gRV.append(RV)
#             gRR.append(RR)
#
# print(spam_UserID)
# print(gen_UserID)
# print(sNT[:len(sNT)//2])
# print(sNT[len(sNT)//2: ])
# vision(sNT[:len(sNT)//2], sNT[len(sNT)//2: ], 'NT')
# vision(sTW[:len(sNT)//2], sTW[len(sNT)//2: ], 'TW')
# vision(sRV[:len(sNT)//2], sRV[len(sNT)//2: ], 'RV')
# vision(sRR[:len(sNT)//2], sRR[len(sNT)//2: ], 'RR')


