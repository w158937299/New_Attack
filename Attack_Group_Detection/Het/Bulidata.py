import numpy as np
import random
import time
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import Parser


def timeformat_to_timestamp(timeformat=None,format = '%Y-%m-%d %H:%M:%S'):
    # try:
    if timeformat:
        time_tuple = time.strptime(timeformat,format)
        res = time.mktime(time_tuple) #转成时间戳
    else:
        res = time.time()        #获取当前时间戳
    return int(res)


def bulid_dataset(file_name):
    UserID = []
    ItemID = []
    Ratings = []
    Attacks = []
    Times = []
    tUserID = []
    tItemID = []
    tRatings = []
    tAttacks = []
    tTimes = []
    UItoR = {}
    tUItoR = {}
    UserToItem = {}
    UserToTime = {}
    UserToRat = {}
    ItemToUser = {}

    with open(file_name, 'r') as f:
        lines = f.readlines()
        train_data = lines[: int(9/10*(len(lines)))]
        test_data = lines[int(9/10*(len(lines))) + 1:]

        for line in train_data:
            line = line.split()
            if len(line) == 5:
                UI = str(line[0]) + '-' + str(line[1])
                UItoR[UI] = line[2]
                UserID.append(int(line[0]))
                ItemID.append(int(line[1]))
                Ratings.append(int(line[2]))
                Attacks.append(int(line[3]))
                Times.append(line[4])
                if line[0] not in UserToItem:
                    UserToItem[line[0]] = [line[1]]
                    new_time = timeformat_to_timestamp(line[4], format='%Y-%m-%d')
                    UserToTime[line[0]] = [new_time]
                    UserToRat[line[0]] = [line[2]]
                else:
                    temp = UserToItem[line[0]]
                    tempt = UserToTime[line[0]]
                    tempr = UserToRat[line[0]]
                    temp.append(line[1])
                    new_time = timeformat_to_timestamp(line[4], format='%Y-%m-%d')
                    tempt.append(new_time)
                    tempr.append(line[2])
                    UserToItem[line[0]] = temp
                    UserToTime[line[0]] = tempt
                    UserToRat[line[0]] = tempr
                if line[1] not in ItemToUser:
                    ItemToUser[line[1]] = [line[0]]
                else:
                    temp = ItemToUser[line[1]]
                    temp.append(line[0])
                    ItemToUser[line[1]] = temp


            else:
                continue

        for line in test_data:
            line = line.split()
            if len(line) == 5:
                UI = str(line[0]) + '-' + str(line[1])
                tUItoR[UI] = line[2]
                tUserID.append(int(line[0]))
                tItemID.append(int(line[1]))
                tRatings.append(int(line[2]))
                tAttacks.append(int(line[3]))
                tTimes.append(line[4])
                if line[0] not in UserToItem:
                    UserToItem[line[0]] = [line[1]]
                    new_time = timeformat_to_timestamp(line[4], format='%Y-%m-%d')
                    UserToTime[line[0]] = [new_time]
                    UserToRat[line[0]] = [line[2]]
                else:
                    temp = UserToItem[line[0]]
                    tempt = UserToTime[line[0]]
                    tempr = UserToRat[line[0]]
                    temp.append(line[1])
                    new_time = timeformat_to_timestamp(line[4], format='%Y-%m-%d')
                    tempt.append(new_time)
                    tempr.append(line[2])
                    UserToItem[line[0]] = temp
                    UserToTime[line[0]] = tempt
                    UserToRat[line[0]] = tempr
                if line[1] not in ItemToUser:
                    ItemToUser[line[1]] = [line[0]]
                else:
                    temp = ItemToUser[line[1]]
                    temp.append(line[0])
                    ItemToUser[line[1]] = temp



            else:
                continue
        # for line in test_data:
        #     line = line.split()
        #     if len(line) == 5:
        #         UserID.append(line[0])
        #         ItemID.append(line[1])
        #         Ratings.append(line[2])
        #         Attacks.append(line[3])
        #         Times.append(line[4])
        #     else:
        #         continue
    new_times = []
    tnew_times = []
    whole_times = []
    for time in Times:
        new_time = timeformat_to_timestamp(time, format='%Y-%m-%d')
        new_times.append(new_time)
        whole_times.append(new_time)
    for time in tTimes:
        new_time = timeformat_to_timestamp(time, format='%Y-%m-%d')
        tnew_times.append(new_time)
        whole_times.append(new_time)



    return UserID, ItemID, Ratings, Attacks, Times, UItoR, \
           ItemToUser, UserToItem, UserToTime, UserToRat, \
           tUserID, tItemID, tRatings, tAttacks,\
           tTimes, tUItoR, new_times, tnew_times, whole_times



