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

# def format_weight(weights):
#     format_weights = []
#     temp_max = max(weights)
#     temp_min = min(weights)
#
#     for ele in weights:
#         format_weights.append(round(10 * (ele - temp_min / 4) / (temp_max - temp_min), 2))
#     return format_weights

def format_weight(weights, corre_users):
    format_weights = []
    total = 0
    count = 0
    low_users = []
    for weight in weights:
        total = total + weight
    for i in range(len(weights)):
        weight = weights[i]
        new_weight = float(format(weight/total,'.6f'))
        if new_weight == 0.0:

            count = count + 1
            low_users.append(corre_users[i])
        format_weights.append(new_weight)


    # a = 0
    # for weight in format_weights:
    #     a = a + weight
    # print(a)
    # alias = float(format((1 - a)/count,'.5f'))
    # for i in range(len(format_weights)):
    #     weight = format_weights[i]
    #     if weight == 0.0:
    #         format_weights[i] = alias
    #

    return format_weights

def format_edge(edges, weights, type):
    format_edges = {}

    for user in edges:
        temp_weights = []
        corres_users = []
        format_weights = []
        for item in edges[user]:
            if type == 'u':
                ui = 'u' + str(user) + 'u' + str(item)
            elif type == 'i':
                ui = 'i' + str(user) + 'i' + str(item)
            temp_weights.append(float(weights[ui]))
            corres_users.append(ui)

        format_weights = format_weight(temp_weights, corres_users)
        for i in range(len(corres_users)):
            format_edges.setdefault(corres_users[i], format_weights[i])



    return format_edges

def get_Des(file_path, type):
    UUweight = {}
    U_neigh = {}
    with open(file_path, 'r') as f:

        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            weight = line[1].strip('\n')
            UUweight.setdefault(line[0], weight)
            if type == 'u':
                Users = line[0].split('u')
            elif type == 'i':
                Users = line[0].split('i')
            user = int(Users[1])
            neighbor = int(Users[2])
            if user not in U_neigh.keys():
                U_neigh.setdefault(user, [neighbor])
            else:
                temp = U_neigh[user]
                temp.append(neighbor)
                U_neigh[user] = temp

    UUweight = format_edge(U_neigh, UUweight, type)
    if type == 'u':
        save_uuweight_path = './edges/format_UUedge1.txt'
        with open(save_uuweight_path, 'a') as file:
            for items in UUweight.keys():
                file.write(items + '   ')
                file.write(str(UUweight[items]) + '\n')
    elif type == 'i':
        save_uiweight_path = './edges/format_IIedge1.txt'
        with open(save_uiweight_path, 'a') as file:
            for items in UUweight.keys():
                file.write(items + '   ')
                file.write(str(UUweight[items]) + '\n')



if __name__ == '__main__':
    args = Parser.Define_Params()
    file_path = '.' + args.file_path
    save_uuweight_path = args.UUedge_weight
    UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
    uuedge_file_path = args.UUedge_weight
    uiedge_file_path = args.UIedge_weight

    UUweight = {}
    U_neigh = {}
    uuedge_file_path = './edges/uuedge1.txt'

    # get_Des(uuedge_file_path, 'u')


    iiedge_file_path = './edges/iiedge1.txt'
    get_Des(iiedge_file_path, 'i')
    # with open(uuedge_file_path, 'r') as f:
    #
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split(' ')
    #
    #         weight = line[1].strip('\n')
    #         UUweight.setdefault(line[0], weight)
    #         Users = line[0].split('u')
    #         user = int(Users[1])
    #         neighbor = int(Users[2])
    #
    #         if user not in U_neigh.keys():
    #
    #             U_neigh.setdefault(user, [neighbor])
    #
    #         else:
    #
    #             temp = U_neigh[user]
    #             temp.append(neighbor)
    #             U_neigh[user] = temp
    #
    #
    #
    # UUweight = format_edge(U_neigh, UUweight)
    # save_uuweight_path = args.format_UUedge_weight
    # save_uuweight_path = './edges/format_UUedge1.txt'
    # with open(save_uuweight_path, 'a') as file:
    #     for items in UUweight.keys():
    #         file.write(items + '   ')
    #         file.write(str(UUweight[items]) + '\n')





