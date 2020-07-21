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

import random
import getData
from tools import create_table
# from tools import get_het_random_walk
from tools import Compute_Sample_ratio
from tools import sample_neigh
from getData import get_last_data

def get_train_data():
    args = Parser.Define_Params()
    uu_neigh, ui_neigh, iu_neigh, ii_neigh, uu_edge_weight, ui_edge_weight = get_last_data()

    u_l = args.u_l
    i_l = args.i_l
    walk_length = args.walk_length
    windows_length = args.windows_len
    types = [1, 2]

    whole_l = 0
    random_walk = []
    # args = Parser.Define_Params()
    # file_path = '.' + args.file_path
    # UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)
    # print(len(UserToItem))
    # print(len(uu_neigh))
    # print(len(ui_neigh))
    # print(ui_neigh)
    # print(uu_neigh)





    # 制作训练集
    uutriple_list = []
    uitriple_list = []
    iutriple_list = []
    iitriple_list = []
    # random_walk_filepath = args.het_random_walk_fliepath
    random_walk_filepath = './edges/het_random_walk1.txt'
    random_walk_filepath = './edges/het_random_walk2.txt'
    random_walk_list = []
    with open(random_walk_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:

            line = line.strip('\n')[0:-1].split(',')
            temp = []
            for single in line:
                temp.append(single[2:-1])
            random_walk_list.append(temp)

    compute_ratio = Compute_Sample_ratio(random_walk_list)

    for random_walk in random_walk_list:
        for i in range(len(random_walk)):
            if random_walk[i][0] == 'u':
                for k in range(i - windows_length, i + windows_length + 1):
                    # 不用多加k!=0的条件吧
                    if k != i and k < walk_length and k:
                        neigh_node = random_walk[k]
                        if neigh_node[0] == 'u' and random.random() < compute_ratio['uu']:
                            neg_node = random.randint(0, u_l)
                            neg_node = 'u' + str(neg_node)
                            if random_walk[i] in uu_neigh:
                                while neg_node in uu_neigh[random_walk[i]]:
                                    neg_node = random.randint(0, u_l)
                                    neg_node = 'u' + str(neg_node)

                                uutriple_list.append([int(random_walk[i][1:]), int(neigh_node[1:]), int(neg_node[1:])])

                        elif neigh_node[0] == 'i' and random.random() < compute_ratio['ui']:
                            neg_node = random.randint(0, i_l)
                            neg_node = 'i' + str(neg_node)
                            while neg_node in ui_neigh[random_walk[i]]:
                                neg_node = random.randint(0, i_l)
                                neg_node = 'i' + str(neg_node)
                            uitriple_list.append([int(random_walk[i][1:]), int(neigh_node[1:]), int(neg_node[1:])])

            elif random_walk[i][0] == 'i':
                for k in range(i - windows_length, i + windows_length + 1):
                    # 不用多加k!=0的条件吧
                    if k != i and k < walk_length and k:
                        neigh_node = random_walk[k]
                        if neigh_node[0] == 'u' and random.random() < compute_ratio['iu']:
                            neg_node = random.randint(0, u_l)
                            neg_node = 'u' + str(neg_node)
                            while neg_node in iu_neigh[random_walk[i]]:
                                neg_node = random.randint(0, u_l)
                                neg_node = 'u' + str(neg_node)
                            iutriple_list.append([int(random_walk[i][1:]), int(neigh_node[1:]), int(neg_node[1:])])
                        elif neigh_node[0] == 'i' and random.random() < compute_ratio['ii']:
                            neg_node = random.randint(0, i_l)
                            neg_node = 'i' + str(neg_node)
                            while neg_node in ii_neigh[random_walk[i]]:
                                neg_node = random.randint(0, i_l)
                                neg_node = 'i' + str(neg_node)
                            iitriple_list.append([int(random_walk[i][1:]), int(neigh_node[1:]), int(neg_node[1:])])



    triple_list = [[],[],[],[]]
    triple_list[0] = uutriple_list
    triple_list[1] = uitriple_list
    triple_list[2] = iutriple_list
    triple_list[3] = iitriple_list
    # print(iutriple_list)
    # print(uutriple_list)
    # print(uitriple_list)

    return triple_list

    # for user in ui_neigh:
    #     while whole_l < windows_length:
    #         types = random.randint(1, 2)
    #
    #         # 对uu进行处理
    #         if user in uu_neigh.keys() and types == 1 :
    #             temp_neigh = uu_neigh[user]
    #             temp_edge = uu_edge_weight
    #             temptable = create_table(user, temp_neigh, types)
    #             neigh_node = sample_neigh(temptable)
    #             # print(neigh_node)
    #             neg_node = random.randint(0, u_l)
    #             neg_node = 'u'+ str(neg_node)
    #             while neg_node in uu_neigh[user]:
    #                 neg_node = random.randint(0, u_l)
    #                 neg_node = 'u' + str(neg_node)
    #             uutriple_list.append([user, neigh_node, neg_node])
    #             # print(user, neigh_node, neg_node)
    #             # exit(0)
    #         elif user not in uu_neigh.keys() and types == 1:
    #             break
    #         # 对ui进行处理
    #         elif user in ui_neigh.keys() and types == 2:
    #             temp_neigh = ui_neigh[user]
    #
    #             temp_neigh = temp_neigh[1:-1].split(',')
    #
    #             neigh_node = random.choice(list(temp_neigh))[2:-1]
    #
    #             neg_node = random.randint(0, i_l)
    #             neg_node = 'i' + str(neg_node)
    #             while neg_node in ui_neigh[user]:
    #                 neg_node = random.randint(0, i_l)
    #                 neg_node = 'i' + str(neg_node)
    #             uitriple_list.append([user, neigh_node, neg_node])
    #         elif user not in ui_neigh.keys() and types == 2:
    #             break
    #     print(uutriple_list)
    #     print(windows_length)





