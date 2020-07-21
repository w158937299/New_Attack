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
import getData
import networkx
import datetime
import dateutil
import math
import scipy.stats
from dateutil.relativedelta import relativedelta

import random


# from tools import get_het_random_walk
from tools import Compute_Sample_ratio
from tools import sample_neigh

def bulid_dataset(file_name):
    UItoRT = {}
    Times = []
    UserToItem = {}
    ItemToUser = {}
    newUItoRT = {}

    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0] not in UserToItem:
                UserToItem[line[0]] = [line[1]]

            else:
                temp = UserToItem[line[0]]
                temp.append(line[1])
                UserToItem[line[0]] = temp

            if line[1] not in ItemToUser:
                ItemToUser[line[1]] = [line[0]]
            else:
                temp = ItemToUser[line[1]]
                temp.append(line[0])
                ItemToUser[line[1]] = temp

            UI = 'u' + line[0] + 'i' + line[1]
            newUItoRT.setdefault(UI, [line[2], line[4]])
            UItoRT.setdefault(UI, [line[0], line[1], line[2], line[4]])
            line[4] = datetime.datetime.strptime(line[4], "%Y-%m-%d")

            Times.append(line[4])

    return UItoRT, Times, UserToItem, ItemToUser, newUItoRT



def get_last_data():
    args = Parser.Define_Params()

    format_uu_edge_filepath = args.format_UUedge_weight
    format_uu_edge_filepath = './edges/format_UUedge1.txt'
    format_ii_edge_filepath = './edges/format_IIedge1.txt'
    ui_edge_filepath = args.UIedge_weight
    uu_neigh_filepath = args.uu_neigh
    uu_neigh_filepath = './edges/uu_neigh1.txt'
    ii_neigh_filepath = './edges/ii_neigh1.txt'
    ui_neigh_filepath = args.ui_neigh
    iu_neigh_filepath = args.iu_neigh
    uu_edge_weight = {}
    ui_edge_weight = {}
    ii_edge_weight = {}
    uu_neigh = {}
    ui_neigh = {}
    iu_neigh = {}
    ii_neigh = {}


    with open(iu_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:

            line = line.strip('\n')
            line = line.split('  ')
            item = 'i' + line[0]
            temp = []
            Users = line[1][1:-1].split(',')
            for user in Users:

                des_user = 'u' + user.strip()[1:-1]
                temp.append(des_user)
            iu_neigh.setdefault(item, temp)



    with open(format_uu_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('  ')
            uu_edge_weight.setdefault(line[0], line[1])


    with open(ui_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            ui_edge_weight.setdefault(line[0], line[1])


    with open(uu_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            uu_neigh.setdefault(line[0], line[1])

    with open(ii_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            ii_neigh.setdefault(line[0], line[1])


    with open(ui_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            ui_neigh.setdefault(line[0], line[1])


    return uu_neigh, ui_neigh, iu_neigh, ii_neigh, uu_edge_weight, ui_edge_weight


def gen_start_weight():
    args = Parser.Define_Params()
    unum = args.u_l
    inum = args.i_l
    embed_dim = args.embed_dim
    u2e = nn.Embedding(unum, embed_dim)
    i2e = nn.Embedding(inum, embed_dim)



    file_path = args.start_node_embedding
    f = open(file_path, 'r+')
    f.truncate()
    f.close()
    num = 1
    count = 1
    with open(file_path, 'a') as file:
        for single_weight in list(u2e.weight):

            file.write('u' + str(num) + '  ' + str(single_weight.detach().numpy().tolist())[1: -1] + '\n')
            num = num + 1

    with open(file_path, 'a') as file:
        for single1_weight in list(i2e.weight):
            file.write('i' + str(count) + '  ' + str(single1_weight.detach().numpy().tolist())[1: -1] + '\n')
            count = count + 1
    # print('jinali', count)



gen_start_weight()


