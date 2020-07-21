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
from tools import sample_neigh
if __name__ == '__main__':
    args = Parser.Define_Params()

    # format_uu_edge_filepath = args.format_UUedge_weight
    format_uu_edge_filepath = './edges/format_UUedge1.txt'
    format_ii_edge_filepath = './edges/format_IIedge1.txt'
    ui_edge_filepath = args.UIedge_weight
    uu_neigh_filepath = args.uu_neigh
    uu_neigh_filepath = './edges/uu_neigh1.txt'
    ii_neigh_filepath = './edges/ii_neigh1.txt'

    ui_neigh_filepath = args.ui_neigh
    iu_neigh_filepath = args.iu_neigh
    uu_edge_weight = {}

    ii_edge_weight = {}
    uu_neigh = {}
    ui_neigh = {}
    iu_neigh = {}
    ii_neigh = {}
    u_l = args.u_l
    i_l = args.i_l
    # walk_length = args.walk_length
    walk_length = 40

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

    with open(format_ii_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('  ')
            ii_edge_weight.setdefault(line[0], line[1])



    with open(uu_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            uu_neigh.setdefault(line[0], line[1])


    with open(ui_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            ui_neigh.setdefault(line[0], line[1])

    with open(ii_neigh_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split('   ')
            ii_neigh.setdefault(line[0], line[1])

    # gen_random_walk_filepath = args.het_random_walk_fliepath
    # gen_random_walk_filepath = './edges/het_random_walk1.txt'
    gen_random_walk_filepath = './edges/het_random_walk2.txt'
    # 无向图管他什么重启呢都一样
    f = open(gen_random_walk_filepath, 'r+')
    f.truncate()
    f.close()

    with open(gen_random_walk_filepath, 'a') as file:
        for user in uu_neigh:
            print(user)
            walk = 0
            temp_u = 0
            temp_i = 0
            curnode = user
            het_rand = []
            het_rand.append(user)
            while walk < walk_length:
                if str(curnode)[0] == 'u':

                    type = random.randint(1, 2)
                    if type == 1 and curnode in ui_neigh and temp_i < 20:
                        temp = ui_neigh[curnode][1:-1].split(',')
                        temp = random.choice(list(temp))[2:-1]

                        if temp[0] != 'i':
                            temp = 'i' + temp
                        curnode = temp
                        het_rand.append(curnode)
                        walk = walk + 1
                        temp_i = temp_i + 1
                    elif type == 2 and curnode in uu_neigh and temp_u < 20:

                        temp_uu_neigh = uu_neigh[curnode][1:-1].split(',')
                        temp_uu_neigh = list(map(lambda x:x[2:-1], temp_uu_neigh))
                        temp_uu_neigh[0] = 'u' + temp_uu_neigh[0]
                        tables = create_table(curnode, temp_uu_neigh, 1)

                        neigh_node = sample_neigh(tables)

                        curnode = neigh_node

                        het_rand.append(curnode)
                        # back_p = 1 / len(uu_neigh[user])
                        walk = walk + 1
                        temp_u = temp_u + 1
                elif str(curnode)[0] == 'i':
                    type = random.randint(1, 2)

                    if curnode in iu_neigh and type == 1 and temp_u < 20:
                        tempu = random.choice(iu_neigh[curnode])
                        curnode = tempu
                        het_rand.append(curnode)
                        walk = walk + 1
                        temp_u = temp_u + 1
                    elif curnode in ii_neigh and type == 2 and temp_i < 20:
                        temp_ii_neigh = ii_neigh[curnode][1:-1].split(',')
                        temp_ii_neigh = list(map(lambda x: x[2:-1], temp_ii_neigh))
                        temp_ii_neigh[0] = 'i' + temp_ii_neigh[0]
                        tables = create_table(curnode, temp_ii_neigh, 2)

                        neigh_node = sample_neigh(tables)

                        curnode = neigh_node

                        het_rand.append(curnode)
                        # back_p = 1 / len(uu_neigh[user])
                        walk = walk + 1
                        temp_i = temp_i + 1

            file.write(str(het_rand) + '\n')







