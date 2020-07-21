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



args = Parser.Define_Params()

format_uu_edge_filepath = args.format_UUedge_weight
format_uu_edge_filepath = './edges/format_UUedge1.txt'
format_ii_edge_filepath = './edges/format_IIedge1.txt'
ui_edge_filepath = args.UIedge_weight
uu_neigh_filepath = args.uu_neigh
uu_neigh_filepath = './edges/uu_neigh1.txt'
ui_neigh_filepath = args.ui_neigh
uu_edge_weight = {}
ui_edge_weight = {}
ii_edge_weight = {}
uu_neigh = {}
ui_neigh = {}
het_random_walk_fliepath = args.het_random_walk_fliepath

windows_length = args.windows_len

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

with open(ui_edge_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(' ')
        ui_edge_weight.setdefault(line[0], line[1])

random_walk_filepath = args.het_random_walk_fliepath
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



def create_table(curnode, edges, types):
    tables = {}
    if types == 1:
        edges = str(edges).strip('[').strip(']').split(', ')
        sum = 0.0
        for edge in edges:
            edge = edge[1:-1]
            uu = curnode + edge
            if uu in uu_edge_weight:
                sum = sum + float(uu_edge_weight[uu])
            else:
                uu1 = edge + curnode
                sum = sum + float(uu_edge_weight[uu1])
            tables.setdefault(sum, edge)

    elif types == 2:
        edges = str(edges).strip('[').strip(']').split(', ')
        sum = 0.0
        for edge in edges:
            edge = edge[1:-1]
            ui = curnode + edge
            if ui in ii_edge_weight:
                sum = sum + float(ii_edge_weight[ui])
            else:
                iu = edge + curnode
                sum = sum + float(ii_edge_weight[iu])
            tables.setdefault(sum, edge)

    return tables

def sample_neigh(tables):
    des = 0
    Randfloat = random.random()

    for i in range(len(tables)):
        if Randfloat < list(tables.keys())[i]:
            des = list(tables.values())[i]
            break
    if des == 0:
        des = list(tables.values())[-1]
    return des

# def get_het_random_walk(uu_neigh, ui_neigh, uu_edge, ui_edge, walk_length):
#     with open(het_random_walk_fliepath, 'a') as f:

def Compute_Sample_ratio(random_walk_list):
    Compute_ratio = {}
    Compute_ratio.setdefault('uu', 0)
    Compute_ratio.setdefault('ui', 0)
    Compute_ratio.setdefault('iu', 0)
    Compute_ratio.setdefault('ii', 0)
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for random_walk in random_walk_list:
        for j in range(len(random_walk)):
            curnode = random_walk[j]
            if curnode[0] == 'u':
                for k in range(j - windows_length, j + windows_length + 1):
                    if k != j and k < len(random_walk) and k:
                        neigh_node = random_walk[k]
                        if neigh_node[0] == 'u':
                            temp1 = temp1 + 1
                            Compute_ratio['uu'] = temp1
                        elif neigh_node[0] == 'i':
                            temp2 = temp2 + 1
                            Compute_ratio['ui'] = temp2
            elif curnode[0] == 'i':
                for k in range(j - windows_length, j + windows_length + 1):
                    if k != j and k <len(random_walk) and k:
                        neigh_node = random_walk[k]
                        if neigh_node[0] == 'u':
                            temp3 = temp3 + 1
                            Compute_ratio['iu'] = temp3
                        elif neigh_node[0] == 'i':
                            temp4 = temp4 + 1
                            Compute_ratio['ii'] = temp4
    count = 0
    for i in Compute_ratio.values():
        count = count + i
    Compute_ratio['uu'] = Compute_ratio['uu']/(count*40)
    Compute_ratio['ui'] = Compute_ratio['ui']/(count*40)
    Compute_ratio['iu'] = Compute_ratio['iu']/(count*40)
    Compute_ratio['ii'] = Compute_ratio['ii']/(count*40)
    return Compute_ratio















