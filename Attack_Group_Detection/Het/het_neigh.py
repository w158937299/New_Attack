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



if __name__ == '__main__':
    args = Parser.Define_Params()
    format_uu_edge_filepath = args.format_UUedge_weight
    format_uu_edge_filepath = './edges/format_UUedge1.txt'
    format_ii_edge_filepath = './edges/format_IIedge1.txt'
    ui_edge_filepath = args.UIedge_weight
    print(format_uu_edge_filepath)
    print(ui_edge_filepath)
    ui_neigh_list = {}
    uu_neigh_list = {}
    ii_neigh_list = {}
    with open(ui_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            line = line[0].split('i')

            item = 'i' + line[1]
            if line[0] not in ui_neigh_list:
                ui_neigh_list.setdefault(line[0], [item])
            else:
                temp = ui_neigh_list[line[0]]
                temp.append(item)
                ui_neigh_list[line[0]] = temp


    with open(format_uu_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            line = line[0].split('u')

            curnode = 'u' + line[1]
            neigh_node = 'u' + line[2]
            if curnode not in uu_neigh_list:
                uu_neigh_list.setdefault(curnode, [neigh_node])
            else:
                temp = uu_neigh_list[curnode]
                temp.append(neigh_node)
                uu_neigh_list[curnode] = temp

    with open(format_ii_edge_filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            line = line[0].split('i')

            curnode = 'i' + line[1]
            neigh_node = 'i' + line[2]
            if curnode not in ii_neigh_list:
                ii_neigh_list.setdefault(curnode, [neigh_node])
            else:
                temp = ii_neigh_list[curnode]
                temp.append(neigh_node)
                ii_neigh_list[curnode] = temp

    uu_neigh_filepath = args.uu_neigh
    uu_neigh_filepath = './edges/uu_neigh1.txt'
    ui_neigh_filepath = args.ui_neigh
    ii_neigh_filepath = './edges/ii_neigh1.txt'


    f = open(uu_neigh_filepath, 'r+')
    f.truncate()
    f.close()

    f = open(ii_neigh_filepath, 'r+')
    f.truncate()
    f.close()

    # with open(uu_neigh_filepath, 'a') as file:
    #     for user in uu_neigh_list:
    #         file.write(user + '   ')
    #         file.write(str(uu_neigh_list[user]) + '\n')
    with open(ii_neigh_filepath, 'a') as file:
        for user in ii_neigh_list:
            file.write(user + '   ')
            file.write(str(ii_neigh_list[user]) + '\n')




    with open(uu_neigh_filepath, 'a') as file:
        for user in uu_neigh_list:
            file.write(user + '   ')
            file.write(str(uu_neigh_list[user]) + '\n')

    # with open(ui_neigh_filepath, 'a') as file:
    #     for user in ui_neigh_list:
    #         file.write(user + '   ')
    #         file.write(str(ui_neigh_list[user]) + '\n')
    # print(uu_neigh_filepath)
    # print(ui_neigh_filepath)




