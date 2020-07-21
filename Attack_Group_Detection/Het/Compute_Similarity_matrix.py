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
from getData import get_last_data
from model import HetAD
from model import cross_entropy_loss
from data_input import get_train_data
from Mean_shift_next import get_needed_cut


def get_OJLD_distance(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    op2 = np.linalg.norm(list1 - list2)
    op2 = round(op2, 4)
    return op2

def get_MHD_distance(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    op2 = np.linalg.norm(list1 - list2, ord = 1)
    op2 = round(op2, 4)
    return op2

def get_COS_distance(list1, list2):
    vector1 = np.array(list1)
    vector2 = np.array(list2)
    op7 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    op7 = round(op7, 4)
    return op7


args = Parser.Define_Params()
file_path = args.new_node_embedding
UserEmbedding = {}
file_path = './edges/new_node_embedding1.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split('  ')
        if line[0][0] == 'u':
            embeddings = list(map(float, line[1].split(', ')))
            UserEmbedding.setdefault(line[0], embeddings)
        elif line[0][0] == 'i':
            embeddings = list(map(float, line[1].split(', ')))
            UserEmbedding.setdefault(line[0], embeddings)


OJLD_file_path = args.OJLD
MHD_file_path = args.MHD
COS_file_path = args.COS
OJLD_file_path = './edges/OJLD_Similar_matrix1.txt'

UserEdge = []
i = 0


with open(OJLD_file_path, 'a') as file:
    for User1 in UserEmbedding:
        temp = []
        i = i + 1
        for User2 in UserEmbedding:
            d = get_OJLD_distance(UserEmbedding[User1], UserEmbedding[User2])
            # d = get_MHD_distance(UserEmbedding[User1], UserEmbedding[User2])
            # d = get_COS_distance(UserEmbedding[User1], UserEmbedding[User2])
            temp.append(d)
        file.write(str(User1) + '  ' + str(temp) + '\n')
        UserEdge.append(temp)
        print(i)

print(len(UserEdge))


#
# needed_group = get_needed_cut(30)
# for group in needed_group:
#
#     print(group)














