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
import matplotlib.pyplot as plt
import random

import math

args = Parser.Define_Params()
whole_file_path = 'C:/Users/wjy666/PycharmProjects/ADEmbedding/dataset/AMAZON.txt'

UserAttack = {}
Sample = {}
with open(whole_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        user = line[0]
        attack = line[3]
        if attack == str(-1):
            Sample.setdefault(user, attack)
        if user not in UserAttack:
            UserAttack.setdefault(user, attack)

Sample = len(Sample)
print(Sample)
SpamGroup = []
file_path = '../edges/ScanAMSpamGroupatt.txt'
# file_path = '../edges/AMSpamGroup.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').split('  ')[1][1: -1].split(', ')
        SpamGroup.append(line)
GSBCSpamGroup = []
file_path = '../edges/newSP.txt'
# file_path = '../edges/SP.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')[2: -1]
        GSBCSpamGroup.append(line)

def indicate_2000(SpamGroups):
    num = 0
    group2000 = []
    ll_user = []
    ll_ratio = []
    shuchuuser = []
    num1 = -1
    for group in SpamGroups:
        num = num + len(group)
        if num < 1937:
            group2000.append(group)
            for user in group:
                ll_user.append(user[1: -1])
                num1 = num1 + 1
                shuchuuser.append(num1)

    IsAttack = 0
    num = 0
    for group in group2000:
        for user in group:
            num = num + 1
            if int(UserAttack[str(user)[1: -1]]) == -1:
                IsAttack = IsAttack + 1
    other = 0
    other1 = 0
    for user in ll_user:
        if int(UserAttack[user]) == -1:
            other = other + 1
        ll_ratio.append(other / 1937)

    return IsAttack/num, ll_ratio, shuchuuser

def indicate_2000s(SpamGroups):
    num = 0
    num1 = -1
    group2000 = []
    ll_user = []
    shuchuuser = []
    ll_ratio = []
    for group in SpamGroups:
        num = num + len(group)
        if num < 1937:
            group2000.append(group)
            for user in group:
                ll_user.append(user)
                num1 = num1 + 1
                shuchuuser.append(num1)
    IsAttack = 0
    num = 0
    for group in group2000:
        for user in group:
            num = num + 1
            if int(UserAttack[str(user)]) == -1:
                IsAttack = IsAttack + 1
    other = 0
    other1 = 0
    for user in ll_user:
        if int(UserAttack[user]) == -1:
            other = other + 1
        ll_ratio.append(other/1937)
    return  IsAttack/num, ll_ratio, shuchuuser


# ratio, ll_ratio, ll_user = indicate_2000(SpamGroup)
# ratio1, ll_ratio1, ll_user1 = indicate_2000s(GSBCSpamGroup)
# # print(len(ll_ratio))
# # print(len(ll_ratio1))
# # print(len(ll_user))
# # print(len(ll_user1))
# length = min(len(ll_user),len(ll_user1))
# print(ll_ratio)
# print(ll_ratio1)
# ll_ratio1 = ll_ratio1[:length]
# ll_user1 = ll_user1[:length]
# print(ll_user1)
# print(ll_ratio1)
# print('HetGNN',ratio)
# print('GSBC',ratio1)
# plt.plot(ll_user, ll_ratio, color = 'red', label = 'HetGNN')
# plt.plot(ll_user1, ll_ratio1, color = 'blue', label = 'GSBC')
# plt.xlabel('topKUser')
# plt.ylabel('Recall')
# plt.title('Recall')
# plt.legend(loc='best')
# plt.show()
#
# exit(0)


HetP = [0]
HetR = [0]
GSBCP = [0]
GSBCR = [0]
topk = [0]
for k in np.arange(1, 600):
    dummy = 0
    real = 0
    count = 0
    topK = SpamGroup[:k]
    for group in topK:
        for user in group:
            count = count + 1
            if UserAttack[user[1:-1]] == str(-1):
                dummy = dummy + 1
            else:
                real = real + 1
    recall = dummy/Sample
    precision = dummy/count
    HetP.append(precision)
    HetR.append(recall)
    # print('top',k,'    HetGNN_precision', precision, '   HetGNN_recall', recall)

    dummy = 0
    real = 0
    count = 0
    topK = GSBCSpamGroup[:k]
    for group in topK:
        for user in group:
            count = count + 1
            if UserAttack[user] == str(-1):
                dummy = dummy + 1
            else:
                real = real + 1

    recall = dummy/Sample
    precision = dummy/count
    # print('top',k,'    GSBC_precision', precision, '    GSBC_recall', recall)
    GSBCP.append(precision)
    GSBCR.append(recall)
    topk.append(k)

plt.plot(topk, HetP, color = 'red', label = 'HetGNN')
plt.plot(topk, GSBCP, color = 'blue', label = 'GSBC')
plt.xlabel('topK')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend(loc='best')
plt.show()

plt.plot(topk, HetR, color = 'red', label = 'HetGNN')
plt.plot(topk, GSBCR, color = 'blue', label = 'GSBC')
plt.xlabel('topK')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend(loc='best')
plt.show()






