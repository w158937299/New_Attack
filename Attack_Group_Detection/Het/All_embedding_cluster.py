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


# file_path = './edges/testgroup.txt'
# SpamGroup = []
# with open(file_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         users = line.strip('\n')[1: -1].split(',')
#
#         temp = []
#         for user in users:
#             if int(user) <= 5055:
#                 temp.append(user)
#         if len(temp) != 0 :
#             SpamGroup.append(temp)
# num = 0
# k = 0
# for i in SpamGroup:
#     if len(i) > 1:
#         num = num + 1
#     else:
#         k = k + 1



file_path = './edges/testgroup.txt'
# f = open(file_path, 'r+')
# f.truncate()
# f.close()
Spam_group = []
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')[1: -1].split(', ')
        temp = []
        for user in line:
            if user[1] == ' ':
                user = user.replace(' ','')
            user = user[1: -1]
            temp.append(user)
        Spam_group.append(temp)

print(Spam_group)
file_path = './edges/All_embedding_group.txt'
with open(file_path, 'a') as file:
    for group in Spam_group:
        file.write(str(group) + '\n')





















