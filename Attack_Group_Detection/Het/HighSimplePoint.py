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
import sys
from sklearn.cluster import KMeans

# import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
#
# # X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，对应x和y轴，共4个簇，
# # 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
#                   cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
# print(X)
# from sklearn.cluster import KMeans
#
# y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()


args = Parser.Define_Params()
node_file_path = args.new_node_embedding

X = []
with open(node_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:

        if line[0] == 'u':
            line = line.strip('\n').split('  ')
            val = line[1].split(', ')
            val = list(map(float, val))
            X.append(val)
        else:
            break
X = np.array(X)


double_point = []
file_path = './edges/last_candidate_group30.txt'
Simple_point = []
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')[1: -1].split(', ')
        if len(line) == 1:
            Simple_point.append(line[0][1: -1])
        else:
            temp = []
            for i in line:
                i = i[1: -1]
                temp.append(i)
            # double_point.append()
            double_point.append(temp)


Simple_point_weight = []
for point in Simple_point:

    Simple_point_weight.append(X[int(point[1: -1]) - 1])

Simple_point_weight = np.array(Simple_point_weight)

y_pred = KMeans(n_clusters=len(Simple_point)//2, random_state=9).fit_predict(Simple_point_weight)
other_group = {}
num = 0
for label in y_pred:
    num = num + 1
    if label not in other_group:
        other_group.setdefault(label, [Simple_point[num - 1]])
    else:
        temp = other_group[label]
        temp.append(Simple_point[num - 1])
        other_group[label] = temp

file_path = './edges/last_candidate_group30.txt'
f = open(file_path, 'r+')
f.truncate()
f.close()
with open(file_path, 'a') as f:
    for group in list(other_group.values()):
        if len(group) > 1:
            f.write(str(group) + '\n')
    for group1 in double_point:
        if len(group1) > 1:
            f.write(str(group1) + '\n')
# print(other_group.values())
# print(double_point)




















