# class test():
#
#         # self.b = 10
#     def a(self):
#         self.b = 1
#     def a1():
#         bc = 2
#
#
#
# a = test()
# a.a()
# print(a.b)
import math

import numpy as np
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
a = np.array([1,2,3,4,5,6])
b = 3
print(a>=b)
a = [0.5,0.6,-0.3]
c = [1, -1 ,-1]
# for b in range(len(a)):
#     if a[b]> 0:
#         a[b] = 1
#     else:
#         a[b] = -1
# for i in range(len(a)):
#     c[i] = (a[i] == c[i])
# print(c)
#
# a = 0.35
# b = 0.44
# print('a%.4ffff%.4f'%(a, b))

# a = sqrt(mean_squared_error(a, c))
# print(a)
# b= [1,2,3,4]
# tensor= b.clone().detach()
# print(tensor)
#
# a = torch.tensor([2,2,3,4])
# b = torch.tensor([2,2,3,4])
# c = torch.tensor([2,2,3,4])
# d = torch.tensor([2,2,3,4])
# print((a+b+c+d)/4)
# import os
# os.makedirs("./models/checkpoint")

# a = torch.tensor([ 1.3456e-01,  9.9333e-01,  6.5492e-01, -1.5778e+00, -1.0226e+00,
#           1.2524e-01,  5.1094e-02, -2.6239e-01,  5.6332e-01,  6.6113e-01,
#          -2.6338e-01,  7.7566e-01, -7.6876e-01,  1.2512e+00, -7.2119e-01,
#          -1.4337e+00,  1.3029e+00,  1.0104e+00,  6.8115e-01, -1.0170e+00,
#          -6.8627e-01, -2.0907e+00,  3.1111e-01, -2.4406e-01,  1.5166e+00,
#           1.5340e+00, -3.7050e-01,  4.8469e-01,  2.1151e+00,  3.9957e-01,
#           9.1154e-01, -2.7420e-01])
# b = torch.ones([1, 32]).squeeze()
# print(a+b)
# print(b)
# print(a.float())
#
# a = torch.empty(5, 5, dtype=float)
# for i in range(5):
#     a[i] = torch.DoubleTensor([1,2,3,4,5])
# print(a)

a = ['1','2','3','4']
b = ['u', 'u', 'u', 'u']
b = a + b
print(b)
# print(math.tanh(0.3))
#
# import numpy as np
# import scipy.stats
# p=np.asarray([0.65,0.25,0.07,0.03])
# q=np.array([0.6,0.25,0.1,0.05])
# q2=np.array([0.1,0.2,0.3,0.4])
# p = ['5', '5', '5', '5', '5']
# q = ['1', '1', '1', '1', '5']
# # p = ['5', '4', '4', '5', '5', '5', '3']
# # q = ['3', '3', '3', '4', '3', '3', '3']
#
# p = np.array(list(map(int, p)))
# #p = p/5
#
# q = np.array(list(map(int, q)))
# # q = q/5
#
# def JS_divergence(p,q):
#     M=(p+q)/2
#     return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
#
# print(JS_divergence(q,q)) # 0.0
# print(JS_divergence(p,p)) # 0.0
# print(12*(JS_divergence(p,q))) # 0.0
# #
# a = [1,2,3,4]
# b = [5,6,7,8]
# for i,j in zip(a,b):
#     print(i, j)

#
# a = []
# a.append([0] * 2)
# a.append([0] * 3)
# a.append([0] * 4)
#
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         a[i][j] = np.zeros(4)
# print(a)


for i in range(1,3):
    print(i)

a = '2019-10-8'
b = '2019-9-29'
print(a < b)
a = [1,23,3,44,5,5,6]
print(a[:3])


a = [1,2,3,4,5]
b = 1
if b in a:
    print('111')


