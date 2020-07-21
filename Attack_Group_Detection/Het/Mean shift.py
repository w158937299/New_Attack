import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import Parser
import time
# #############################################################################
# Generate sample data  造用于聚类的数据

args = Parser.Define_Params()
node_file_path = args.new_node_embedding
node_file_path = './edges/new_node_embedding3.txt'
X = []
X1 = []
with open(node_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:

        if line[0] == 'u':
            line = line.strip('\n').split('  ')
            val = line[1].split(', ')
            val = list(map(float, val))
            X.append(val)

        elif line[0] == 'i':
            line = line.strip('\n').split('  ')
            val = line[1].split(', ')
            val = list(map(float, val))
            X.append(val)


##产生随机数据的中心
# centers = [[1, 1], [-1, -1], [1, -1]]
# ##产生的数据个数
# n_samples=10000
# ##生产数据
# X, _ = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6,
#                   random_state =0)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
#
# for i in np.arange(0.0004, 0.1, 0.001):
#     bandwidth = estimate_bandwidth(X, quantile=i)
#     print(bandwidth)
# exit(0)
# for i in np.arange(100, 5000, 100):
#     bandwidth = estimate_bandwidth(X, quantile=0.0004)
#     print(bandwidth)
#
# exit(0)

# n_samples对band_width影响不大，主要是quantile！
# for i in np.arange(1.9, 2.5, 0.1):

# quantile： 选取每队样本的距离的quantile分位数作为返回值，目前可以估计出来的bandwidth值最小为2.3213
# for i in np.arange(2, 2.5, 0.1):
#     ms = MeanShift(bandwidth= i, bin_seeding=False)
#     ms.fit(X)  # 训练模型
#     labels = ms.labels_  # 所有点的的labels
#     cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
#     # 用来设定初始核的位置参数的生成方式，default False,默认采用所有点的
#     # 位置平均
#     labels_unique = np.unique(labels)
#     n_clusters_ = len(labels_unique)
#     print(n_clusters_)
# exit(0)






# for i in np.arange(0.1,1,0.1):
#     bandwidth = estimate_bandwidth(X, quantile=i)
#     print(i, bandwidth)
# exit(0)
# 2.3213   176
#2.5173



# 2.516
# 2.5172
# 3.1
# for i in np.arange(2.36,2.39,0.02):
start = time.clock()
# 2.31
ms = MeanShift(bandwidth= 2.31, bin_seeding=False)
ms.fit(X)  # 训练模型
labels = ms.labels_  # 所有点的的labels
cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
# 用来设定初始核的位置参数的生成方式，default False,默认采用所有点的
# 位置平均
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
end = time.clock()
costs = end - start
print(2.35,'it ',costs,'s   ','generate',n_clusters_)

double_group = []
Cluster_group = {}
for i in range(0, len(labels)):
    if labels[i] not in Cluster_group:
        Cluster_group.setdefault(labels[i], [i+1])
    else:
        temp = Cluster_group[labels[i]]
        temp.append(i+1)
        Cluster_group[labels[i]] = temp

for group in Cluster_group.values():
    if len(group) > 1:
        double_group.append(group)
# print('more than 1', len(double_group))



file_path = './edges/testgroup.txt'
f = open(file_path, 'r+')
f.truncate()
f.close()
with open(file_path, 'a') as f:
    for label in Cluster_group:
        f.write(str(Cluster_group[label]) + '\n')
print("number of estimated clusters : %d" % n_clusters_)
double_group = []
for group in Cluster_group.values():
    if len(group) > 1:
        double_group.append(group)

file_path = './edges/testgroup.txt'
# f = open(file_path, 'r+')
# f.truncate()
# f.close()
SpamGroup = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        users = line.strip('\n')[1: -1].split(',')

        temp = []
        for user in users:
            if int(user) <= 5055:
                temp.append(user)
        SpamGroup.append(temp)
num = 0
file_path = './edges/testgroup.txt'
f = open(file_path, 'r+')
f.truncate()
f.close()
with open(file_path, 'a') as file:
    for i in SpamGroup:
        file.write(str(i) + '\n')
        if len(i) > 1:
            num = num + 1
    print('delete 5055+ item',num)


# double_group = []
# with open('./edges/candidate_group.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip('\n')[1:-1].split(', ')
#         if len(line) > 1:
#             double_group.append(line)
# print(len(double_group))
# exit(0)

# #############################################################################
# Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     print(cluster_center)
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
