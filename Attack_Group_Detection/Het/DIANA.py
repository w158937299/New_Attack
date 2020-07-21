# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:02:59 2018

@author: Divyang Vashi
"""
import time

import Parser
import numpy as np;
import pandas as pd
from Mean_shift_next import get_needed_cut


class DianaClustering:
    def __init__(self, n_samples, n_features):
        '''
        constructor of the class, it takes the main data frame as input
        '''

        self.n_samples = n_samples
        self.n_features = n_features

    def fit(self, num, similarity_matrix):
        '''
        this method uses the main Divisive Analysis algorithm to do the clustering

        arguements
        ----------
        n_clusters - integer
                     number of clusters we want

        returns
        -------
        cluster_labels - numpy array
                         an array where cluster number of a sample corrosponding to
                         the same index is stored
        '''
        #similarity_matrix = DistanceMatrix(self.data)  # similarity matrix of the data
          # similarity matrix of the data
        similarity_matrix = np.array(similarity_matrix)

        clusters = [list(range(self.n_samples))]  # list of clusters, initially the whole dataset is a single cluster

        while True:
            count = 0
            c_diameters = [np.max(similarity_matrix[cluster][:, cluster]) for cluster in clusters]  # cluster diameters

            max_cluster_dia = np.argmax(c_diameters)  # maximum cluster diameter
            max_difference_index = np.argmax(
                np.mean(similarity_matrix[clusters[max_cluster_dia]][:, clusters[max_cluster_dia]], axis=1))
            splinters = [clusters[max_cluster_dia][max_difference_index]]  # spinter group
            last_clusters = clusters[max_cluster_dia]
            del last_clusters[max_difference_index]
            while True:
                split = False
                for j in range(len(last_clusters))[::-1]:
                    splinter_distances = similarity_matrix[last_clusters[j], splinters]
                    last_distances = similarity_matrix[last_clusters[j], np.delete(last_clusters, j, axis=0)]
                    if np.mean(splinter_distances) <= np.mean(last_distances):
                        splinters.append(last_clusters[j])
                        del last_clusters[j]
                        split = True
                        break
                if split == False:
                    break
            del clusters[max_cluster_dia]
            clusters.append(splinters)
            clusters.append(last_clusters)
            for cluster in clusters:
                if len(cluster) > num:
                    count =  count + 1
            if count == 0:
                break


        cluster_labels = np.zeros(self.n_samples)
        for i in range(len(clusters)):
            cluster_labels[clusters[i]] = i

        return cluster_labels



def get_OJLD_distance(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    op2 = np.linalg.norm(list1 - list2)
    op2 = round(op2, 6)
    return op2



# # 计算每个点的平均相异度
# def avg_dissim_within_group_element(ele, element_list):
#     max_diameter = -np.inf
#     sum_dissm = 0
#     # 计算平均相异度
#     for i in element_list:
#         sum_dissm += dissimilarity_matrix[ele][i]
#         if (dissimilarity_matrix[ele][i] > max_diameter):
#             max_diameter = dissimilarity_matrix[ele][i]
#     if (len(element_list) > 1):
#         avg = sum_dissm / (len(element_list) - 1)
#     else:
#         avg = 0
#
#     return avg
#
#
# def avg_dissim_across_group_element(ele, main_list, splinter_list):
#     if len(splinter_list) == 0:
#         return 0
#     sum_dissm = 0
#     for j in splinter_list:
#         sum_dissm = sum_dissm + dissimilarity_matrix[ele][j]
#     avg = sum_dissm / (len(splinter_list))
#     return avg
#
#
# def splinter(main_list, splinter_group):
#     most_dissm_object_value = -np.inf
#
#     most_dissm_object_index = None
#     for ele in main_list:
#         # 每一个元素在原本的list平均相异度
#         x = avg_dissim_within_group_element(ele, main_list)
#
#         # 计算每一个元素在分割出来的群组的平均相异度
#         y = avg_dissim_across_group_element(ele, main_list, splinter_group)
#
#         diff = x - y
#         # 找最近的点，然后记录index
#         if diff > most_dissm_object_value:
#             most_dissm_object_value = diff
#             most_dissm_object_index = ele
#
#     if (most_dissm_object_value > 0):
#         return (most_dissm_object_index, 1)
#     else:
#         return (-1, -1)
#
#
# def split(element_list):
#     main_list = element_list
#     splinter_group = []
#     (most_dissm_object_index, flag) = splinter(main_list, splinter_group)
#     while (flag > 0):
#         # 从原本的list去掉那个被分割的组
#         main_list.remove(most_dissm_object_index)
#         splinter_group.append(most_dissm_object_index)
#         # 当平均相异度的值差没有了，结束迭代
#         (most_dissm_object_index, flag) = splinter(element_list, splinter_group)
#
#     return (main_list, splinter_group)
#
# # 找最大相似度的簇
# def max_diameter(cluster_list):
#     max_diameter_cluster_index = None
#     max_diameter_cluster_value = -np.inf
#     index = 0
#     for element_list in cluster_list:
#         for i in element_list:
#             for j in element_list:
#                 # 找到最大的那个相似度的簇
#                 if dissimilarity_matrix[i][j] > max_diameter_cluster_value:
#                     max_diameter_cluster_value = dissimilarity_matrix[i][j]
#                     max_diameter_cluster_index = index
#
#         index += 1
#
#     if (max_diameter_cluster_value <= 0):
#         return -1
#     # 就是在每个簇中找到最大相似的簇。确保是最大的。
#     return max_diameter_cluster_index
#
# def break_con(current_clusters):
#     label = False
#     for cluster in current_clusters:
#         if len(cluster) >= 30:
#             label = True
#             break
#     return label




args = Parser.Define_Params()
file_path = args.OJLD

all_elements = []
mat = []
# with open(file_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip('\n').split('  ')
#         all_elements.append(line[0])
#
#         val = line[1][1: -1].split(', ')
#         new_val = list(map(float, val))
#         mat.append(new_val)

args = Parser.Define_Params()
file_path = args.new_node_embedding
file_path = './edges/new_node_embedding1.txt'
file_path = './edges/new_node_embedding2.txt'
file_path = './edges/new_node_embedding3.txt'
UserEmbedding = {}

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split('  ')
        if line[0][0] == 'u':
            embeddings = list(map(float, line[1].split(', ')))
            UserEmbedding.setdefault(line[0], embeddings)
        elif line[0][0] == 'i':
            break

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


def diana_cluster(needed_group, num):
    last_candidate_group = []
    if num != 2:
        for group in needed_group:
            UserEdge = []
            all_elements = []
            last_dict = {}
            for singles in group:
                if singles[0] == "'":
                    singles = singles[1: -1]
                singles = singles.strip()
                singles = 'u' + str(singles)
                all_elements.append(singles)
                temp = []
                for other_singles in group:
                    # 每次实验的时候看看这里是不是需要更改

                    if other_singles[0] == "'":
                        other_singles = other_singles[1: -1]
                    other_singles = other_singles.strip()
                    other_singles = 'u' + other_singles



                    distance = get_OJLD_distance(UserEmbedding[singles], UserEmbedding[other_singles])
                    temp.append(distance)
                UserEdge.append(temp)

            diana = DianaClustering(len(group), args.embed_dim)
            cluster_label = diana.fit(num, UserEdge)

            for i in range(len(cluster_label)):
                label = cluster_label[i]
                if label not in last_dict:
                    last_dict.setdefault(label, [i])
                else:
                    temp = last_dict[label]
                    temp.append(i)
                    last_dict[label] = temp

            for last in last_dict:
                temp1 = []
                for index in last_dict[last]:
                    temp1.append(group[index])
                last_candidate_group.append(temp1)
    elif num == 2:
        UserEdge = []
        all_elements = []
        last_dict = {}
        for singles in needed_group:
            if singles[0] == "'":
                singles = singles[1: -1]
            singles = 'u' + str(singles)
            all_elements.append(singles)
            temp = []
            for other_singles in needed_group:
                if other_singles[0] == "'":
                    other_singles = other_singles[1: -1]
                other_singles = 'u' + other_singles

                distance = get_OJLD_distance(UserEmbedding[singles], UserEmbedding[other_singles])
                temp.append(distance)
            UserEdge.append(temp)

        diana = DianaClustering(len(needed_group), args.embed_dim)
        cluster_label = diana.fit(num, UserEdge)

        for i in range(len(cluster_label)):
            label = cluster_label[i]
            if label not in last_dict:
                last_dict.setdefault(label, [i])
            else:
                temp = last_dict[label]
                temp.append(i)
                last_dict[label] = temp

        for last in last_dict:
            temp1 = []
            for index in last_dict[last]:
                temp1.append(needed_group[index])
            last_candidate_group.append(temp1)
    return last_candidate_group

# file_path = './edges/All_embedding_group.txt'
file_path = './edges/testgroup1.txt'
file_path = './edges/Candidate_group_att.txt'
# file_path = './edges/testgroup.txt'



needed_group, finish_group = get_needed_cut(30,file_path)

SimplePoint = []
NormalGroup = []
# ', '
for group in finish_group:
    if len(group) == 1:
        SimplePoint.append(group[0])
    else:
        NormalGroup.append(group)
# other_group = diana_cluster(SimplePoint, 2)





# minnum = np.inf
# index = 0
# other_group = []
# for i in range(len(SimplePoint)):
#     print(len(SimplePoint))
#     print(i)
#     for j in range(i + 1, len(SimplePoint)):
#         if minnum > get_OJLD_distance(UserEmbedding['u' + SimplePoint[i][0]], UserEmbedding['u' + SimplePoint[j][0]]):
#             minnum = get_OJLD_distance(UserEmbedding['u' + SimplePoint[i][0]], UserEmbedding['u' + SimplePoint[j][0]])
#             index = j
#     other_group.append([SimplePoint[i][0], SimplePoint[index][0]])
#     del SimplePoint[index]
#
# print(other_group)


last_candidate_group = diana_cluster(needed_group, 30)


file_name = './edges/Candidate_group_att.txt'
f = open(file_name, 'r+')
f.truncate()
f.close()

# file_name = './edges/last_testgroup.txt'
with open(file_name, 'a') as f:
    for i in last_candidate_group:
        f.write(str(i) + '\n')
    for j in NormalGroup:
        f.write(str(j) + '\n')
    for k in SimplePoint:
        f.write(str([k]) + '\n')
    # for k in other_group:
    #     f.write(str(k) + '\n')




    # 其中一种DIANA方法
    # num_clusters = 0
    # mat = np.array(UserEdge)
    # # mat = np.array([[0, 2, 6, 10, 9], [2, 0, 5, 9, 8], [6, 5, 0, 4, 5], [10, 9, 4, 0, 3], [9, 8, 5, 3, 0]])
    # # all_elements = ['a', 'b', 'c', 'd', 'e']
    # dissimilarity_matrix = pd.DataFrame(mat, index=all_elements, columns=all_elements)
    #
    # current_clusters = ([all_elements])
    #
    # level = 1
    # index = 0
    # label = False
    #
    # while (index != -1) and (label == False):
    #     # print(level, current_clusters)
    #     start = time.clock()
    #
    #     (a_clstr, b_clstr) = split(current_clusters[index])
    #     end = time.clock()
    #     print('Cluster costs %f s' % (end - start))
    #     # 对分割的簇进行变更
    #     del current_clusters[index]
    #
    #     current_clusters.append(a_clstr)
    #     current_clusters.append(b_clstr)
    #     print(current_clusters)
    #     label = break_con(current_clusters)
    #     # 找到最大相似度的簇对其进行分解
    #     index = max_diameter(current_clusters)
    #     level += 1
    #
    # print(level, current_clusters)
    # exit(0)

# print(mat)












