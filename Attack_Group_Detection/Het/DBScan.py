import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import Parser
import time
import math
import copy
from sklearn.cluster import DBSCAN


def loadDataSet(fileName, splitChar='\t'):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def dist(a, b):
    """
    :param a: 样本点
    :param b: 样本点
    :return: 两个样本点之间的欧式距离
    """
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def returnDk(matrix, k):
    """
    :param matrix: 距离矩阵
    :param k: 第k最近
    :return: 第k最近距离集合
    """
    Dk = []
    for i in range(len(matrix)):
        Dk.append(matrix[i][k])
    return Dk


def returnDkAverage(Dk):
    """
    :param Dk: k-最近距离集合
    :return: Dk的平均值
    """
    sum = 0
    for i in range(len(Dk)):
        sum = sum + Dk[i]
    return sum / len(Dk)


def CalculateDistMatrix(dataset):
    """
    :param dataset: 数据集
    :return: 距离矩阵
    """
    DistMatrix = [[0 for j in range(len(dataset))] for i in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            DistMatrix[i][j] = dist(dataset[i], dataset[j])
    return DistMatrix


def returnEpsCandidate(dataSet):
    """
    :param dataSet: 数据集
    :return: eps候选集合
    """
    DistMatrix = CalculateDistMatrix(dataSet)
    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()
    EpsCandidate = []
    for k in range(1, len(dataSet)):
        Dk = returnDk(tmp_matrix, k)
        DkAverage = returnDkAverage(Dk)
        EpsCandidate.append(DkAverage)
    return EpsCandidate


def returnMinptsCandidate(DistMatrix, EpsCandidate):
    """
    :param DistMatrix: 距离矩阵
    :param EpsCandidate: Eps候选列表
    :return: Minpts候选列表
    """
    MinptsCandidate = []
    for k in range(len(EpsCandidate)):
        tmp_eps = EpsCandidate[k]
        tmp_count = 0
        for i in range(len(DistMatrix)):
            for j in range(len(DistMatrix[i])):
                if DistMatrix[i][j] <= tmp_eps:
                    tmp_count = tmp_count + 1
        MinptsCandidate.append(tmp_count / len(dataSet))
    return MinptsCandidate


def returnClusterNumberList(dataset, EpsCandidate, MinptsCandidate):
    """
    :param dataset: 数据集
    :param EpsCandidate: Eps候选列表
    :param MinptsCandidate: Minpts候选列表
    :return: 聚类数量列表
    """
    np_dataset = np.array(dataset)  # 将dataset转换成numpy_array的形式
    ClusterNumberList = []
    for i in range(len(EpsCandidate)):
        clustering = DBSCAN(eps=EpsCandidate[i], min_samples=MinptsCandidate[i]).fit(np_dataset)
        num_clustering = max(clustering.labels_)
        ClusterNumberList.append(num_clustering)

if __name__ == '__main__':

    # dataSet = loadDataSet('788points.txt', splitChar=',')
    # with open('./edges/OJLD_Similar_matrix1.txt', 'r') as file:
    #     lines = file.readlines()
    #
    # EpsCandidate = returnEpsCandidate(dataSet)
    # DistMatrix = CalculateDistMatrix(dataSet)
    # print(len(DistMatrix))
    # print(len(DistMatrix[0]))
    # exit(0)
    # MinptsCandidate = returnMinptsCandidate(DistMatrix,EpsCandidate)
    #
    # print(EpsCandidate)
    # print(MinptsCandidate)
    # print('cluster number list is')
    # print(ClusterNumberList)
    # print(np.array(dataSet))
    # print(dataSet)

    UserEmbedding = {}
    file_path = './edges/new_node_embedding3.txt'
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
    np_dataset = np.array(list(UserEmbedding.values()))
    # print(np_dataset)
    # exit(0)
    # ClusterNumberList = []
    # 2.56, 1
    # for i in np.arange(1,10,1):
    ClusterGroup = {}
    # min_samples过小，类别数越少
    # 新的模型DBscan 2.2-2.3比较好
    # for i in np.arange(2.23,2.27,0.02):
    # for k in np.arange(1,5,1):


    ClusterGroup = {}
    # 2.56 1
    # (没有注意力机制)2.5 2.6效果都还不错
    clustering = DBSCAN(eps=2.23, min_samples=1).fit(np_dataset)
    num_clustering = max(clustering.labels_)
    num = 1
    for label in clustering.labels_:
        if label not in ClusterGroup:
            ClusterGroup.setdefault(label, [num])
        else:
            temp = ClusterGroup[label]
            temp.append(num)
            ClusterGroup[label] = temp
        num = num + 1
    UserGroup = []
    otherGroup = []
    for group in list(ClusterGroup.values()):
        temp = []
        for user in group:
            if int(user) <= 5055 :
                temp.append(user)
        if len(temp)!=0:
            UserGroup.append(temp)
        if len(temp) > 1:
            otherGroup.append(temp)
    print(1, 'num',len(ClusterGroup), len(UserGroup),len(otherGroup),'canshuzhi', num_clustering)


    file_path = './edges/Candidate_group_att.txt'
    f = open(file_path, 'r+')
    f.truncate()
    f.close()

    with open(file_path, 'a') as file:
        for user in UserGroup:
            file.write(str(user) + '\n')


    # ClusterNumberList.append(num_clustering)

