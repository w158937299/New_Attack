import math
import copy
import numpy as np
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


def dist(a,b):
    """
    用来计算两个样本点之间的距离
    :param a: 样本点
    :param b: 样本点
    :return: 两个样本点之间的距离
    """
    return math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2))


def returnDk(matrix,k):
    """
    用来计算第K最近的距离集合
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
    求第K最近距离集合的平均值
    :param Dk: k-最近距离集合
    :return: Dk的平均值
    """
    sum = 0
    for i in range(len(Dk)):
        sum = sum + Dk[i]
    return sum/len(Dk)


def CalculateDistMatrix(dataset):
    """
    计算距离矩阵
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
    计算Eps候选列表
    :param dataSet: 数据集
    :return: eps候选集合
    """
    # 求距离矩阵
    DistMatrix = CalculateDistMatrix(dataSet)

    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()

    EpsCandidate = []
    for k in range(1,len(dataSet)):
        Dk = returnDk(tmp_matrix,k)


        DkAverage = returnDkAverage(Dk)
        EpsCandidate.append(DkAverage)

    return EpsCandidate


def returnMinptsCandidate(DistMatrix,EpsCandidate):
    """
    计算Minpts候选列表
    :param DistMatrix: 距离矩阵
    :param EpsCandidate: Eps候选列表
    :return: Minpts候选列表
    """

    MinptsCandidate = []
    for k in range(len(EpsCandidate)):
        # 第k近距离的平均值
        tmp_eps = EpsCandidate[k]
        tmp_count = 0
        for i in range(len(DistMatrix)):
            for j in range(len(DistMatrix[i])):
                if DistMatrix[i][j] <= tmp_eps:
                    tmp_count = tmp_count + 1
        MinptsCandidate.append(tmp_count/len(dataSet))
    return MinptsCandidate


def returnClusterNumberList(dataset,EpsCandidate,MinptsCandidate):
    """
    计算聚类后的类别数目 
    :param dataset: 数据集
    :param EpsCandidate: Eps候选列表
    :param MinptsCandidate: Minpts候选列表
    :return: 聚类数量列表
    """
    np_dataset = np.array(dataset)  #将dataset转换成numpy_array的形式
    ClusterNumberList = []
    for i in range(len(EpsCandidate)):
        clustering = DBSCAN(eps= EpsCandidate[i],min_samples= MinptsCandidate[i]).fit(np_dataset)
        num_clustering = max(clustering.labels_)
        ClusterNumberList.append(num_clustering)
        ClusterGroup = {}
        num = 0
        for label in clustering.labels_:

            if label not in ClusterGroup:
                ClusterGroup.setdefault(label, [num])
            else:
                temp = ClusterGroup[label]
                temp.append(num)
                ClusterGroup[label] = temp
            num = num + 1
        print(len(ClusterGroup))

        print(clustering.labels_)
        print(EpsCandidate[i], num_clustering)
        exit(0)
    return ClusterNumberList

if __name__ == '__main__':
    dataSet = loadDataSet('788points.txt', splitChar=',')

    # 简单点说，就是先计算各个点的距离，然后每一行按从小到大排序，然后计算每个用户的第k近距离求他们的平均距离作为这个候选值
    EpsCandidate = returnEpsCandidate(dataSet)

    DistMatrix = CalculateDistMatrix(dataSet)
    # 通过计算距离矩阵然后和第k近距离的平均值进行比较最后求一个比例
    MinptsCandidate = returnMinptsCandidate(DistMatrix,EpsCandidate)

    ClusterNumberList = returnClusterNumberList(dataSet,EpsCandidate,MinptsCandidate)
    print(EpsCandidate)
    print(MinptsCandidate)
    # print('cluster number list is')
    # print(ClusterNumberList)