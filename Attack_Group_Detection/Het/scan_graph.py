import random
from collections import Counter

import networkx as nx
import math
import numpy as np
file_path = './edges/new_node_embedding1.txt'
UserEmbedding = {}
Users = []
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split('  ')
        user = line[0]
        Users.append(user)
        embedding = line[1]
        UserEmbedding.setdefault(user, embedding)

# 在这加embedding的相似度。
# def similarity(G,v,u):
#     v_set = set(G.neighbors(v))
#     u_set = set(G.neighbors(u))
#     inter = v_set.intersection(u_set)
#     if inter == 0:
#         return 0
#     #need to account for vertex itself, add 2(1 for each vertex)
#     sim = (len(inter) + 2)/(math.sqrt((len(v_set) + 1 )*(len(u_set) + 1)))
#     return sim

def similarity(v,u):
    # Embedding1 = UserEmbedding[v]
    # Embedding2 = UserEmbedding[u]
    Embedding1 = list(map(float, UserEmbedding[v].split(', ')))
    Embedding2 = list(map(float, UserEmbedding[u].split(', ')))
    list1 = np.array(Embedding1)
    list2 = np.array(Embedding2)
    op2 = np.linalg.norm(list1 - list2)
    return op2



# 图， 节点
def neighborhood(G,v,eps):
    eps_neighbors =[]

    v_list = G.neighbors(v)

    for u in v_list:
        # print(similarity(G,u,v),u,v)

        if(similarity(u,v)) < eps:
            eps_neighbors.append(u)
    return eps_neighbors
    
def hasLabel(cliques,vertex):
    for k,v in cliques.items():

        if vertex in v:
            return True
    return False

def isNonMember(li,u):
    if u in li:
        return True
    return False

def sameClusters(G,clusters,u):
    n = list(G.neighbors(u))
    # print('#######################################3')
    # print(n)
    # print(len(n))
    #belong 
    b = []
    i = 0
    
    while i < len(n):
        for k,v in clusters.items():
            if n[i] in v:
                if k in b:
                    continue
                else:
                    b.append(k)
        i = i + 1
    if len(b) > 1:
        return False
    return True
                
    
    
def scan(G,eps=0.7, mu=1):
    c = 0
    clusters = dict()
    nomembers = []

    # n:节点 nbrs：该节点的邻居
    for n,nbrs in G.adjacency():
        # print(n, nbrs)
        # 看看n这个节点是否在clusters聚类中
        if hasLabel(clusters,n):
            # print(clusters, n)
            continue
        else:
            # 如果该节点的邻居节点的权重值》thresold，则将其邻居节点加入进去
            N = neighborhood(G,n,eps)

            # 检验该节点是否是核心点
            #test if vertex is core
            if len(N) > mu :
                '''Generate a new cluster-id c'''
                c = c + 1
                Q = neighborhood(G,n,eps)
                clusters[c] = []
                # append core vertex itself
                clusters[c].append(n)
                while len(Q) != 0:
                    w = Q.pop(0)
                    R = neighborhood(G,w,eps)
                    # include current vertex itself
                    R.append(w)
                    for s in R:
                        if not(hasLabel(clusters,s)) or isNonMember(nomembers,s):
                            clusters[c].append(s)
                        if not(hasLabel(clusters,s)):
                            Q.append(s)
            else:
                nomembers.append(n)

    outliers = []
    hubs = []
    for v in nomembers:
        if not sameClusters(G,clusters,v):
            hubs.append(v)
        else:
            outliers.append(v)
        

    return clusters,hubs,outliers

                        
                        
def main():
    G=nx.Graph()
    embedding_file_path = './edges/new_node_embedding1.txt'
    uu_file_path = './edges/uu_neigh1.txt'
    ui_file_path = './edges/ui_neigh.txt'
    uu_neigh = []
    ui_neigh = []
    with open(uu_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split('  ')
            user1 = line[0]
            users = line[1][2:-1].split(",")

            for user in users:
                if user[1]!='u':
                    user2 = user[2: -1]
                else:
                    user2 = user[1: -1]
                uu_neigh.append((user1, user2))


    with open(ui_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split('   ')
            user = line[0]

            items = line[1][1: -1].split(',')
            for item in items:
                if item[1]!='i':
                    item = item[2: -1]
                else:
                    item = item[1: -1]
                ui_neigh.append((user, item))
    G.add_edges_from(uu_neigh)
    G.add_edges_from(ui_neigh)


    # G.add_edges_from([('a','b'),('a','f'),('a','d'),('a','e'),
    #               ('f','b'),('f','c'),('f','e'),('b','c'),
    #               ('b','d'),('d','e'),('d','c'),('e','l'),
    #               ('d','h'),('e','k'),('h','g'),('h','i'),
    #               ('h','j'),('h','k'),('k','i'),('g','j'),
    #               ('k','j'),('k','g')])



    # for k in G.edges:
    #     print(k)
    # exit(0)

    # 计算eps
    # 平均值， 分位数
    num = 50
    temp = []

    for i in range(num):
        user = random.choice(list(UserEmbedding.keys())[:5055])
        item = random.choice(list(UserEmbedding.keys())[5055:])
        temp.append(user)
        temp.append(item)

    des = []
    for user1 in temp:
        for user2 in temp:
            if similarity(user1, user2) != 0:
                des.append(similarity(user1, user2))
    a = np.array(des)
    # print(a)

    media = np.median(a)
    # media = np.percentile(a,55)
    # for k in np.arange(1, 100, 1):
    # 10效果不好 50还行 66buxing  34buxing
    media = np.percentile(a, 50)

    # print(media)
    avg = 0
    for s in des:
        avg = s + avg
    avg = avg / (len(temp) * (len(temp) - 1))
    # media = avg
    # print(num, 'average', avg, 'Fenweishu', media)
    eps = media

    # eps = math.floor(avg)
    # print(eps)
    UserGroup = []
    AllUserGroup = []
    (clusters,hubs,outliers) = scan(G, eps=eps)
    # print('clusters: ')
    for j,v in clusters.items():
        # print(k,v)
        temp = []
        for i in v:
            if i[0] == 'u':
                temp.append(i)
        AllUserGroup.append(temp)
        if len(temp) > 1:
            UserGroup.append(temp)
    print(len(clusters))
    # print('hubs',hubs)
    # print('outliers',outliers)
    print(50, eps, len(UserGroup))

    path  = './edges/CandidateGroup.txt'
    f = open(path, 'r+')
    f.truncate()
    f.close()
    with open(path, 'a') as file:
        for group in AllUserGroup:
            if len(group) != 0:
                file.write(str(group) + '\n')





main()




