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
from Attention import Attention
from NeighAgg import neighAgg
from NewAttention import NewAttention


class graphDec(nn.Module):

    def __init__(self, u2e, i2e, r2e, t2e, embed_dim, tmpU, tmpI, tmpR, tmpT, UserToItem, UserToTime, UserToRat):
        super(graphDec, self).__init__()
        self.u2e = u2e
        self.i2e = i2e
        self.r2e = r2e
        self.t2e = t2e
        self.embed_dim = embed_dim
        self.tmpU = tmpU
        self.tmpI = tmpI
        self.tmpR = tmpR
        self.tmpT = tmpT
        self.UserToItem = UserToItem
        self.UserToRat = UserToRat
        self.UserToTime = UserToTime

        self.linear = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w1 = nn.Linear(self.embed_dim , self.embed_dim//4)
        self.w2 = nn.Linear(self.embed_dim//4, self.embed_dim//8)
        self.w3 = nn.Linear(self.embed_dim//8, 1)
        self.neigh_e = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.neigh_e1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.bn1 = nn.BatchNorm1d(self.embed_dim //4, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim//8, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        # self.att = Attention(self.embed_dim)
        self.att = NewAttention(self.embed_dim)


    def forward(self, User, Item, Rating, Time):

        User = User.tolist()
        Item = Item.tolist()
        Rating = Rating.tolist()
        Time = Time.tolist()
        All_emb = torch.empty(len(User), self.embed_dim, dtype=torch.float)
        User_embs = torch.empty(len(User), self.embed_dim, dtype=torch.float)

        for i in range(len(User)):

            Item_neigh = self.UserToItem[str(User[i])]
            Rat_neigh = self.UserToRat[str(User[i])]
            Time_neigh = self.UserToTime[str(User[i])]

            item_group = []
            rat_group = []
            time_group = []
            for j in range(len(Item_neigh)):
                item_group.append(self.tmpI.index(str(Item_neigh[j])))
                rat_group.append(self.tmpR.index(int(Rat_neigh[j])))
                time_group.append(self.tmpT.index(Time_neigh[j]))


            Item_neigh_emb = self.i2e.weight[item_group]
            Rat_neigh_emb = self.r2e.weight[rat_group]
            Time_neigh_emb = self.t2e.weight[time_group]
            UserEmb = self.u2e.weight[self.tmpU.index(str(User[i]))]

            User_embs[i] = UserEmb

            x = torch.cat([Item_neigh_emb, Rat_neigh_emb, Time_neigh_emb], 1)
            # 在这加个注意力机制
            x = torch.relu(self.neigh_e(x))


            x = F.relu(self.neigh_e1(x))
            att = self.att(UserEmb, x)


            single_neigh = torch.mm(x.t(), att)

            single_neigh_emb = single_neigh.t()

            All_emb[i] = single_neigh_emb


        combined = torch.cat([User_embs, All_emb], dim = 1)

        combined = F.relu(self.linear(combined))



            # 在这里在对和用户相关的信息进行聚合编码
            # UserEmb = self.u2e.weight[self.tmpU.index(str(User[i]))]
            # ItemEmb = self.i2e.weight[self.tmpI.index(str(Item[i]))]
            # RatEmb = self.r2e.weight[self.tmpR.index(Rating[i])]
            # TimeEmb = self.t2e.weight[self.tmpT.index(Time[i])]
            # UserEmb = torch.unsqueeze(UserEmb, 0)
            # ItemEmb = torch.unsqueeze(ItemEmb, 0)
            # RatEmb = torch.unsqueeze(RatEmb, 0)
            # TimeEmb = torch.unsqueeze(TimeEmb, 0)
            # node_emb = torch.cat([UserEmb, ItemEmb, RatEmb, TimeEmb], 0)
            # 注意力
            # att = self.att(UserEmb, ItemEmb, RatEmb, TimeEmb)
            # a = torch.mm(node_emb.T, att)
            # a = torch.squeeze(a)
            # All_emb[i] = a


        All_embs = F.relu(self.bn1(self.w1(combined)))

        All_embs = F.dropout(All_embs, training= self.training)

        All_embs = F.relu(self.bn2(self.w2(All_embs)))

        All_embs = F.dropout(All_embs, training= self.training)

        scores = self.w3(All_embs)

        # print(scores)
        return torch.as_tensor(scores.squeeze(), dtype= torch.float32)




