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
from graphDec import graphDec
from tqdm import tqdm


if __name__ == '__main__':
    args = Parser.Define_Params()
    file_path = args.file_path
    UserID, ItemID, Ratings, Attacks, Times, UItoR, ItemToUser, UserToItem, UserToTime, UserToRat, \
    tUserID, tItemID, tRatings, tAttacks, tTimes, tUItoR, newTimes, tnewTimes, whole_times = Bulidata.bulid_dataset(file_path)
    # UserID = [[[1,2,3], [4,5,6] ,[7,8,9]],[[3,2,1],[6,5,4],[9,8,7]]]
    # print(UserID)
    # print(torch.LongTensor(UserID))
    # train_set = torch.utils.data.TensorDataset(torch.LongTensor(UserID))
    # print(train_set)
    # train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)
    # print(train_loader)
    # for a in train_loader:
    #     print(a)
    # exit(0)
    Rats = [1, 2, 3, 4, 5]

    train_set = torch.utils.data.TensorDataset(torch.LongTensor(UserID), torch.LongTensor(ItemID), torch.LongTensor(Ratings), torch.LongTensor(newTimes), torch.FloatTensor(Attacks))
    # print(train_set)
    # exit(0)
    test_set = torch.utils.data.TensorDataset(torch.LongTensor(tUserID), torch.LongTensor(tItemID), torch.LongTensor(tRatings), torch.LongTensor(tnewTimes), torch.FloatTensor(tAttacks))
    # 这个shuffle是整体都shuffle么？,是的依然是互相对应的。
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    unum = len(UserToItem)
    inum = len(ItemToUser)
    rnum = len(Rats)
    tnum = len(whole_times)
    tmpU = list(UserToItem.keys())
    tmpI = list(ItemToUser.keys())
    tmpT = whole_times

    tmpR = Rats



    embed_dim = args.embed_dim
    u2e = nn.Embedding(unum, embed_dim)
    # print(u2e.weight)
    # print(u2e.weight.shape)
    # exit(0)
    i2e = nn.Embedding(inum, embed_dim)

    r2e = nn.Embedding(rnum, embed_dim)
    t2e = nn.Embedding(tnum, embed_dim)

    model = graphDec(u2e, i2e, r2e, t2e, embed_dim, tmpU, tmpI, tmpR, tmpT, UserToItem, UserToTime, UserToRat)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    start_epoch = 1
    RESUME = False
    if RESUME:
        path_checkpoint = "./last_models/checkpoint/yelpckpt_best_2.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    for epoch in range(start_epoch, 100):

        print('Now training the %dth epoch'%epoch)

        model.train()
        for data in tqdm(train_loader):
            UserID, ItemID, Ratings, newTimes, Attacks = data
            optimizer.zero_grad()
            score = model(UserID, ItemID, Ratings, newTimes)

            criteria = torch.nn.MSELoss()
            loss = criteria(score, Attacks)

            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch%5 == 0:

            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if not os.path.isdir("./last_models/checkpoint"):
                os.makedirs("./last_models/checkpoint")
            torch.save(checkpoint, './last_models/checkpoint/yelpckpt_best_%s.pth' % (str(epoch)))
        if epoch%1 == 0:
            model.eval()
            tru = 0
            num = 0
            recall = 0
            FN = 0
            with torch.no_grad():
                for data in test_loader:
                    tUserID, tItemID, tRatings, tTimes, tAttacks = data
                    tAttacks = tAttacks.tolist()
                    scores = model(tUserID, tItemID, tRatings, tTimes)
                    rmse = sqrt(mean_squared_error(tAttacks, scores))
                    for score in range(len(scores)):
                        if scores[score] >= 0:
                            scores[score] = 1
                        else:
                            scores[score] = -1
                    for i in range(len(scores)):
                        num = num + 1
                        if tAttacks[i] == scores[i]:
                            tru = tru + 1
                            if tAttacks[i] == 1:
                                recall = recall + 1
                        else:
                            if tAttacks[i] == 1:
                                FN = FN + 1



                correct_p = tru/num
                recall1 = recall/(recall + FN)
                tru = 0
                print('after %d epochs, the test accuracy is %.4f and the rmse is %.4f and the recall is %.4f'%(epoch, correct_p, rmse, recall1))



    
