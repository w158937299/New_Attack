import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.att1 = nn.Linear(self.embed_dim , self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)


    def forward(self, UserEmb, ItemEmb, RatEmb, TimeEmb):

        x = torch.cat([UserEmb, ItemEmb, RatEmb, TimeEmb], 0)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)

        att = F.softmax(x, dim=0)

        return att


