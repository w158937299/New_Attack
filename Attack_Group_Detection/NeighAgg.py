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

class neighAgg(nn.Module):

    def __init__(self, i2e, r2e, u2e, t2e, embed_dim):
        self.u2e = u2e
        self.i2e = i2e
        self.r2e = r2e
        self.t2e = t2e
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, user, UserToItem, UserToTime, UserToRat):

        pass
