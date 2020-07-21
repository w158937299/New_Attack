import torch.nn.functional as F
import torch
import torch.nn as nn


class NewAttention(nn.Module):
    def __init__(self, embedding_dims):
        super(NewAttention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)


    def forward(self, UserEmb, neighEmb):

        num_neighs = len(neighEmb)

        uv_reps = UserEmb.repeat(num_neighs, 1)

        x = torch.cat((neighEmb, uv_reps), 1)

        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)

        x = self.att3(x)

        att = F.softmax(x, dim=0)

        # x = torch.cat([UserEmb, ItemEmb, RatEmb, TimeEmb], 0)
        # x = F.relu(self.att1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.att3(x)
        #
        # att = F.softmax(x, dim=0)

        return att

