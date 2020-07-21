import torch.nn.functional as F
import torch
import torch.nn as nn


class NewAttention(nn.Module):
    def __init__(self, embedding_dims):
        super(NewAttention, self).__init__()
        self.embed_dim = embedding_dims

        self.att1 = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.att2 = nn.Linear(self.embed_dim//2, self.embed_dim//4)
        self.att3 = nn.Linear(self.embed_dim//4, 1)
        self.softmax = nn.Softmax(0)


    def forward(self, UserEmb):

        # num_neighs = len(neighEmb)

        # uv_reps = UserEmb.repeat(num_neighs, 1)
        #
        # x = torch.cat((neighEmb, uv_reps), 1)

        # x = UserEmb.view(UserEmb.shape[0] * UserEmb.shape[1], self.embed_dim)
        x = UserEmb

        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)

        x = self.att3(x)

        att = F.softmax(x, dim=1)

        # x = torch.cat([UserEmb, ItemEmb, RatEmb, TimeEmb], 0)
        # x = F.relu(self.att1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.att3(x)
        #
        # att = F.softmax(x, dim=0)

        return att

