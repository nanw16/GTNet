import torch
import torch.nn as nn
from utils import SparseDropout

class GCN_layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCN_layer, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.reset_param()
    def forward(self, x, A):
        x = self.linear(x)
        return torch.sparse.mm(A, x)
    def reset_param(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)

class GAT_base_layer(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha=0.2):
        super(GAT_base_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(2*n_out, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.linear = nn.Linear(n_in, n_out)
        self.reset_param()
    def forward(self, x, s, t, I):
        x = self.linear(x)
        w = self.attn(torch.cat((x[s], x[t]), dim=-1))
        w = torch.exp(self.leakyrelu(w))
        div = torch.sparse.mm(I, w)
        w = self.dropout(w)
        x = torch.sparse.mm(I, w * x[t])
        x = x/div
        return x
    def reset_param(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.attn.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.attn.weight)

class GAT_layer(nn.Module):
    def __init__(self, n_in, n_out, n_head, dropout, alpha=0.2, concat=True):
        super(GAT_layer, self).__init__() 
        self.n_head = n_head
        self.layers = nn.ModuleList([GAT_base_layer(n_in, n_out, dropout, alpha) for _ in range(n_head)])
        self.concat = concat
    def forward(self, x, s, t, I):
        out = [self.layers[i](x, s, t, I) for i in range(self.n_head)]
        if self.concat:
            out = torch.cat(out, dim=-1)
        else:
            out = torch.mean(torch.stack(out), dim=0)
        return out

class APPNP_layer(nn.Module):
    def __init__(self, n_in, k, dropout=0.5, alpha=0.1):
        super(APPNP_layer, self).__init__() 
        self.dropout = SparseDropout(dropout)
        self.alpha = alpha
        self.k = k
    def forward(self, x, A):
        H = x
        for _ in range(self.k):
            A1 = self.dropout(A)
            H = (1 - self.alpha) * torch.sparse.mm(A1, H) + self.alpha * x
        return H
