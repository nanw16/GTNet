import torch
import torch.nn as nn
from layers import GCN_layer, APPNP_layer
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing, GATConv

class GCN(nn.Module):
    def __init__(self, n_dims, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(len(n_dims)-1):
            self.layers.append(GCN_layer(n_dims[i], n_dims[i+1]))
    def forward(self, x, g):
        A = g['A']
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, A)
            x = self.relu(x)
            x = self.dropout(x)
        return self.layers[-1](x, A)
 
class GAT(nn.Module):
    def __init__(self, n_dims, n_heads, dropout=0.6, attn_dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.layers = nn.ModuleList()
        n_in = n_dims[0]
        for i in range(len(n_dims)-2):
            self.layers.append(GATConv(n_in, n_dims[i+1], n_heads[i], concat=True, negative_slope=alpha, dropout=attn_dropout))
            n_in = n_dims[i+1] * n_heads[i]
        self.layers.append(GATConv(n_in, n_dims[-1], n_heads[-1], concat=False, negative_slope=alpha, dropout=attn_dropout))
    def forward(self, x, g):
        edge_index = g['edge_index']
        x = self.dropout(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x
    
class APPNP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, k=5, alpha=0.1):
        super(APPNP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.appnp = APPNP_layer(n_out, k, dropout2, alpha)
        self.reset_param()
    def forward(self, x, g):
        A = g['A']
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.appnp(x, A)
        return x
    def reset_param(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        
class DAGNN(nn.Module):
    def __init__(self, n_in_dim, n_hid_dim, n_out_dim, hop, dropout=0):
        super(DAGNN, self).__init__()
        self.hop = hop
        self.linear1 = nn.Linear(n_in_dim, n_hid_dim)
        self.linear2 = nn.Linear(n_hid_dim, n_out_dim)
        self.s = nn.Parameter(torch.FloatTensor(n_out_dim, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def forward(self, x, g):
        A = g['A']
        #x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        out = [x]
        for _ in range(self.hop):
            x = torch.sparse.mm(A, x)
            out.append(x)
        H = torch.stack(out, dim=1)
        S = torch.sigmoid(torch.matmul(H, self.s))
        S = S.permute(0, 2, 1)
        H = torch.matmul(S, H).squeeze()
        return H
    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.s, gain=gain)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1)

class GTCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=10):
        super(GTCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(n_in, n_hid)
        self.layer2 = nn.Linear(n_hid, n_hid)
        self.layer3 = nn.Linear(n_hid, n_out)
        #self.reset_parameters()
    def forward(self, x, g):
        A1, A2 = g['A1'], g['A2']
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        h = x
        for i in range(self.hop):
            h = torch.sparse.mm(A1, h) + A2 * x
        h = self.relu(h)
        h = self.dropout2(h)
        h = self.layer3(h)
        return h
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.layer3.weight, gain=1.414)

class GTAN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, attn_dropout=0.5, hop=10, layerwise=True, zero_init=False):
        super(GTAN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.hop = hop
        self.layerwise = layerwise
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if not self.layerwise:
            self.attn1 = nn.Linear(n_hid, 1, bias=False)#nn.Parameter(torch.FloatTensor(n_hid, 1))
            self.attn2 = nn.Linear(n_hid, 1, bias=False)#nn.Parameter(torch.FloatTensor(n_hid, 1))
        else:
            self.attn1 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
            self.attn2 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
        self.fc3 = nn.Linear(n_hid, n_out)
        self.elu = nn.ELU()
        self.reset_parameters(zero_init)
    def forward(self, x, g):
        N = x.size(0)
        s, t = g['edge_index']
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        h = x
        for i in range(self.hop):
            if self.layerwise:
                attn1 = self.attn1[i]
                attn2 = self.attn2[i]
            else:
                attn1 = self.attn1
                attn2 = self.attn2
            x1 = attn1(x)
            h1 = attn2(h)
            w1 = x1[s] + h1[t]
            w2 = x1 + attn2(x)
            w1 = torch.exp(self.leakyrelu(w1))
            w2 = torch.exp(self.leakyrelu(w2))
            div = scatter(w1, s, dim=0, dim_size=N) + w2
            h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
            h = h/div
            h = self.elu(h)
        h = self.dropout2(h)
        h = self.fc3(h)
        return h
    def reset_parameters(self, zero_init):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        if not self.layerwise:
            if zero_init:
                nn.init.zeros_(self.attn1.weight)
                nn.init.zeros_(self.attn2.weight)
            else:
                nn.init.xavier_uniform_(self.attn1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.attn2.weight, gain=nn.init.calculate_gain('relu'))
        else:
            if zero_init:
                for i in range(self.hop):
                    nn.init.zeros_(self.attn1[i].weight)
                    nn.init.zeros_(self.attn2[i].weight)
            else:
                for i in range(self.hop):
                    nn.init.xavier_uniform_(self.attn1[i].weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.xavier_uniform_(self.attn2[i].weight, gain=nn.init.calculate_gain('relu'))
                    
class TreeLSTM(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=10):
        super(TreeLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.fc = nn.Linear(n_hid, n_out)
        self.lstm = nn.LSTMCell(n_in, n_hid)
        self.relu = nn.ReLU()
        self.reset_param()
    def forward(self, x, g):
        A = g['A']
        x = self.dropout(x)
        h, c = self.lstm(x)
        h = torch.sparse.mm(A, h)
        for i in range(1, self.hop):
            h, c = self.lstm(x, (h, c))
            h = torch.sparse.mm(A, h)
        h = self.relu(h)
        h = self.dropout2(h)
        h = self.fc(h)
        return h
    def reset_param(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)

# GTCN with 'linear + relu' added at each layer
class GTCN2(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=10):
        super(GTCN2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(n_in, n_hid)
        self.layer2 = nn.Linear(n_hid, n_out)
        self.layer3 = nn.Linear(n_hid, n_hid)
        self.reset_parameters()
    def forward(self, x, g):
        A1, A2 = g['A1'], g['A2']
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        h = x
        for i in range(self.hop):
            h = torch.sparse.mm(A1, h) + A2 * x
            h = self.relu(h + self.layer3(h))
            h = self.dropout2(h)
        h = self.layer2(h)
        return h
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.layer3.weight, gain=1.414)
        #nn.init.zeros_(self.layer3.weight)