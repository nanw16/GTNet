import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GCN(nn.Module):
    def __init__(self, n_dims, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(len(n_dims)-1):
            self.layers.append(GCNConv(n_dims[i], n_dims[i+1]))
    def forward(self, x, g):
        edge_index = g['edge_index']
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x
 
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

class APPNP2(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, k=5, alpha=0.1):
        super(APPNP2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.appnp = APPNP(k, alpha, dropout2, cached=True)
        self.reset_parameters()
    def forward(self, x, g):
        edge_index = g['edge_index']
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.appnp(x, edge_index)
        return x
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = nn.Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()

class DAGNN(nn.Module):
    def __init__(self, n_in_dim, n_hid_dim, n_out_dim, hop, dropout=0):
        super(DAGNN, self).__init__()
        self.lin1 = nn.Linear(n_in_dim, n_hid_dim)
        self.lin2 = nn.Linear(n_hid_dim, n_out_dim)
        self.prop = Prop(n_out_dim, hop)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, g):
        edge_index = g['edge_index']
        x = self.dropout(x)
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x

class GTCN(MessagePassing):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=10):
        super().__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(n_in, n_hid)
        self.layer2 = nn.Linear(n_hid, n_out)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.414)
    def forward(self, x, g):
        edge_index = g['edge_index']
        edge_weight1, edge_weight2 = g['edge_weight1'], g['edge_weight2']
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        h = x
        for k in range(self.hop):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight1)
            h = h + edge_weight2 * x
            h = self.dropout2(h)
        h = self.layer2(h)
        return h
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

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
        if self.layerwise:
            #self.attn = nn.Linear(2*n_hid, 1, bias=False)
            self.attn1 = nn.Linear(n_hid, 1, bias=False)#nn.Parameter(torch.FloatTensor(n_hid, 1))
            self.attn2 = nn.Linear(n_hid, 1, bias=False)#nn.Parameter(torch.FloatTensor(n_hid, 1))
        else:
            self.attn1 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
            self.attn2 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
        self.fc2 = nn.Linear(n_hid, n_out)
        self.elu = nn.ELU()
        self.reset_parameters(zero_init)
    def forward(self, x, g):
        s, t = g['edge_index']
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        h = x
        if self.layerwise:
            for i in range(self.hop):
                x1 = self.attn1(x)
                w2 = x1 + self.attn2(x)
                h1 = self.attn2(h)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0) + w2
                h = scatter(w1 * h[t], s, dim=0) + w2 * x
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
        else:
            for i in range(self.hop):
                x1 = self.attn1[i](x)
                h1 = self.attn2[i](h)
                w1 = x1[s] + h1[t]
                w2 = x1 + self.attn2[i](x)
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0) + w2
                h = scatter(w1 * h[t], s, dim=0) + w2 * x
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
        h = self.fc2(h)
        return h
    def reset_parameters(self, zero_init):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if self.layerwise:
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
                    