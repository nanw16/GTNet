import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import multiprocessing as mp
from multiprocessing import Pool

def clean_A(A):
    s, t = A._indices().tolist()
    N = A.size(0)
    idx = []
    for i in range(len(s)):
        if s[i] == t[i]:
            idx.append(i)
    #print('self_loop # = ', len(idx))
    for i in idx[::-1]:
        del s[i]
        del t[i]
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    return A
            
def data_split(x, y, training_samples=20, val_samples=70):
    n_class = len(set(y.numpy()))
    #print('n_class = ', n_class)
    sel_samples = training_samples + val_samples
    sampling_strategy, train_samples = {}, {}
    for i in range(n_class):
        sampling_strategy[i] = sel_samples
        train_samples[i] = training_samples
    #sampling_strategy = {0:90, 1:90, 2:90, 3:90, 4:90, 5:90, 6:90}
    #train_samples = {0:20, 1:20, 2:20, 3:20, 4:20, 5:20, 6:20}
    rus1 = RandomUnderSampler(sampling_strategy=sampling_strategy)
    rus2 = RandomUnderSampler(sampling_strategy=train_samples)
    x_res, y_res = rus1.fit_resample(x.numpy(), y.numpy())
    test_indice = set(range(len(y)))
    selected_indice = rus1.sample_indices_
    test_indice.difference_update(set(selected_indice))
    test_indice = np.array(list(test_indice), dtype=np.int64)

    rus2.fit_resample(x_res, y_res)
    selected_indice2 = rus2.sample_indices_
    unselected_indice2 = set(range(len(y_res)))
    unselected_indice2.difference_update(set(selected_indice2))
    train_indice = selected_indice[selected_indice2]
    val_indice = selected_indice[list(unselected_indice2)]
    return train_indice, val_indice, test_indice

# D^-0.5 x A x D^-0.5
def norm_adj(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = torch.pow(degrees, -0.5)
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, torch.sparse.mm(A, D))

# D^-1 x A
def norm_adj2(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = 1/degrees
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, A)

def get_I(A, self_loop=False):
    if self_loop:
        A = norm_adj(A, True)
    s, t = A._indices()
    N = A.size(0)
    s1 = s.tolist()
    I = torch.sparse_coo_tensor([s1, list(range(len(s1)))], torch.ones(len(s1)), (N, len(s1)))
    return s, t, I

def separate(A, norm_type=1):
    if norm_type == 1:
        A = norm_adj(A, self_loop=True)
    else:
        A = norm_adj2(A, self_loop=True)
    s, t = A._indices().tolist()
    N = A.size(0)
    values = A._values().tolist()
    value1 = [0] * N
    value2 = []
    s1, t1 = [], []
    for i in range(len(s)):
        if s[i] == t[i]:
            value1[s[i]] = values[i]
        else:
            s1.append(s[i])
            t1.append(t[i])
            value2.append(values[i])
    A1 = torch.sparse_coo_tensor([s1, t1], torch.tensor(value2, dtype=torch.float32), (N, N))
    A2 = torch.tensor(value1, dtype=torch.float32).unsqueeze(-1)
    return A1, A2
    
def remove_edge_pts(accs, pct=0.1):
    accs = sorted(list(accs))
    N = len(accs)
    M = int(N * pct)
    accs = np.array(accs[M:N-M])
    return accs

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, A):
        A1 = A.coalesce()
        val = F.dropout(A1._values(), self.p, self.training)
        return torch.sparse.FloatTensor(A1._indices(), val, A1.shape)


    