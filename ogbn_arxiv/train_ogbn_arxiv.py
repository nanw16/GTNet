import numpy as np
import torch
import torch.nn as nn
from GNN_models import GCN, GAT, APPNP, DAGNN, GTCN2, TreeLSTM, GTCN, GTAN
from utils import norm_adj, clean_A, separate
from collections import Counter
import dgl.data
import time
from sklearn import metrics
import argparse
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            
def sym_matrix(A):
    A = clean_A(A)
    N = A.size(0)
    s, t = A._indices().tolist()
    s, t = s + t, t + s
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    return A

def evaluate_f1(model, x, y, g, train_mask, val_mask, test_mask, average='macro'):
    model.eval()
    with torch.no_grad():
        y_pred = model(x, g)
        y_pred_train = y_pred[train_mask].argmax(1)
        y_pred_val = y_pred[val_mask].argmax(1)
        y_pred_test = y_pred[test_mask].argmax(1)
        train_f1 = metrics.f1_score(y[train_mask].cpu().numpy(), y_pred_train.cpu().numpy(), average=average)
        val_f1 = metrics.f1_score(y[val_mask].cpu().numpy(), y_pred_val.cpu().numpy(), average=average)
        test_f1 = metrics.f1_score(y[test_mask].cpu().numpy(), y_pred_test.cpu().numpy(), average=average)
        return train_f1, val_f1, test_f1
    
def validate(model, x, y, g, train_mask, val_mask, test_mask):
    evaluator = Evaluator(name='ogbn-arxiv')
    model.eval()
    with torch.no_grad():
        y_pred = model(x, g).argmax(1)
        train_acc = evaluator.eval({'y_true': y[train_mask].unsqueeze(-1), 'y_pred': y_pred[train_mask].unsqueeze(-1)})['acc']
        val_acc = evaluator.eval({'y_true': y[val_mask].unsqueeze(-1), 'y_pred': y_pred[val_mask].unsqueeze(-1)})['acc']
        test_acc = evaluator.eval({'y_true': y[test_mask].unsqueeze(-1), 'y_pred': y_pred[test_mask].unsqueeze(-1)})['acc']
        return train_acc, val_acc, test_acc
    
def train(model, x, y, g, train_mask, val_mask, test_mask, args, run, logger):
    device = args.device
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    
    tic = time.time()
    for epoch in range(1, args.num_iter + 1):    
        model.train()
        y_pred = model(x, g)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.eval_metric == 'acc':
            result = validate(model, x, y, g, train_mask, val_mask, test_mask)
        elif args.eval_metric == 'f1-macro':
            result = evaluate_f1(model, x, y, g, train_mask, val_mask, test_mask, average='macro')
        else:
            result = evaluate_f1(model, x, y, g, train_mask, val_mask, test_mask, average='micro')
        logger.add_result(run, result)
        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Time elapsed: {time.time() - tic:.2f}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            tic = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_run(x, y, A, train_mask, val_mask, test_mask, args):
    logger = Logger(args.num_test, None)
    g = {}
    for i in range(args.num_test):
        if args.model == 'GCN':
            n_dims = [args.n_in] + [args.n_hid] * (args.hop - 1) + [args.n_out]
            model = GCN(n_dims, args.dropout)
            g['A'] = norm_adj(A, self_loop=True).to(args.device)
        elif args.model == 'GAT':
            n_dims = [args.n_in] + [args.n_hid//args.num_heads] * (args.hop - 1) + [args.n_out]
            n_heads = [args.num_heads] * (args.hop - 1) + [args.num_out_heads]
            model = GAT(n_dims, n_heads, args.dropout, args.dropout2)
            g['edge_index'] = A._indices().to(args.device)
            # g['s'], g['t'], g['I'] = get_I(A, self_loop=True)
            # g['I'] = g['I'].to(args.device)
        elif args.model == 'APPNP':
            model = APPNP(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop, args.alpha)
            g['A'] = norm_adj(A, self_loop=True).to(args.device)
        elif args.model == 'DAGNN':
            model = DAGNN(args.n_in, args.n_hid, args.n_out, args.hop, args.dropout)
            g['A'] = norm_adj(A, self_loop=True).to(args.device)
        elif args.model == 'TreeLSTM':
            model = TreeLSTM(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop)
            g['A'] = norm_adj(A, self_loop=False).to(args.device)
        elif args.model == 'GTCN':
            model = GTCN(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop)
            g['A1'], g['A2'] = separate(A, norm_type=2)
            g['A1'], g['A2'] = g['A1'].to(args.device), g['A2'].to(args.device)
            #A1, A2 = separate(A, norm_type=2)
            #g['edge_index'] = A1._indices().to(args.device)
            #g['edge_weight1'], g['edge_weight2'] = A1._values().to(args.device), A2.to(args.device)
        elif args.model == 'GTAN':
            # model = GTAN(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, 
            #               args.hop, args.num_heads, layerwise=True, zero_init=True)
            model = GTAN(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, 
                          args.hop, layerwise=args.layerwise, zero_init=args.zero_init)
            edge_index = A._indices()
            g['edge_index'] = edge_index.to(args.device)
            # g['edge_attr'], g['size'] = None, None
        elif args.model == 'GTCN2':
            model = GTCN2(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop)
            g['A1'], g['A2'] = separate(A, norm_type=2)
            g['A1'], g['A2'] = g['A1'].to(args.device), g['A2'].to(args.device)
        if i == 0:
            print('#Parameters:', sum(p.numel() for p in model.parameters()))
        train(model, x, y, g, train_mask, val_mask, test_mask, args, i, logger)
        logger.print_statistics(i)
    logger.print_statistics()
    return logger

def main(args):
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    data = dataset[0]
    x, y = data.x, data.y[:, 0]
    scaler = StandardScaler()
    x = torch.from_numpy(scaler.fit_transform(x.numpy()))
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), (data.num_nodes, data.num_nodes))
    splitted_idx = dataset.get_idx_split()
    train_mask, val_mask, test_mask = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    A = clean_A(A)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.n_in, args.n_out = dataset.num_features, dataset.num_classes

    #print('label info:')
    #print(Counter(y[train_mask].tolist()))
    logger = test_run(x, y, A, train_mask, val_mask, test_mask, args)
    return logger

if __name__ == "__main__":
    """
        Test Settings
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', default='ogbn-arxiv', help='Name of dataset.')
    parser.add_argument('--model', default='GTAN', help='GNN models.')
    parser.add_argument('--n_hid', type=int, default=128, help='num of hidden features. For multilayer GNN, n_hid is the same for each layer')
    parser.add_argument('--num_heads', type=int, default=1, help='intermediate layer num heads (for GAT use only)')
    parser.add_argument('--num_out_heads', type=int, default=1, help='output layer num heads (for GAT use only)')
    parser.add_argument('--dropout', type=float, default=0.2, help='input dropout or GCN layer dropout (if GCN is used)')
    parser.add_argument('--dropout2', type=float, default=0, 
                        help='2nd dropout, used by models other than GCN, like GAT, APPNP edge dropout')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-5, help='weight decay in Adam optimizer.')
    parser.add_argument('--num_iter', type=int, default=1000, help='Max epochs to run.')
    parser.add_argument("--num_test", type=int, default=10, help='num of runs to test accuracy.')
    parser.add_argument("--hop", type=int, default=4, help='hop of GNN models.')
    parser.add_argument('--alpha', type=float, default=0.1, help='APPNP teleport probability.')
    parser.add_argument('--layerwise', action='store_true', default=False, 
                        help="whether to use layerwise parameters")
    parser.add_argument('--zero_init', action='store_true', default=False, 
                        help="zero initialize attention params")
    parser.add_argument('--eval_metric', default='acc', choices=["acc", "f1-macro", "f1-micro"], help='evaluation metrics.')
    parser.add_argument('--log_steps', type=int, default=100, help="training log steps")
    args = parser.parse_args()
    logger = main(args)

