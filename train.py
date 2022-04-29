import numpy as np
import torch
import torch.nn as nn
from GNN_models import GAT, APPNP, DAGNN, GTCN2, TreeLSTM, GTAN2, GCN, GTAN, GTCN, SimpleGCN, SimpleGAT
from utils import norm_adj, data_split, clean_A, remove_edge_pts, separate
from collections import Counter
import dgl.data
import dgl
import time
from sklearn import metrics
import argparse

def evaluate_f1_macro(model, x, y, g, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        y_pred = model(x, g)
        y_pred_val = y_pred[val_mask]
        y_pred_test = y_pred[test_mask]
        pred_val = y_pred_val.argmax(1)
        pred_test = y_pred_test.argmax(1)
        val_f1 = metrics.f1_score(y[val_mask].cpu().numpy(), pred_val.cpu().numpy(), average='macro')
        test_f1 = metrics.f1_score(y[test_mask].cpu().numpy(), pred_test.cpu().numpy(), average='macro')
        return val_f1, test_f1

def evaluate_f1_micro(model, x, y, g, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        y_pred = model(x, g)
        y_pred_val = y_pred[val_mask]
        y_pred_test = y_pred[test_mask]
        pred_val = y_pred_val.argmax(1)
        pred_test = y_pred_test.argmax(1)
        val_f1 = metrics.f1_score(y[val_mask].cpu().numpy(), pred_val.cpu().numpy(), average='micro')
        test_f1 = metrics.f1_score(y[test_mask].cpu().numpy(), pred_test.cpu().numpy(), average='micro')
        return val_f1, test_f1
    
def evaluate_acc(model, x, y, g, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        y_pred = model(x, g)
        y_pred_val = y_pred[val_mask]
        y_pred_test = y_pred[test_mask]
        pred_val = y_pred_val.argmax(1)
        pred_test = y_pred_test.argmax(1)
        val_acc = (y[val_mask] == pred_val).float().mean().item()
        test_acc = (y[test_mask] == pred_test).float().mean().item()
        return val_acc, test_acc
    
def train(model, x, y, g, train_mask, val_mask, test_mask, args):
    device = args.device
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.eval_metric == 'acc':
        eval_metric = evaluate_acc
    elif args.eval_metric == 'f1-macro':
        eval_metric = evaluate_f1_macro
    elif args.eval_metric == 'f1-micro':
        eval_metric = evaluate_f1_micro
    best_val = 0
    best_test = 0
    count = 0
    Y_pred = None
    
    tic = time.time()
    t_train = 0
    for epoch in range(1, args.num_iter + 1):    
        tic_train = time.time()
        model.train()
        y_pred = model(x, g)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_train += time.time() - tic_train
        val, test = eval_metric(model, x, y, g, val_mask, test_mask)
        # Save the best validation and the corresponding test accuracy/f1.
        if best_val < val:
            best_val = val
            best_test = test
            Y_pred = y_pred.detach()
            count = 0
        elif count >= args.patience:
            break
        else:
            count += 1
        if args.log:
            if epoch % 5 == 0:
                print_msg = 'Epoch {}, time elapsed: {:.3f}, loss: {:.3f}, val ' + args.eval_metric + ': {:.3f} (best {:.3f}), test ' + args.eval_metric + ': {:.3f} (test ' + args.eval_metric +' at best val {:.3f})'
                print(print_msg.format(epoch, time.time() - tic, loss.detach().item(), val, best_val, test, best_test))
                tic = time.time()
    print_msg = 'time duration = {:.3f}, best val ' + args.eval_metric + ' = {:.3f}, test ' + args.eval_metric + ' = {:.3f}'
    print(print_msg.format(time.time() - tic, best_val, best_test))
    # print('training time per epoch = {:.4f}'.format(t_train/epoch))
    #model = model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_test, model 

def test_run(x, y, A, train_mask, val_mask, test_mask, args):
    test_metrics = []
    g = {}
    for _ in range(args.num_test):
        if args.model == 'GCN':
            n_dims = [args.n_in] + [args.n_hid] * (args.hop - 1) + [args.n_out]
            model = GCN(n_dims, args.dropout)
            g['A'] = norm_adj(A, self_loop=True).to(args.device)
            #g['edge_index'] = A._indices().to(args.device)
            # norm_A = norm_adj(A, self_loop=True)
            # g['edge_index'] = norm_A._indices().to(args.device)
            # g['edge_weight'] = norm_A._values().to(args.device)
        elif args.model == 'GCNII':
            n_dims = [args.n_in] + [args.n_hid] * (args.hop - 1) + [args.n_out]
            model = GCNII(args.n_in, args.hop, args.n_hid, args.n_out, args.dropout, args.lamda, args.alpha, args.variant)
            g = norm_adj(A, self_loop=True).to(args.device)
        elif args.model == 'GAT':
            n_dims = [args.n_in] + [args.n_hid] * (args.hop - 1) + [args.n_out]
            n_heads = [args.num_heads] * (args.hop - 1) + [args.num_out_heads]
            model = GAT(n_dims, n_heads, args.dropout, args.dropout2)
            g['edge_index'] = A._indices().to(args.device)
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
            A1, A2 = separate(A, norm_type=1)
            g['edge_index'] = A1._indices().to(args.device)
            g['edge_weight1'], g['edge_weight2'] = A1._values().to(args.device), A2.to(args.device)
        elif args.model == 'GTAN':
            model = GTAN(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop, layerwise=args.layerwise, zero_init=args.zero_init)
            g['edge_index'] = A._indices().to(args.device)
        elif args.model == 'GTCN2':
            model = GTCN2(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop, args.layerwise)
            #g['A1'], g['A2'] = separate(A, norm_type=1)
            #g['A1'], g['A2'] = g['A1'].to(args.device), g['A2'].to(args.device)
            A1, A2 = separate(A, norm_type=1)
            g['edge_index'] = A1._indices().to(args.device)
            g['edge_weight1'], g['edge_weight2'] = A1._values().to(args.device), A2.to(args.device)
        elif args.model == 'GTAN2':
            model = GTAN2(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop, layerwise=args.layerwise, zero_init=args.zero_init)
            g['edge_index'] = A._indices().to(args.device)
        elif args.model == 'SimpleGCN':
            model = SimpleGCN(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop)
            A1, A2 = separate(A, norm_type=1)
            #g['edge_index'] = A1._indices().to(args.device)
            #g['edge_weight1'], g['edge_weight2'] = A1._values().to(args.device), A2.to(args.device)
            g['A1'], g['A2'] = A1.to(args.device), A2.to(args.device)
        elif args.model == 'SimpleGAT':
            model = SimpleGAT(args.n_in, args.n_hid, args.n_out, args.dropout, args.dropout2, args.hop, layerwise=args.layerwise, zero_init=args.zero_init)
            g['edge_index'] = A._indices().to(args.device)
        test_metric, _ = train(model, x, y, g, train_mask, val_mask, test_mask, args)
        test_metrics.append(test_metric) 
    test_metrics = np.array(test_metrics)
    return test_metrics

def main(args):
    if args.data == 'Citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    elif args.data == 'Pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    elif args.data == 'Coauthor-CS':
        dataset = dgl.data.CoauthorCSDataset()
    else:
        dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    A = g.adjacency_matrix()
    A = clean_A(A)
    x = g.ndata['feat']
    y = g.ndata['label']

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.n_in, args.n_out = x.size(1), len(set(y.tolist()))
    if args.data == 'Coauthor-CS' or args.random_label_split:
        if args.data_load:
            train_mask = torch.load(args.root_dir + '/label_split/train_mask_' + args.data + str(args.test_id) + '.pt')
            val_mask = torch.load(args.root_dir + '/label_split/val_mask_' + args.data + str(args.test_id) + '.pt')
            test_mask = torch.load(args.root_dir + '/label_split/test_mask_' + args.data + str(args.test_id) + '.pt')
        else:
            # generate label split randomly
            train_mask, val_mask, test_mask = data_split(x, y, training_samples=args.num_train, val_samples=args.num_val)
            torch.save(train_mask, args.root_dir + '/label_split/train_mask_' + args.data + str(args.test_id) + '.pt')
            torch.save(val_mask, args.root_dir + '/label_split/val_mask_' + args.data + str(args.test_id) + '.pt')
            torch.save(test_mask, args.root_dir + '/label_split/test_mask_' + args.data + str(args.test_id) + '.pt')
    else:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    # print('label info:')
    # print(Counter(y[train_mask].tolist()), Counter(y[val_mask].tolist()), Counter(y[test_mask].tolist()))
    test_metrics = test_run(x, y, A, train_mask, val_mask, test_mask, args)
    print('test ' + args.eval_metric + ' (mean, std): ', test_metrics.mean(), test_metrics.std())
    test_metric = remove_edge_pts(test_metrics, pct=args.filter_pct)
    print('test ' + args.eval_metric + ' (mean, std) after filter: ', test_metric.mean(), test_metric.std())
    return test_metrics

if __name__ == "__main__":
    """
        Test Settings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Cora', help='Name of dataset.')
    parser.add_argument('--model', default='GTCN', help='GNN models.')
    parser.add_argument('--n_hid', type=int, default=64, help='num of hidden features. For multilayer GNN, n_hid is the same for each layer')
    parser.add_argument('--num_heads', type=int, default=1, help='intermediate layer num heads (for GAT use only)')
    parser.add_argument('--num_out_heads', type=int, default=1, help='output layer num heads (for GAT use only)')
    parser.add_argument('--dropout', type=float, default=0.6, help='input dropout or GCN layer dropout (if GCN is used)')
    parser.add_argument('--dropout2', type=float, default=0.6, 
                        help='2nd dropout, used by models other than GCN, like GAT, APPNP edge dropout')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, help='weight decay in Adam optimizer.')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience.')
    parser.add_argument('--num_iter', type=int, default=1000, help='Max epochs to run.')
    parser.add_argument("--num_test", type=int, default=30, help='num of runs to test accuracy.')
    parser.add_argument("--hop", type=int, default=10, help='hop of GNN models.')
    parser.add_argument('--alpha', type=float, default=0.1, help='APPNP teleport probability.')
    parser.add_argument('--layerwise', action='store_true', default=False, 
                        help="whether to use layerwise parameters")
    parser.add_argument('--zero_init', action='store_true', default=False, 
                        help="zero initialize attention params")
    parser.add_argument('-random', '--random_label_split', action='store_true', default=False, 
                        help="use random label split or not")
    parser.add_argument("--num_train", type=int, default=20, 
                        help='number of training samples per class, used for random label split.')
    parser.add_argument("--num_val", type=int, default=30, 
                        help='number of validation samples per class, used for random label split.')
    parser.add_argument('--data_load', action='store_true', default=False, 
                        help="load the saved label split to rerun the test (for result reproduce purpose)")
    parser.add_argument("--test_id", type=int, default=1, 
                        help='number of the test, only used to record the ith number of the random label split (for reproduce purpose).')
    parser.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    parser.add_argument('--log', action='store_true', default=False, help="whether to show the training log or not")
    parser.add_argument('--eval_metric', default='acc', choices=["acc", "f1-macro", "f1-micro"], help='evaluation metrics.')
    parser.add_argument('--root_dir', default='.', help='dir of the source code.')
    args = parser.parse_args()
    test_metrics = main(args)

