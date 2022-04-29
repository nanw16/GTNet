# Test for obgn-arxiv data.

## Usage
Test a model using command: python train_ogbn_arxiv.py
    with the optional input arguments:
	'--model', default='GTAN', choices=["GCN", "GAT", "APPNP", "DAGNN", "TreeLSTM", "GTCN", "GTAN"], help='GNN models.'
    	'--n_hid', type=int, default=128 help='num of hidden features. For multilayer GNN, n_hid is the same for each layer'
	'--num_heads', type=int, default=1, help='intermediate layer num heads (for GAT use only)'
	'--num_out_heads', type=int, default=1, help='output layer num heads (for GAT use only)'
	'--dropout', type=float, default=0.2, help='input dropout or GCN layer dropout (if GCN is used)'
	'--dropout2', type=float, default=0, help='2nd dropout, used by models other than GCN, like GAT, APPNP edge dropout'
	'-lr', '--learning_rate', type=float, default=0.01, help='Learning rate.'
	'-wd', '--weight_decay', type=float, default=0.00005, help='weight decay in Adam optimizer.'
	'--num_iter', type=int, default=1000, help='Max epochs to run.'
	"--num_test", type=int, default=30, help='num of runs to test accuracy.'
	"--hop", type=int, default=10, help='hop of GNN models.'
	'--alpha', type=float, default=0.1, help='APPNP teleport probability.'
	'--layerwise', action='store_true', default=False, help='whether to use layerwise parameters for GTAN'
	'--zero_init', action='store_true', default=False, help='zero initialize attention params'
	'--eval_metric', default='acc', choices=["acc", "f1-macro", "f1-micro"], help='evaluation metrics.'
	'--log_steps', default=100, help='training log steps.'

Examples: 
	1. Use GTAN: python train_ogbn_arxiv.py
	2. Use GTCN: python train_ogbn_arxiv.py --model GTCN --n_hid 256 --dropout2 0.2 --hop 5
	3. Use APPNP: python train_ogbn_arxiv.py --model APPNP --n_hid 256 --hop 5

DAGNN can tested either using the above command, or using train_DAGNN.py - a version from the original author (https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/main_ogbnarxiv.py).

Running hardware: The results are obtained by RTX 3060 12GB. 
Note: GTAN 4-layer with 128 hidden features requires 10.5 GB VRAM, same as GAT. Therefore, we only choose 128 hidden features instead of commonly used 256 features by other models.

All the results are recorded in Jupyter notebooks. Check the names of the Jupyter notebook files for corresponding results.

## Tools used by the source code
1. Pytorch 1.9.1 (CUDA 11.1)
2. Torch_geometric (CUDA 11.1)
3. torch_scatter
4. ogb
5. DGL 0.6.1
6. Anaconda 4.10.3, Python 3.8.8
7. Scikit-learn
8. Imblearn