# Graph Tree Networks (GTNets)
This folder contains the code to reproduce the results in our paper GTNet: A Tree-Based Deep Graph Learning Architecture.

## Datasets
There are 4 datasets used for test (data from DGL library): 
1. Cora, for which the train/validation/test lable split is provided in DGL dataset.
2. Citeseer, for which the lable split is provided in DGL dataset.
3. PubMed, for which the lable split is provided in DGL dataset.
4. MS Coauthor-CS, for which there are 3 random labels splits generated (train_mask: 20 samples/class, validation_mask; 30 samples/class, test_mask: all remaining samples). The random 3 label splits used in the paper are in the folder "label_split".

## Usage
Test a model using command: `python train.py` with the optional input arguments:

`--data`, default=`Cora`, choices=[`Cora`, `Citeseer`, `Pubmed`, `Coauthor-CS`], help=`Name of dataset`.

`--model`, default=`GTCN`, choices=[`GCN`, `GAT`, `APPNP`, `DAGNN`, `TreeLSTM`, `GTCN`, `GTAN`, `GCNII`], help=`GNN models`.

`--n_hid`, type=`int`, default=`64`, help=`num of hidden features`. For multilayer GNN, `n_hid` is the same for each layer.

`--num_heads`, type=`int`, default=`1`, help=`intermediate layer num heads` (for GAT use only).

`--num_out_heads`, type=`int`, default=`1`, help=`output layer num heads` (for GAT use only).

`--dropout`, type=`float`, default=`0.6`, help=`input dropout` or `GCN layer dropout` (if GCN is used).

`--dropout2`, type=`float`, default=`0.6`, help=`2nd dropout`, used by models other than GCN, like GAT, APPNP edge dropout.

`-lr`, `--learning_rate`, type=`float`, default=`0.01`, help=`earning rate`.

`-wd`, `--weight_decay`, type=`float`, default=`0.0005`, help=`weight decay in Adam optimizer`.

`--patience`, type=`int`, default=`200`, help=`Early stopping patience`.

`--num_iter`, type=`int`, default=`1000`, help=`Max epochs to run`.

`--num_test`, type=`int`, default=`30`, help=`num of runs to test accuracy`.

`--hop`, type=`int`, default=`10`, help=`hop of GNN models`.

`--alpha`, type=`float`, default=`0.1`, help=`APPNP teleport probability`.

`--layerwise`, action=`store_true`, default=`False`, help=`whether to use layerwise parameters for GTAN`.

`--zero_init`, action=`store_true`, default=`False`, help=`zero initialize attention params`.

`-random`, `--random_label_split`, action=`store_true`, default=`False`, help=`use random label split or not`.

`--num_train`, type=`int`, default=`20`, help=`number of training samples per class, used for random label split`.

`--num_val`, type=`int`, default=`30`, help=`number of validation samples per class, used for random label split`.

`--data_load`, action=`store_true`, default=`False`, help=`load the saved label split to rerun the test` (for result reproduce purpose).

`--test_id`, type=`int`, default=`1`, help=`number of the test`, only used to record the i-th number of the random label split (for reproduce purpose).

`--filter_pct`, type=`float`, default=`0.1`, help=`remove the top and bottom filer_pct points before obtaining statistics of test accuracy`.

`--log`, action=`store_true`, default=`False`, help=`whether to show the training log or not`.

`--eval_metric`, default=`acc`, choices=[`acc`, `f1-macro`, `f1-micro`], help=`evaluation metrics`.

`--root_dir`, default=`.`, help=`dir of the source code`.

Examples: 
1. Run 10-hop GTCN with Cora dataset and accuracy metric: `python train.py`
2. Run 10-hop GTCN with Citeseer dataset and accuracy metric: `python train.py --data Citeseer --dropout 0.8`
3. Run 10-hop GTAN with Coauthor-CS dataset, loaded data split (#3), 2 number of runs and f1-macro metric: `python train.py --data Coauthor-CS --model GTAN --dropout 0.2 --dropout2 0.2 -wd 5e-3 --zero_init --num_test 2 --data_load --test_id 2 --eval_metric f1-macro`
4. Run 5-hop GTAN with Cora dataset and accuracy metric: `python train.py --model GTAN --hop 5 --patience 300 --layerwise`
5. Run 10-hop DAGNN with Coauthor-CS dataset, loaded data split (#2) and f1-macro metric: `python train.py --data Coauthor-CS --model DAGNN --dropout 0.8 -wd 5e-3 --data_load --eval_metric f1-macro`

Note that for Coauthor-CS data, you can either use the label splits we created by turning on "--data_load" argument, or use the new generated random split by turning on "random_label_split" and without "--data_load".

## Results in the paper
All results are recorded in Jupyter files, which are clear and organized in the folder "result", where both accuracy and macro-f1 results are included. The Jupyter files include model name and hop# (except APPNP and DAGNN which use fixed 10 hops). For instance, for GTCN with 10 hops, the corresponding Jupyter file is *acc_test-GTCN_hop10.ipynb*.

In each Jupyter file, it has the run test of 4 datasets - Cora, Citeseer, PubMed, Coauthor-CS (3-split tests). For each dataset test, it records all 30 runs' result - best validation accuracy (at early stop) and test accuracy.

## Tools used by the source code
1. Pytorch 1.9.1 (CUDA 11.1)
2. DGL 0.6.1 (note that DGL 0.7x made some changes to the Coauthor-CS data such that the label split is not balanced)
3. Anaconda 4.10.3, Python 3.8.8
4. Imblearn 0.8.0 (only used for label split)
5. Torch_geometric CUDA version


## ogbn-arxiv
The code for tests on ogbn-arxiv dataset is in the folder *ogbn_arxiv*, check README in *ogbn_arxiv* for the guide to run the code.

## Runtime test:
We made a torch_geometric version of all models except GTAN, all models are in `torch_geometric_models.py`, which will be used when running `runtime_test.py`.
