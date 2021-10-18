# EigenGNN
This is a sample implementation of "Eigen-GNN: a Graph Structure Preserving Plug-in for GNNs, TKDE 2021". [(Paper)](https://zw-zhang.github.io/files/2021_TKDE_EigenGNN.pdf)

### Requirements
* numpy
* scipy
* networkx

### Usage
As indicated by the title, EigenGNN is a plug-in instead of a stand-along model. Therefore, the code in the main file will only generate initial node representations (i.e., eigenvectors of a graph structure matrix), which can be used together with other GNNs.

The initial node representation can be generated as follows:
```bash
from EigenGNN import Eigen_multi
features = Eigen_multi(adj, d, adj_normalize, feature_abs)
```

An example to use EigenGNN with GCN being the GNN backbone for node classification benchmark datasets (e.g., Cora, CiteSeer, PubMed) is provided in `example_gcn.py` and can be run as:
```bash
python example_gcn.py --method eigen --dataset cora
```
for only using the eigenspace, or 
```bash
python example_gcn.py --method feat_eigen --dataset cora
```
for concatenating the eigenspace with input node features.

Another example to use EigenGNN with GIN being the GNN backbone for graph isomorphism test (i.e., graph classification) is provided in `example_gin.py` and can be run as:
```bash
python example_gin.py --cv-fold 0 --model-type EigenGin --num-mlp-hidden 1 --num-gnn-layers 1 --out-weight-dir dir1 --out-log-dir dir2
```
 
We will provide more examples in the near future.

### Cite
If you find this code useful, please cite our paper:
```
@article{zhang2021eigengnn,
  title={Eigen-GNN: a Graph Structure Preserving Plug-in for GNNs},
  author={Zhang, Ziwei and Cui, Peng and Pei, Jian and Wang, Xin and Zhu, Wenwu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```