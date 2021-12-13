# code is adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
# make sure you can run gcn.py and put example_pyg_gcn.py in the same folder before running the code
# example usage: python example_pyg_gcn.py --use_eigengnn
# EigenGNN is applied in line 42-52

import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import numpy as np
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--use_eigengnn', action='store_true',
                    help='Use EigenGNN feature.')
parser.add_argument('--use_feature', action='store_true',
                    help='Whether use node feature when using EigenGNN.')
parser.add_argument('--dim', type=int, default=32,
                    help='EigenGNN dimensionaltiy')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)
elif args.use_eigengnn:
    from EigenGNN import Eigen_multi
    temp_edge_list = data.edge_index.numpy()
    adj = sp.csr_matrix((np.ones(temp_edge_list.shape[1]), (temp_edge_list[0,:],temp_edge_list[1,:])), shape=(data.x.shape[0], data.x.shape[0]))
    new_feat = torch.from_numpy(Eigen_multi(adj = adj, d = args.dim, adj_normalize = True, feature_abs = False)).to(torch.float32)
    if args.use_feature:
        temp_n1 = torch.linalg.norm(data.x,'fro')
        temp_n2 = torch.linalg.norm(new_feat,'fro')
        data.x = torch.cat((new_feat * (temp_n1 / temp_n2), data.x), 1)
    else:
        data.x = new_feat
# notice that the feature dimensionality may change and dataset.num_features is no longer valid

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.x.shape[1], 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')