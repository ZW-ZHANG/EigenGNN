import numpy as np
import scipy.sparse as sp
import networkx as nx

def Eigen(adj, d, adj_normalize, feature_abs):
    """
    Calculate the top-d eigenvectors of the adjacency matrix
    
    Input:
    adj: adjacency matrix in scipy sparse format, could be replaced by other symmetric matrices, e.g., the Laplacian matrix
    d: the dimensionality of the eigenspace
    adj_normalize: whether to symmatrically normalize adj as in GCN
    feature_abs: whether to use absolute function 
    
    Output:
    X: n * d numpy array
    """
    if adj_normalize:
       adj = normalize_adj(adj)
    lamb, X = sp.linalg.eigs(adj, d)
    lamb, X = lamb.real, X.real
    X = X[:, np.argsort(lamb)]
    if feature_abs:
        X = np.abs(X)
    else: # a heuristic approach to ensure sign consistency of eigenvectors
        for i in range(X.shape[1]):    
            if X[np.argmax(np.absolute(X[:,i])),i] < 0:
                X[:,i] = -X[:,i]
    return X
    
def Eigen_multi(adj, d, adj_normalize, feature_abs):
    """
    Handle if the graph has multiple connected components
    Arguments are the same as Eigen    
    """
    G = nx.from_scipy_sparse_matrix(adj)
    comp = list(nx.connected_components(G))
    X = np.zeros((adj.shape[0],d))
    for i in range(len(comp)):
        node_index = np.array(list(comp[i]))
        d_temp = min(len(node_index) - 2, d)
        if d_temp < 1:
            continue
        adj_temp = adj[node_index,:][:,node_index].asfptype()
        X[node_index,:d_temp] = Eigen(adj_temp, d_temp, adj_normalize, feature_abs)
    return X

def normalize_adj(adj):
    """ Symmetrically normalize adjacency matrix."""
    """ Copy from https://github.com/tkipf/gcn """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()