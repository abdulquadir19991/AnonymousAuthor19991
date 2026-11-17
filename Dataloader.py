import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import torch
from sklearn.preprocessing import normalize


def construct_adj_hat(adj, self_loop=1.0):
    adj = ss.coo_matrix(adj, dtype=np.float64)
    adj = adj.maximum(adj.T)                       
    adj = ss.csr_matrix(adj)
    adj.setdiag(0)                                
    adj.eliminate_zeros()
    adj_ = adj + self_loop * ss.eye(adj.shape[0])
    rowsum = np.asarray(adj_.sum(1)).flatten()
    d_inv_sqrt = ss.diags(np.power(rowsum, -0.5))
    return (d_inv_sqrt @ adj_ @ d_inv_sqrt).tocoo()


def to_torch(adj_hat, device, sparse=False, dtype=torch.float64):
    if sparse:
        idx = torch.from_numpy(np.vstack([adj_hat.row, adj_hat.col])).long()
        val = torch.from_numpy(adj_hat.data)
        return torch.sparse_coo_tensor(idx, val, adj_hat.shape).coalesce().to(device, dtype)
    return torch.from_numpy(adj_hat.toarray()).to(device, dtype)


def load_graph(path, dataset, device, self_loop=1.0, sparse_adj=False,
               dtype=torch.float64):

    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    if ss.issparse(features):
        features = features.toarray()
    features = normalize(np.asarray(features, dtype=np.float64))   

    labels = np.unique(np.asarray(data['Y']).flatten(), return_inverse=True)[1]
    n_class = int(labels.max()) + 1
    print('dataset: {} | # node: {} | # class: {}'.format(dataset, features.shape[0], n_class))

    adj_hat = to_torch(construct_adj_hat(data['adj'], self_loop), device, sparse_adj, dtype)
    features = torch.from_numpy(features).to(device, dtype)
    labels = torch.from_numpy(labels).long().to(device)
    return adj_hat, features, labels, n_class, data


def generate_partition(data):
    idx_train = np.asarray(data['train']).flatten().astype(np.int64)
    idx_val = np.asarray(data['val']).flatten().astype(np.int64)
    idx_test = np.asarray(data['test']).flatten().astype(np.int64)
    return idx_train, idx_val, idx_test


def generate_partition_random(labels, num_perclass=20, num_val=500, num_test=1000, seed=0):
    
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    remaining = {c: num_perclass for c in np.unique(labels)}
    idx_train, rest = [], []
    for i in perm:
        if remaining[labels[i]] > 0:
            remaining[labels[i]] -= 1
            idx_train.append(i)
        else:
            rest.append(i)
    idx_val = rest[:num_val]
    idx_test = rest[num_val:num_val + num_test]
    return (np.asarray(idx_train, dtype=np.int64),
            np.asarray(idx_val, dtype=np.int64),
            np.asarray(idx_test, dtype=np.int64))


def make_split(data, labels, rand_split, seed=0, num_perclass=20):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if rand_split:
        tr, va, te = generate_partition_random(labels, num_perclass, seed=seed)
    else:
        tr, va, te = generate_partition(data)
    assert not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    print('train: {}, val: {}, test: {}'.format(len(tr), len(va), len(te)))
    return tr, va, te


def load_data(path, dataset, rand_split, device, seed=0):
    adj_hat, features, labels, n_class, data = load_graph(path, dataset, device)
    tr, va, te = make_split(data, labels, rand_split, seed)
    return adj_hat, features, labels, n_class, tr, va, te


def load_tu_graphs(name, root='./data/TU', max_degree_onehot=None):
    from torch_geometric.datasets import TUDataset   
    ds = TUDataset(root, name=name)
    raw, max_deg = [], 0
    for g in ds:
        ei = g.edge_index.numpy()
        n = int(g.num_nodes)
        deg = np.bincount(ei[0], minlength=n) if ei.size else np.zeros(n, dtype=np.int64)
        max_deg = max(max_deg, int(deg.max()) if n else 0)
        raw.append((ei, n, None if g.x is None else g.x.numpy().astype(np.float64), deg))
    max_deg = max_deg if max_degree_onehot is None else min(max_deg, max_degree_onehot)
    graphs = []
    for ei, n, x, deg in raw:
        if x is None:
            x = np.zeros((n, max_deg + 1))
            x[np.arange(n), np.minimum(deg, max_deg)] = 1.0
        graphs.append((ei, n, x))
    labels = np.unique(ds.y.numpy(), return_inverse=True)[1]
    print('dataset: {} | # graphs: {} | # class: {}'.format(name, len(graphs), labels.max() + 1))
    return graphs, labels



