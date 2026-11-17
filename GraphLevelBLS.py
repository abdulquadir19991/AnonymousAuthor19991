import math

import numpy as np
import scipy.sparse as ss
import torch

from Dataloader import construct_adj_hat
from GraphkernelBLS import (aggregate, aggregate_gram, rbf_from_gram, group_sigmas,
                            rff_map, ridge_solve, combine_layers)


def build_batch(graphs, device, self_loop=1.0, dtype=torch.float64):
    blocks, xs, sizes = [], [], []
    for ei, n, x in graphs:
        a = ss.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        blocks.append(construct_adj_hat(a, self_loop))
        xs.append(x)
        sizes.append(n)
    A = ss.block_diag(blocks).tocoo()
    A = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([A.row, A.col])).long(),
                                torch.from_numpy(A.data), A.shape).coalesce().to(device, dtype)
    X = torch.from_numpy(np.vstack(xs)).to(device, dtype)
    graph_id = np.repeat(np.arange(len(graphs)), sizes)
    P = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([graph_id, np.arange(len(graph_id))])).long(),
        torch.ones(len(graph_id), dtype=torch.float64),
        (len(graphs), len(graph_id))).coalesce().to(device, dtype)
    return A, X, P


def _pool_gram(P, G):
    return torch.sparse.mm(P, torch.sparse.mm(P, G).T)


def graph_kernels_exact(A, X, P, sigma=1.0, n_layer=5, n_groups=1, sigma_spread=4.0,
                        q=1, sigma_enh=None, normalize=False):
    sigma_enh = sigma if sigma_enh is None else sigma_enh
    sigmas = group_sigmas(sigma, n_groups, sigma_spread)
    G = X @ X.T                                    
    out = []
    for _ in range(n_layer):
        Gbar = aggregate_gram(A, G, q)                              
        Kmean = sum(rbf_from_gram(Gbar, s) for s in sigmas) / len(sigmas)   
        E = rbf_from_gram(aggregate_gram(A, Kmean, q), sigma_enh)  
        G = Kmean + E                                              
        KG = _pool_gram(P, G)                                      
        if normalize:                                              
            d = torch.sqrt(torch.diagonal(KG)).unsqueeze(0)
            KG = KG / (d.T * d)
        out.append(KG)
        del Gbar, Kmean, E
    return out


def graph_features_rff(A, X, P, sigma=1.0, n_layer=5, n_groups=1, sigma_spread=4.0,
                       q=1, n_rff=512, n_rff_enh=None, sigma_enh=None, seed=0):
    n_rff_enh = n_rff if n_rff_enh is None else n_rff_enh
    sigma_enh = sigma if sigma_enh is None else sigma_enh
    sigmas = group_sigmas(sigma, n_groups, sigma_spread)
    gen = torch.Generator().manual_seed(int(seed))

    def omega(d, D, s, like):
        return (torch.randn(d, D, generator=gen, dtype=torch.float64) / float(s)).to(like)

    H, out = X, []
    for _ in range(n_layer):
        Xbar = aggregate(A, H, q)
        Z = torch.hstack([rff_map(Xbar, omega(Xbar.shape[1], n_rff, s, Xbar)) for s in sigmas])
        Z = Z / math.sqrt(len(sigmas))
        Zbar = aggregate(A, Z, q)
        E = rff_map(Zbar, omega(Zbar.shape[1], n_rff_enh, sigma_enh, Zbar))
        H = torch.hstack([Z, E])
        out.append(torch.sparse.mm(P, H))          
    return out


def _one_hot(y, n_class, like):
    return torch.nn.functional.one_hot(torch.as_tensor(y, device=like.device), n_class).to(like.dtype)


def layer_scores_kernel(K_list, y, idx_tr, idx_ev, C, n_class):
    tr = torch.as_tensor(idx_tr, device=K_list[0].device)
    ev = torch.as_tensor(idx_ev, device=K_list[0].device)
    Y = _one_hot(y[idx_tr], n_class, K_list[0])
    scores = []
    for K in K_list:
        Ktr = K[tr][:, tr]
        alpha = torch.linalg.solve(Ktr + C * torch.eye(len(tr), dtype=K.dtype, device=K.device), Y)
        scores.append(K[ev][:, tr] @ alpha)
    return scores


def layer_scores_features(U_list, y, idx_tr, idx_ev, C, n_class):
    tr = torch.as_tensor(idx_tr, device=U_list[0].device)
    ev = torch.as_tensor(idx_ev, device=U_list[0].device)
    Y = _one_hot(y[idx_tr], n_class, U_list[0])
    return [U[ev] @ ridge_solve(U[tr], Y, C) for U in U_list]


def ensemble_accuracy(scores, y_true, n_layers=None, rule='mean'):
    pred = combine_layers(scores[: n_layers or len(scores)], rule).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y_true)).mean())
