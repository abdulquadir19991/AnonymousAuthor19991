import math
from numbers import Number

import torch
import torch.nn.functional as F


def spmm(A, X):
    return torch.sparse.mm(A, X) if A.is_sparse else A @ X


def aggregate(A, X, q=1):
    for _ in range(q):
        X = spmm(A, X)
    return X


def aggregate_gram(A, G, q=1):
    for _ in range(q):
        G = spmm(A, spmm(A, G).T)      
    return G


def sigma_to_gamma(sigma):
    return 1.0 / (2.0 * float(sigma) ** 2)


def rbf_from_features(X, sigma):
    sq = (X * X).sum(1, keepdim=True)
    D2 = (sq + sq.T - 2.0 * (X @ X.T)).clamp_min_(0.0)
    return torch.exp(-sigma_to_gamma(sigma) * D2)


def rbf_from_gram(G, sigma):
    d = torch.diagonal(G).unsqueeze(0)
    D2 = (d + d.T - 2.0 * G).clamp_min_(0.0)
    return torch.exp(-sigma_to_gamma(sigma) * D2)


def group_sigmas(sigma, m, spread=4.0):
    if m == 1:
        return [float(sigma)]
    if spread <= 1.0:
        raise ValueError("sigma_spread must be > 1 when n_groups > 1, "
                         "otherwise the feature groups are identical.")
    exps = torch.linspace(-1.0, 1.0, m, dtype=torch.float64)
    return [float(sigma) * float(spread) ** float(e) for e in exps]


def ridge_solve(H, Y, C):
    n, d = H.shape
    if n >= d:
        M = H.T @ H + C * torch.eye(d, dtype=H.dtype, device=H.device)
        return torch.linalg.solve(M, H.T @ Y)
    M = H @ H.T + C * torch.eye(n, dtype=H.dtype, device=H.device)
    return H.T @ torch.linalg.solve(M, Y)


def combine_layers(layer_scores, rule="mean"):
    S = torch.stack(layer_scores)                      
    if rule == "mean":
        return S.mean(0)
    if rule == "vote":
        votes = F.one_hot(S.argmax(-1), S.shape[-1])    
        return votes.sum(0).to(S.dtype)                 
    raise ValueError("rule must be 'mean' or 'vote'")


def _as_index(idx, device):
    return torch.as_tensor(idx, dtype=torch.long, device=device)


class _EnsembleMixin:

    def predict_scores(self, C=None, n_layers=None, rule=None):
        C = self.Cs_[0] if C is None else C
        scores = self.layer_scores_[C][: (n_layers or self.n_layer)]
        return combine_layers(scores, rule or self.ensemble_rule)

    def predict(self, C=None, n_layers=None, rule=None):
        return self.predict_scores(C, n_layers, rule).argmax(1)

    def accuracy(self, y, idx, C=None, n_layers=None, rule=None):
        idx = _as_index(idx, y.device)
        pred = self.predict(C, n_layers, rule).to(y.device)
        return (pred[idx] == y[idx]).float().mean().item()

    def _prepare(self, X, y, idx_train, n_class, C):
        idx_train = _as_index(idx_train, X.device)
        n_class = int(n_class) if n_class is not None else int(y.max()) + 1
        Y = F.one_hot(y.to(X.device)[idx_train], n_class).to(X.dtype)
        C = self.C if C is None else C
        self.Cs_ = [float(C)] if isinstance(C, Number) else [float(c) for c in C]
        self.layer_scores_ = {c: [] for c in self.Cs_}
        return idx_train, Y


class GraphkernelBLS(_EnsembleMixin):

    def __init__(self, sigma=1.0, C=1.0, n_layer=10, n_groups=1,
                 sigma_spread=4.0, q=1, sigma_enh=None, ensemble_rule="mean"):
        self.sigma = sigma
        self.C = C
        self.n_layer = n_layer
        self.n_groups = n_groups
        self.sigma_spread = sigma_spread
        self.q = q
        self.sigma_enh = sigma if sigma_enh is None else sigma_enh
        self.ensemble_rule = ensemble_rule
        self.sigmas_ = group_sigmas(sigma, n_groups, sigma_spread)

    def _layer(self, A, H):
        Xbar = aggregate(A, H, self.q)                                  
        K = [rbf_from_features(Xbar, s) for s in self.sigmas_]
        G = sum(K) / len(K)
        E = rbf_from_gram(aggregate_gram(A, G, self.q), self.sigma_enh)  
        return K + [E]                                                   

    def fit(self, X, y, A, idx_train, n_class=None, C=None):

        idx_train, Y = self._prepare(X, y, idx_train, n_class, C)
        H = X
        for _ in range(self.n_layer):
            blocks = self._layer(A, H)
            H = torch.hstack(blocks)                       

            H_land = torch.hstack([B[:, idx_train] for B in blocks])  
            H_tr = H_land[idx_train]
            del blocks
            for c in self.Cs_:
                W = ridge_solve(H_tr, Y, c)                
                self.layer_scores_[c].append(H_land @ W)   
        return self


def rff_map(X, Omega):
  
    P = X @ Omega
    return torch.hstack([torch.cos(P), torch.sin(P)]) / math.sqrt(Omega.shape[1])


class GraphkernelBLS_E(_EnsembleMixin):
   
    def __init__(self, sigma=1.0, C=1.0, n_layer=10, n_groups=1,
                 sigma_spread=4.0, q=1, n_rff=512, n_rff_enh=None,
                 sigma_enh=None, ensemble_rule="mean", seed=0):
        self.sigma = sigma
        self.C = C
        self.n_layer = n_layer
        self.n_groups = n_groups
        self.sigma_spread = sigma_spread
        self.q = q
        self.n_rff = n_rff
        self.n_rff_enh = n_rff if n_rff_enh is None else n_rff_enh
        self.sigma_enh = sigma if sigma_enh is None else sigma_enh
        self.ensemble_rule = ensemble_rule
        self.seed = seed
        self.sigmas_ = group_sigmas(sigma, n_groups, sigma_spread)

    def _omega(self, d, D, sigma, gen, like):
        W = torch.randn(d, D, generator=gen, dtype=torch.float64) / float(sigma)
        return W.to(dtype=like.dtype, device=like.device)

    def _layer(self, A, H, gen):
        Xbar = aggregate(A, H, self.q)                                        
        d = Xbar.shape[1]
        Z = [rff_map(Xbar, self._omega(d, self.n_rff, s, gen, Xbar))         
             for s in self.sigmas_]                                          
        Z = torch.hstack(Z) / math.sqrt(len(Z))
        Zbar = aggregate(A, Z, self.q)
        E = rff_map(Zbar, self._omega(Zbar.shape[1], self.n_rff_enh,
                                      self.sigma_enh, gen, Zbar))
        return torch.hstack([Z, E])                                           

    def fit(self, X, y, A, idx_train, n_class=None, C=None):
        idx_train, Y = self._prepare(X, y, idx_train, n_class, C)
        gen = torch.Generator().manual_seed(int(self.seed))
        H = X
        for _ in range(self.n_layer):
            H = self._layer(A, H, gen)
            H_tr = H[idx_train]
            for c in self.Cs_:
                W = ridge_solve(H_tr, Y, c)                
                self.layer_scores_[c].append(H @ W)
        return self
