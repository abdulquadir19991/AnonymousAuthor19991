import argparse
import warnings

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from Dataloader import load_tu_graphs
from GraphLevelBLS import (build_batch, graph_kernels_exact, graph_features_rff,
                           layer_scores_kernel, layer_scores_features, ensemble_accuracy)

warnings.filterwarnings("ignore")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='./data/TU')
    p.add_argument('--dataset', default='MUTAG')
    p.add_argument('--model', default='gckebls', choices=['gckebls', 'gckebls_e'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ensemble', default='mean', choices=['mean', 'vote'])
    p.add_argument('--q', type=int, default=1)
    p.add_argument('--groups', type=int, default=1)
    p.add_argument('--n_rff', type=int, default=512)
    p.add_argument('--normalize', action='store_true', help='cosine-normalise K_G (exact model)')
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument('--Cs', type=float, nargs='+', default=[10.0 ** k for k in range(-5, 6)])
    p.add_argument('--max_layers', type=int, default=5)
    return p.parse_args()


def representations(args, A, X, P, sigma):
    kw = dict(sigma=sigma, n_layer=args.max_layers, n_groups=args.groups, q=args.q)
    if args.model == 'gckebls':
        return graph_kernels_exact(A, X, P, normalize=args.normalize, **kw), layer_scores_kernel
    return graph_features_rff(A, X, P, n_rff=args.n_rff, seed=args.seed, **kw), layer_scores_features


if __name__ == '__main__':
    args = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, y = load_tu_graphs(args.dataset, args.root)
    n_class = int(y.max()) + 1
    A, X, P = build_batch(graphs, device)

    reps = {s: representations(args, A, X, P, s) for s in args.sigmas}

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    accs = []
    for k, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all], random_state=args.seed)
        best = (-1, None)
        for s, (R, scorer) in reps.items():
            for C in args.Cs:
                sc = scorer(R, y, tr, va, C, n_class)
                for L in range(1, args.max_layers + 1):
                    v = ensemble_accuracy(sc, y[va], L, args.ensemble)
                    if v > best[0]:
                        best = (v, (s, C, L))
        s, C, L = best[1]
        R, scorer = reps[s]
        sc = scorer(R, y, tr_all, te, C, n_class)               
        acc = ensemble_accuracy(sc, y[te], L, args.ensemble)
        accs.append(acc)
        print('fold {} | sigma={} C={:g} L={} | val {:.4f} | TEST {:.4f}'.format(k, s, C, L, best[0], acc))

    accs = np.array(accs) * 100
    print('{} | {} | TEST ACC: {:.2f} +- {:.2f} ({}-fold CV)'.format(
        args.dataset, args.model, accs.mean(), accs.std(ddof=1), args.folds))
