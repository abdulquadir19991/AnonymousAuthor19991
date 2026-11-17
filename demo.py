import argparse
import itertools
import warnings

import numpy as np
import torch

from Dataloader import load_graph, make_split
from GraphkernelBLS import GraphkernelBLS, GraphkernelBLS_E

warnings.filterwarnings("ignore")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='./data/')
    p.add_argument('--dataset', default='ACM')
    p.add_argument('--model', default='gckebls_e', choices=['gckebls', 'gckebls_e'])
    p.add_argument('--rand_split', action='store_true')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--seed', type=int, default=None,
               help='base seed; omit for a fresh random seed each execution')
    p.add_argument('--ensemble', default='mean', choices=['mean', 'vote'])
    p.add_argument('--self_loop', type=float, default=1.0)
    p.add_argument('--sparse_adj', action='store_true')
    p.add_argument('--q', type=int, default=1)
    p.add_argument('--n_rff', type=int, default=512)
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument('--groups', type=int, nargs='+', default=[1, 3, 5])
    p.add_argument('--Cs', type=float, nargs='+', default=[10.0 ** k for k in range(-5, 6)])
    p.add_argument('--max_layers', type=int, default=10)
    return p.parse_args()


def build(args, sigma, m, seed):
    common = dict(sigma=sigma, n_layer=args.max_layers, n_groups=m,
                  q=args.q, ensemble_rule=args.ensemble)
    if args.model == 'gckebls':
        return GraphkernelBLS(**common)
    return GraphkernelBLS_E(n_rff=args.n_rff, seed=seed, **common)


def run_once(args, A, X, y, n_class, tr, va, te, seed):
    best = (-1.0, None)
    for sigma, m in itertools.product(args.sigmas, args.groups):
        model = build(args, sigma, m, seed).fit(X, y, A, tr, n_class=n_class, C=args.Cs)
        for C in args.Cs:
            for L in range(1, args.max_layers + 1):          
                val = model.accuracy(y, va, C=C, n_layers=L)
                if val > best[0]:
                    test = model.accuracy(y, te, C=C, n_layers=L)
                    train = model.accuracy(y, tr, C=C, n_layers=L)
                    best = (val, dict(sigma=sigma, m=m, C=C, L=L,
                                      train=train, val=val, test=test))
        del model
    return best[1]


if __name__ == '__main__':
    args = parse()
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy % (2 ** 31))   
    print('base seed: {}'.format(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    A, X, y, n_class, data = load_graph(args.path, args.dataset, device,
                                        self_loop=args.self_loop, sparse_adj=args.sparse_adj)
    results = []
    for r in range(args.runs):
        seed = args.seed + r
        tr, va, te = make_split(data, y, args.rand_split, seed=seed)   
        res = run_once(args, A, X, y, n_class, tr, va, te, seed)
        print('run {} | sigma={sigma} m={m} C={C:g} L={L} | train {train:.4f} '
              'val {val:.4f} TEST {test:.4f}'.format(r, **res))
        results.append(res['test'])

    acc = np.array(results) * 100
    print('{} | {} | {} split | ensemble={} | TEST ACC: {:.2f} +- {:.2f} (over {} runs)'.format(
        args.dataset, args.model, 'random' if args.rand_split else 'standard',
        args.ensemble, acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0.0, len(acc)))
