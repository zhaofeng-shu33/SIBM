from ising import SIBM
from sbm import sbm_graph
from sbm import compare, get_ground_truth
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=16.0)
    parser.add_argument('--b', type=float, default=4.0)
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--repeat', type=int, default=1, help='number of times to generate the SBM graph')
    parser.add_argument('--max_iter', type=int, default=100)
    args = parser.parse_args()
    averaged_acc = 0
    for i in range(args.repeat):
        G = sbm_graph(args.n, args.k, args.a, args.b)    
        gt = get_ground_truth(G)
        sibm = SIBM(G, args.k, estimate_a_b_indicator=False,
                    _alpha=args.alpha, _beta=args.beta)
        results = sibm.metropolis(N=args.max_iter)
        acc = compare(gt, results)
        acc = int(acc) # for exact recovery
        averaged_acc += acc
    averaged_acc /= args.repeat
    print(averaged_acc)