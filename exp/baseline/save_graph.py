import numpy as np
from .run import load_data

if __name__ == "__main__":

    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--reorder', help="use the reordered graph", action="store_true")
    argparser.add_argument('--num-hidden', type=int, default=128)
    g, n_classes = load_data(args)
    
    ind = g.in_degrees().numpy()
    np.save("/realssd/" + args.dataset + "/in_degrees.npy", ind)
    indptr, indices, edge_ids = g.adj_sparse("csc")
    np.save("/realssd/" + args.dataset + "/indptr.npy", indptr)
    np.save("/realssd/" + args.dataset + "/indices.npy", indices)
