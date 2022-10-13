import ogb
from ogb.lsc import MAG240MDataset
import tqdm
import numpy as np
import torch
import dgl
import dgl.function as fn
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default='dataset/', help='Directory to download the OGB dataset.')
parser.add_argument('--author-output-path', type=str, help='Path to store the author features.')
parser.add_argument('--inst-output-path', type=str,
                    help='Path to store the institution features.')
parser.add_argument('--graph-output-path', type=str, default='dataset/mag240m_kddcup2021/dgl_graph_processed', help='Path to store the graph.')
parser.add_argument('--graph-format', type=str, default='csc', help='Graph format (coo, csr or csc).')
parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as DGL homogeneous graph.')
parser.add_argument('--full-output-path', type=str,
                    help='Path to store features of all nodes.  Effective only when graph is homogeneous.')
args = parser.parse_args()

print('Building graph')
dataset = MAG240MDataset(root=args.rootdir)
ei_writes = dataset.edge_index('author', 'writes', 'paper')
ei_cites = dataset.edge_index('paper', 'paper')
ei_affiliated = dataset.edge_index('author', 'institution')

# We sort the nodes starting with the papers, then the authors, then the institutions.
author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions

g = dgl.heterograph({
    ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
    })

g = g.formats(args.graph_format)
dgl.save_graphs(args.graph_output_path, g)