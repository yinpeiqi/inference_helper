import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
import tqdm
import argparse
from inference_helper import InferenceHelper, EdgeControlInferenceHelper, AutoInferenceHelper
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows

import os
from dgl.data.dgl_dataset import DGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl.backend as backend


class OtherDataset(DGLDataset):
    raw_dir = '../dataset/'

    def __init__(self, name, force_reload=False, use_reorder=False,
                verbose=False, transform=None):
        self.dataset_name = name
        self.use_reorder = use_reorder
        if name == 'friendster':
            self.num_classes = 3
        elif name == "orkut":
            self.num_classes = 10
        elif name == "livejournal1":
            self.num_classes = 50
        else:
            self.num_classes = 100
        super(OtherDataset, self).__init__(name=name,
                                                url=None,
                                                raw_dir=OtherDataset.raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

    def process(self):
        graph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '.bin')
        if not os.path.exists(graph_path):
            row = []
            col = []
            cur_node = 0
            node_mp = {}
            with open(OtherDataset.raw_dir + "com-friendster.ungraph.txt", 'r') as f:
                for line in f:
                    arr = line.split()
                    if arr[0] == '#':
                        continue
                    src, dst = int(arr[0]), int(arr[1])
                    if src not in node_mp:
                        node_mp[src] = cur_node
                        cur_node += 1
                    if dst not in node_mp:
                        node_mp[dst] = cur_node
                        cur_node += 1
                    row.append(node_mp[src])
                    col.append(node_mp[dst])
            row = np.array(row)
            col = np.array(col)
            graph = dgl.graph((row, col))
            graph = dgl.to_simple(graph)
            save_graphs(graph_path, self._graph)

        if self.use_reorder:
            reorder_graph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-reorder.bin')
            reorder_bigraph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-bi-reorder.bin')
            if not os.path.exists(reorder_graph_path) and not os.path.exists(reorder_bigraph_path):
                graphs, _ = load_graphs(graph_path)
                t1 = time.time()
                print("Reordering the graph...")
                self._graph = dgl.reorder_graph(graphs[0], node_permute_algo='rcmk', edge_permute_algo='src')
                print("Reorder is done, cost ", time.time()-t1)
                save_graphs(reorder_graph_path, [self._graph])
            
            if not os.path.exists(reorder_bigraph_path):
                graphs, _ = load_graphs(reorder_graph_path)
                self._graph = dgl.to_bidirected(graphs[0])
                save_graphs(reorder_bigraph_path, [self._graph])
            else:
                self._graph, _ = load_graphs(reorder_bigraph_path)
                self._graph = self._graph[0]
        else:
            bigraph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-bi.bin')
            if not os.path.exists(bigraph_path):
                graphs, _ = load_graphs(graph_path)
                self._graph = dgl.to_bidirected(graphs[0])
                save_graphs(bigraph_path, [self._graph])
            else:
                self._graph, _ = load_graphs(bigraph_path)  
                self._graph = self._graph[0]          

    def __getitem__(self, idx):
        assert idx == 0, "This dataset only has one graph"
        return self._graph

    def __len__(self):
        return 1

def load_other_dataset(name, dim, reorder):
    st = time.time()
    dataset = OtherDataset(name, use_reorder=reorder)
    graph = dataset[0]
    features = np.random.rand(graph.number_of_nodes(), dim)
    labels = np.random.randint(0, dataset.num_classes, size=graph.number_of_nodes())
    graph.ndata['train_mask'] = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    graph.ndata['feat'] = backend.tensor(features, dtype=backend.data_type_dict['float32'])
    graph.ndata['label'] = backend.tensor(labels, dtype=backend.data_type_dict['int64'])
    print(dataset[0])
    print(time.time()-st)
    return graph, dataset.num_classes

def load_reddit():
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    return g, data.num_classes

class OgbnDataset(DglNodePropPredDataset):
    def __init__(self, name, root = 'dataset', use_reorder = False, meta_dict = None):
        self.use_reorder = use_reorder
        super().__init__(name, root, meta_dict)

    def pre_process(self):
        if self.use_reorder:
            processed_dir = os.path.join(self.root, 'processed')
            pre_processed_file_path = os.path.join(processed_dir, 'dgl_data_processed_reorder')

            if os.path.exists(pre_processed_file_path):
                self.graph, label_dict = load_graphs(pre_processed_file_path)
                features = np.random.rand(self.graph[0].number_of_nodes(), 100)
                self.graph[0].ndata['feat'] = backend.tensor(features, dtype=backend.data_type_dict['float32'])
                if self.is_hetero:
                    self.labels = label_dict
                else:
                    self.labels = label_dict['labels']
            else:
                super().pre_process()
                t1 = time.time()
                print("Reordering the graph...")
                if isinstance(self.graph, list):
                    self.graph = self.graph[0]
                self.graph = dgl.reorder_graph(self.graph, node_permute_algo='rcmk', edge_permute_algo='src')
                print("Reorder is done, cost ", time.time()-t1)
                if self.is_hetero:
                    label_dict = self.labels
                else:
                    label_dict = {'labels': self.labels}

                print('Saving...')
                self.graph.ndata.pop("feat")
                save_graphs(pre_processed_file_path, self.graph, label_dict)
                self.graph, _ = load_graphs(pre_processed_file_path)
                features = np.random.rand(self.graph.number_of_nodes(), 100)
                self.graph.ndata['feat'] = backend.tensor(features, dtype=backend.data_type_dict['float32'])
        else:
            super().pre_process()


def load_ogb(name, reorder):
    st = time.time()
    data = OgbnDataset(name=name, use_reorder=reorder)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    # graph = dgl.to_bidirected(graph, True)
    graph = dgl.add_self_loop(graph)
    labels = labels[:, 0]
    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['label'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    print(graph)
    print("loading data:", time.time()-st)

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph, num_labels

def train(args):
    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset in ("friendster", "orkut", "livejournal1"):
        dataset = load_other_dataset(args.dataset, args.num_hidden, args.reorder)
    else:
        dataset = load_ogb(args.dataset, args.reorder)
    g : dgl.DGLHeteroGraph = dataset[0]
    nd_feat = g.ndata['feat'].numpy()
    from pathlib import Path
    dir = Path(f"./mmap/{args.dataset}_{args.reorder}/")
    dir.mkdir(parents=True, exist_ok=True)
    mmap_nd = np.memmap(dir / "feat.buf", dtype=np.float32, mode='w+', shape=nd_feat.shape)
    mmap_nd[:] = nd_feat[:]
    for nname in ['feat', 'features']:
        if nname in g.ndata:
            g.ndata.pop(nname)
    dgl.save_graphs(str(dir / "graph.bin"), [g])
    import json
    json_file = dir / "info.json"
    json.dump({"num_classes": dataset[1], "shape": mmap_nd.shape}, json_file.open("w"))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reorder', help="use the reordered graph", action="store_true")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    args = argparser.parse_args()

    train(args)
