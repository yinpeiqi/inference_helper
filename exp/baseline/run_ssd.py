import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
import random
import tqdm
import argparse
from exp_model.gcn import StochasticTwoLayerGCN
from exp_model.sage import SAGE
from exp_model.gat import  GAT
from exp_model.jknet import JKNet
from inference_helper import SSDAutoInferenceHelper

import os
from dgl.data.dgl_dataset import DGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl.backend as backend

from inference_helper.ssd import SSDGraph

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
        reorder_bigraph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-bi-reorder.bin')
        bigraph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-bi.bin')
        graph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '.bin')
        if not os.path.exists(graph_path) and not os.path.exists(bigraph_path) and not os.path.exists(reorder_bigraph_path):
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
            save_graphs(graph_path, graph)
        
        if self.use_reorder:
            reorder_graph_path = os.path.join(OtherDataset.raw_dir, self.dataset_name + '-reorder.bin')
            if not os.path.exists(reorder_graph_path) and not os.path.exists(reorder_bigraph_path):
                graphs, _ = load_graphs(graph_path)
                t1 = time.time()
                print("Reordering the graph...")
                self._graph = dgl.reorder_graph(graphs[0], node_permute_algo='rcmk', edge_permute_algo='src')
                print("Reorder is done, cost ", time.time()-t1)
                save_graphs(reorder_graph_path, self._graph)
            
            self._graph, _ = load_graphs(reorder_graph_path)
            self._graph = self._graph[0]
            if not os.path.exists(reorder_bigraph_path):
                graphs, _ = load_graphs(reorder_graph_path)
                self._graph = dgl.to_bidirected(graphs[0])
                save_graphs(reorder_bigraph_path, [self._graph])
            else:
                self._graph, _ = load_graphs(reorder_bigraph_path)
                self._graph = self._graph[0]
        else:
            self._graph, _ = load_graphs(graph_path)  
            self._graph = self._graph[0]    
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
    labels = np.random.randint(0, dataset.num_classes, size=graph.number_of_nodes())
    graph.ndata['train_mask'] = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    graph.ndata['label'] = backend.tensor(labels, dtype=backend.data_type_dict['int64'])
    print(dataset[0])
    print(time.time()-st)
    if name in ("orkut", "livejournal1"):
        graph = dgl.add_self_loop(graph)
    return graph, dataset.num_classes

def load_reddit():
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=True)
    g = data[0]
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
                self.graph.ndata.pop('feat')
                save_graphs(pre_processed_file_path, self.graph, label_dict)
                self.graph, _ = load_graphs(pre_processed_file_path)
                if isinstance(self.graph, list):
                    self.graph = self.graph[0]
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
    graph.ndata['label'] = labels
    in_feats = 100
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(args):
    graph = SSDGraph(os.path.join('/realssd/', args.dataset))
    dim = 100
    n_classes = 3
    return (graph, n_classes), dim

def train(args):
    setup_seed(20)
    dataset, dim = load_data(args)
    g = dataset[0]

    feat = torch.as_tensor(np.memmap("/realssd/" + args.dataset + "-feat.npy", dtype=np.float32, mode='r+', shape=(g.number_of_nodes(), dim)))

    labels = np.random.randint(0, dataset[1], size=g.number_of_nodes())
    labels = backend.tensor(labels, dtype=backend.data_type_dict['int64'])

    num_classes = dataset[1]
    in_feats = feat.shape[1]
    hidden_feature = args.num_hidden
    if args.model == "GCN":
        model = StochasticTwoLayerGCN(args.num_layers, in_feats, hidden_feature, num_classes)
    elif args.model == "SAGE":
        model = SAGE(in_feats, hidden_feature, num_classes, args.num_layers, F.relu, 0.5)
    elif args.model == "GAT":
        model = GAT(args.num_layers, in_feats, hidden_feature, num_classes, [args.num_heads for _ in range(args.num_layers)], F.relu, 0.5, 0.5, 0.5, 0.5)
    elif args.model == "JKNET":
        model = JKNet(in_feats, hidden_feature, num_classes, args.num_layers)
    else:
        raise NotImplementedError()

    if args.gpu == -1:
        device = "cpu"
    else:
        device = "cuda:" + str(args.gpu)
    model = model.to(torch.device(device))
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    with torch.no_grad():
        if args.rabbit and args.reorder:
            print("rabbit")
        if args.ratio:
            print("Ratio:", args.ratio)

        if args.auto:
            if args.reorder and args.rabbit:
                np_array = np.load("rabbit/" + args.dataset + ".npy")
                nids = torch.tensor(np_array).to(g.device)
            elif args.reorder:
                nids = torch.arange(g.number_of_nodes())
            else:
                nids = torch.randperm(g.number_of_nodes())
            print(args.num_layers, args.model, "auto", args.dataset, args.num_heads, args.num_hidden, "reorder" if args.reorder else "", "SSD")
            helper = SSDAutoInferenceHelper(model, torch.device(device), use_uva = args.use_uva, free_rate=args.free_rate, use_random=not args.reorder, debug = args.debug)
            helper.dataset_name = args.dataset
            helper.ret_shapes = helper._trace_output_shape((feat,))
            torch.cuda.synchronize()
            st = time.time()
            helper_pred = helper.inference(g, feat)
            cost_time = time.time() - st
            helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
            print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))
            if os.path.exists("/realssd/feat_output.npy"):
                os.remove("/realssd/feat_output.npy")
            if os.path.exists("/realssd/feat_relu_2.npy"):
                os.remove("/realssd/feat_relu_2.npy")
            if os.path.exists("/realssd/feat_mean.npy"):
                os.remove("/realssd/feat_mean.npy")

        else:
            if args.gpu == -1:
                print(args.num_layers, args.model, "CPU", args.batch_size, args.dataset, args.num_heads, args.num_hidden)
            else:
                print(args.num_layers, args.model, "GPU", args.batch_size, args.dataset, args.num_heads, args.num_hidden)
            st = time.time()
            if args.reorder:
                if args.rabbit:
                    np_array = np.load("~/inference_helper/rabbit/" + args.dataset + ".npy")
                    nids = torch.tensor(np_array).to(g.device)
                else:
                    nids = torch.arange(g.number_of_nodes()).to(g.device)
            else:
                nids = torch.randperm(g.number_of_nodes()).to(g.device)
            pred = model.inference(g, args.batch_size, torch.device(device), feat, nids, args.use_uva, args.ssd)
            cost_time = time.time() - st
            func_score = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
            if args.gpu != -1:
                print("max memory:", torch.cuda.max_memory_allocated() // 1024 ** 2)
            print("Origin Inference: {}, inference time: {}".format(func_score, cost_time))
        print("\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reorder', help="use the reordered graph", action="store_true")
    argparser.add_argument('--rabbit', help="use the rabbit reordered graph", action="store_true")
    argparser.add_argument('--use-uva', help="use the pinned memory", action="store_true")
    argparser.add_argument('--free-rate', help="free memory rate", type=float, default=0.9)
    argparser.add_argument('--ratio', help="", type=float, default=None)

    # Different inference mode. 
    argparser.add_argument('--topdown', action="store_true")
    argparser.add_argument('--cpufull', action="store_true")
    argparser.add_argument('--gpufull', action="store_true")
    argparser.add_argument('--gpu', help="GPU device ID. Use -1 for CPU training", type=int, default=0)
    argparser.add_argument('--auto', action="store_true")
    # argparser.add_argument('--l', action='append', default=[5], help='<Required> Set flag')

    argparser.add_argument('--model', help="can be GCN, GAT, SAGE and JKNET", type=str, default='GCN')
    argparser.add_argument('--debug', action="store_true")
    argparser.add_argument('--num-epochs', type=int, default=0)
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-heads', type=int, default=2)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=2000)
    argparser.add_argument('--load-data', action="store_true")
    args = argparser.parse_args()

    if args.load_data:
        g, n_classes = load_data(args)
        print(g)
        
    else:
        train(args)
