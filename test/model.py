import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import time
from torch.profiler import profile, record_function, ProfilerActivity


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(blocks[l], h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        print(g)
        g.to(th.device('cpu'))
        
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for l, layer in enumerate(self.layers):
                    dataloader_time = 0
                    graph_to_gpu = 0
                    feat_index = 0
                    feat_to_gpu = 0
                    compute = 0
                    to_cpu = 0

                    y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

                    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                    dataloader = dgl.dataloading.NodeDataLoader(
                        g,
                        th.arange(g.num_nodes()).to(g.device),
                        sampler,
                        device=g.device if num_workers == 0 else None,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False,
                        num_workers=num_workers)

                    t0 = time.time()
                    for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                        t1 = time.time()
                        dataloader_time += t1 - t0

                        block = blocks[0]
                        block = block.int().to(device)
                        t2 = time.time()
                        graph_to_gpu += t2 - t1
                        
                        h = x[input_nodes]
                        t2d5 = time.time()
                        feat_index += t2d5 - t2
                        
                        h = h.to(device)
                        t3 = time.time()
                        feat_to_gpu += t3 - t2d5

                        h = layer(block, h)
                        if l != len(self.layers) - 1:
                            h = self.activation(h)
                            h = self.dropout(h)
                        t4 = time.time()
                        compute += t4 - t3

                        y[output_nodes] = h.cpu()
                        t5 = time.time()
                        to_cpu += t5 - t4

                        t0 = time.time()

                    print()
                    print("Time comsume:")
                    print("dataloading: {}".format(dataloader_time))
                    print("feat index: {}".format(feat_index))
                    print("feat to gpu: {}".format(feat_to_gpu))
                    print("graph to gpu: {}".format(graph_to_gpu))
                    print("inference: {}".format(compute))
                    print("feat to cpu: {}".format(to_cpu))
                    print()

                    x = y
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return y
