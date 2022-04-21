import torch.fx
import torch as th
import torch.nn as nn
import gc

import dgl
from dgl.nn import GATConv
import tqdm
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.hidden_features = num_hidden
        self.heads = heads
        self.out_features = num_classes
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree=True))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g[l], h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g[-1], h).mean(1)
        return logits

    def forward_full(self, g, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

    def inference(self, g, batch_size, device, x, use_uva = False):
        for k in list(g.ndata.keys()):
            g.ndata.pop(k)
        for k in list(g.edata.keys()):
            g.edata.pop(k)

        torch.cuda.reset_peak_memory_stats()
        for l, layer in enumerate(self.gat_layers):
            gc.collect()
            th.cuda.empty_cache()
            if l != self.num_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.heads[l] * self.hidden_features)
            else:
                y = th.zeros(g.number_of_nodes(), self.out_features)

            nids = th.arange(g.number_of_nodes()).to(g.device)
            if use_uva:
                pin_memory_inplace(x)
                nids = nids.to(device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, nids, sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                use_uva=use_uva,
                device=device,
                num_workers=0)
            memorys = []
            a, b, c, d, e = 0, 0, 0, 0, 0
            import time
            sss = time.time()
            t0 = time.time()
            # for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            tot_input = 0
            for input_nodes, output_nodes, blocks in dataloader:
                print(blocks)
                tot_input += input_nodes.shape[0]
                t1 = time.time()
                a += t1-t0
                # torch.cuda.reset_peak_memory_stats()
                th.cuda.empty_cache()
                t2 = time.time()
                e += t2-t1

                block = blocks[0].to(device)
                if use_uva:
                    h = gather_pinned_tensor_rows(x, input_nodes)
                else:
                    h = x[input_nodes].to(device)
                t3 = time.time()
                b += t3-t2

                h = layer(block, h)
                if l == self.num_layers - 1:
                    logits = h.mean(1)
                    t4 = time.time()
                    c += t4-t3
                    y[output_nodes] = logits.cpu()
                else:
                    h = h.flatten(1)
                    t4 = time.time()
                    c += t4-t3
                    y[output_nodes] = h.cpu()
                t0 = time.time()
                d += t0-t4
                memorys.append(torch.cuda.max_memory_allocated() // 1024 ** 2)
            if use_uva:
                unpin_memory_inplace(x)
            x = y
            # print(memorys)
            print(a, b, c, d, e)
            print("tot:", tot_input)
            print(time.time()- sss)
        # print("memory: ", torch.cuda.max_memory_allocated() // 1024 ** 2)
        return y
