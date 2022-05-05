import torch.fx
import torch as th
import torch.nn as nn
import gc
import time

import dgl
from dgl.nn import GATConv
from inference_helper.profiler import Profiler
import tqdm
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
from inference_helper.utils import update_out_in_chunks
import numpy as np

from . import cache

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

    def inference(self, g, batch_size, device, x, nids, use_uva = False):
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
            
            profiler = Profiler()
            profiler.record_and_reset()
            # for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                torch.cuda.empty_cache()
                profiler.tag()
                profiler.record_name("total input nodes", input_nodes.shape[0])
                # print(blocks)
                block = blocks[0].to(device)
                if use_uva:
                    h = gather_pinned_tensor_rows(x, input_nodes)
                else:
                    h = x[input_nodes].to(device)
                profiler.tag()
                # print(h.shape, "%.2f"%(profiler.last()), "s;", "%.2f"%(h.shape[0]*h.shape[1]*4/1000**3), "GB;", "%.2f"%(h.shape[0]*h.shape[1]*4 / profiler.last() / 1000**3), "GB/s")

                h = layer(block, h)
                if l == self.num_layers - 1:
                    logits = h.mean(1)
                    profiler.tag()
                    y[output_nodes] = logits.cpu()
                else:
                    h = h.flatten(1)
                    profiler.tag()
                    update_out_in_chunks(y, output_nodes, h)
                profiler.tag()

                th.cuda.empty_cache()
                profiler.record_and_reset()
                memorys.append(torch.cuda.max_memory_allocated() // 1024 ** 2)
            if use_uva:
                unpin_memory_inplace(x)
            x = y
            # print(memorys)
            profiler.show()
        # print("memory: ", torch.cuda.max_memory_allocated() // 1024 ** 2)
        return y
    
    def inference_with_cache(self, g, batch_size, device, x, args):

        if args.debug:
            degs = g.out_degrees().numpy()
            degs = np.sort(degs)[::-1]
            len = degs.shape[0] - degs.shape[0] % 20
            degs = degs[:len]
            total_degs = np.sum(degs)
            sdegs = [np.sum(arr) for arr in np.split(degs, 20)]
            sum = 0
            for i, d in enumerate(sdegs):
                sum += d
                print('{}% nodes, {}% edges'.format(i * 5 + 5, sum / total_degs))

        for k in list(g.ndata.keys()):
            g.ndata.pop(k)
        for k in list(g.edata.keys()):
            g.edata.pop(k)

        torch.cuda.reset_peak_memory_stats()
        cache_size_gb = args.cache_size
        start = time.time()
        max_feat_dim = x.shape[1]
        for l in range(self.num_layers - 1):
            max_feat_dim = max(max_feat_dim, self.heads[l] * self.hidden_features)
        host_indices, cached_indices, node_pos_map = cache.get_node_pos_map(g, max_feat_dim, cache_size_gb, device)



        # host_in, cache_in, node_pos_map = cache.cache_feat(g, x, 3, device)
        end = time.time()
        print('Inference cache preprocessing time:', end - start)
        for l, layer in enumerate(self.gat_layers):
            gc.collect()
            th.cuda.empty_cache()
            host_in, cache_in = cache.cache_feat_v2(x, host_indices, cached_indices, device)
            if host_in.shape[0] > 0:
                pin_memory_inplace(host_in)
            if l != self.num_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.heads[l] * self.hidden_features)
            else:
                y = th.zeros(g.number_of_nodes(), self.out_features)

            nids = th.arange(g.number_of_nodes()).to(g.device)
            nids = nids.to(device)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, nids, sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                use_uva=True,
                device=device,
                num_workers=0)
            memorys = []

            if args.debug:
                total_hit = 0
                cache_hit = 0
                debug_node_pos_map = node_pos_map.cpu().numpy()
            
            profiler = Profiler()
            profiler.record_and_reset()
            # for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                torch.cuda.empty_cache()
                profiler.tag()
                profiler.record_name("total input nodes", input_nodes.shape[0])
                # print(blocks)
                block = blocks[0].to(device)
                h = cache.read_feat(host_in, cache_in, input_nodes, node_pos_map)
                if args.debug:
                    total_hit += input_nodes.shape[0]
                    cache_hit += cache.get_debug_cache_hit(input_nodes, debug_node_pos_map)

                profiler.tag()
                # print(h.shape, "%.2f"%(profiler.last()), "s;", "%.2f"%(h.shape[0]*h.shape[1]*4/1000**3), "GB;", "%.2f"%(h.shape[0]*h.shape[1]*4 / profiler.last() / 1000**3), "GB/s")

                h = layer(block, h)
                if l == self.num_layers - 1:
                    logits = h.mean(1)
                    profiler.tag()
                    y[output_nodes] = logits.cpu()
                else:
                    h = h.flatten(1)
                    profiler.tag()
                    update_out_in_chunks(y, output_nodes, h)
                profiler.tag()

                th.cuda.empty_cache()
                profiler.record_and_reset()
                memorys.append(torch.cuda.max_memory_allocated() // 1024 ** 2)
            if host_in.shape[0] > 0:
                unpin_memory_inplace(host_in)
            x = y
            # print(memorys)
            profiler.show()
            if args.debug:
                print('Layer {}, cached hit {}, total {}, hit rate {}, cache size {} GB'.format(l, cache_hit, total_hit, cache_hit / total_hit, cache_size_gb))
        # print("memory: ", torch.cuda.max_memory_allocated() // 1024 ** 2)
        return y

