import dgl
import numpy as np
import torch
import torch.nn as nn
import tqdm
import os
import gc
import time

from .profiler import Profiler
from .auto_tuner import get_auto_tuner
from .function_generator import FunctionGenerator
from .data_manager import DataManager
from .custom_dataloader import CustomDataloader
from .utils import get_new_arg_input, update_ret_output

class HeteroInferenceHelper():
    def __init__(self, module: nn.Module, device, use_uva = False, debug = False):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._use_uva = use_uva
        self.m = module
        self._function_generator = FunctionGenerator(module, debug)
        self._traced = self._function_generator.traced
        self._schema = self._function_generator.get_schema()
        self._funcs = self._function_generator.get_funcs()
        self._debug = debug

    def inference(
        self,
        hg: dgl.DGLHeteroGraph,
        batch_size: int,
        num_workers: int,
        embedding_layer: nn.Module,
        device: torch.device,
    ):
        for i, func in enumerate(self._funcs):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            # dataloader = dgl.dataloading.DataLoader(
            #     hg,
            #     {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
            #     sampler,
            #     batch_size=batch_size,
            #     shuffle=False,
            #     drop_last=False,
            #     num_workers=num_workers,
            # )

            dataloader = CustomDataloader(
                hg,
                {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                2000,
                use_uva=self._use_uva,
                shuffle=False)

            if i < self.m._num_layers - 1:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._hidden_feats) for ntype in hg.ntypes}
            else:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._out_feats) for ntype in hg.ntypes}

            max_memory = 0
            memorys = []
            nodes = []
            auto_tuner = get_auto_tuner(self._device)
            profiler = Profiler()
            profiler.record_and_reset()
            for in_nodes, out_nodes, blocks in dataloader:
                profiler.tag()
                try:
                    auto_tuner.reset_state()
                    torch.cuda.empty_cache()
                    auto_tuner.set_free()
                    profiler.record_name("total input nodes", blocks[0].num_src_nodes())

                    in_nodes = {rel: nid for rel, nid in in_nodes.items()}
                    out_nodes = {rel: nid for rel, nid in out_nodes.items()}
                    block = blocks[0].to(device)

                    if i == 0:
                        h = embedding_layer(in_nodes=in_nodes, device=device)
                    else:
                        h = {ntype: x[ntype][in_nodes[ntype]].to(device) for ntype in hg.ntypes}

                    args = ()
                    for ntype in hg.ntypes:
                        args = args + (h[ntype],)

                    profiler.tag()
                    hx = func(block, *args)

                    h_dict = {}
                    for j, ntype in enumerate(hg.ntypes):
                        h_dict[ntype] = hx[j]
                    profiler.tag()
                
                    # if self._debug:
                    #     print(blocks[0], "; max memory = ", torch.cuda.max_memory_allocated() // 1024 ** 2, "MB")

                    for ntype in h_dict:
                        if ntype in out_nodes:
                            y[ntype][out_nodes[ntype]] = h_dict[ntype].cpu()

                    del hx, h_dict

                    profiler.tag()
                    auto_tuner.set_max()
                    nxt_max_node, _ = auto_tuner.search(blocks[0])
                    memorys.append(torch.cuda.max_memory_allocated() // 1024 ** 2)
                    nodes.append(blocks[0].num_dst_nodes())
                    max_memory = max(torch.cuda.max_memory_allocated() // 1024 ** 2, max_memory)

                except Exception as e:
                    print(e)
                    profiler.tag()
                    nxt_max_node, _ = auto_tuner.break_peak(blocks[0])
                    dataloader.reset_batch_node(out_nodes.shape[0])
                    gc.collect()

                finally:
                    dataloader.modify_max_node(nxt_max_node)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    profiler.record_and_reset()

            print(memorys)
            print(nodes)
            profiler.show()
            x = y

        return x

