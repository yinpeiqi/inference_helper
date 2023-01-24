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
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
import tqdm

class HeteroInferenceHelper():
    def __init__(self, module: nn.Module, device, use_uva = False, debug = False, free_rate = 0.8):
        # add a '_' in order not crash with the origin one.
        self._device = device
        self._use_uva = use_uva
        self.m = module
        self._free_rate = free_rate
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
                    dataloader.reset_batch_node(blocks[0].num_dst_nodes())
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

    def static_inference_240m(
        self,
        hg: dgl.DGLHeteroGraph,
        author_feats, institution_feats, paper_feats,
        dataset,
        device: torch.device, batch_size
    ):
        for i, func in enumerate(self._funcs):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                # num_workers=num_workers,
            )

            if i < self.m._num_layers - 1:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._hidden_feats) for ntype in hg.ntypes}
            else:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._out_feats) for ntype in hg.ntypes}

            for ntype in hg.ntypes:
                pin_memory_inplace(y[ntype])

            for in_nodes, out_nodes, blocks in tqdm.tqdm(dataloader):
                in_nodes = {rel: nid.to(device) for rel, nid in in_nodes.items()}
                out_nodes = {rel: nid.to(device) for rel, nid in out_nodes.items()}
                block = blocks[0].to(device)

                if i == 0:
                    new_feat = gather_pinned_tensor_rows(author_feats, in_nodes['author'])
                    new_feat2 = gather_pinned_tensor_rows(institution_feats, in_nodes['institution'])
                    new_feat3 = gather_pinned_tensor_rows(paper_feats, in_nodes['paper'])
                else:
                    new_feat = gather_pinned_tensor_rows(x['author'], in_nodes['author'])
                    new_feat2 = gather_pinned_tensor_rows(x['institution'], in_nodes['institution'])
                    new_feat3 = gather_pinned_tensor_rows(x['paper'], in_nodes['paper'])

                hx = func(block, new_feat, new_feat2, new_feat3)

                h_dict = {}
                for j, ntype in enumerate(hg.ntypes):
                    h_dict[ntype] = hx[j]
                
                for ntype in h_dict:
                    if ntype in out_nodes:
                        y[ntype][out_nodes[ntype]] = h_dict[ntype].cpu()

                del new_feat, new_feat2, new_feat3, h_dict

            x = y

        return x

    def inference_240m(
        self,
        hg: dgl.DGLHeteroGraph,
        author_feats, institution_feats, paper_feats,
        dataset,
        device: torch.device,
    ):
        for i, func in enumerate(self._funcs):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = CustomDataloader(
                hg,
                {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                200,
                use_uva=True,
                shuffle=True, device=device, num_workers=0)

            if i < self.m._num_layers - 1:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._hidden_feats) for ntype in hg.ntypes}
            else:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._out_feats) for ntype in hg.ntypes}

            for ntype in hg.ntypes:
                pin_memory_inplace(y[ntype])

            max_memory = 0
            memorys = []
            nodes = []
            auto_tuner = get_auto_tuner(self._device)
            profiler = Profiler()
            profiler.record_and_reset()
            
            break_points = [hg.num_nodes(ntype) for ntype in hg.ntypes] + [2e18]
            for typs in range(1, len(hg.ntypes) - 1):
                break_points[typs] += break_points[typs-1]
            curr_nodes, curr_type = 0, 0

            pbar = tqdm.tqdm(total=hg.number_of_nodes())
            for in_nodes, out_nodes, blocks in dataloader:
                profiler.tag()
                try:
                    auto_tuner.reset_state()
                    torch.cuda.empty_cache()
                    auto_tuner.set_free(self._free_rate)
                    profiler.record_name("total input nodes", blocks[0].num_src_nodes())

                    in_nodes = {rel: nid for rel, nid in in_nodes.items()}
                    out_nodes = {rel: nid for rel, nid in out_nodes.items()}
                    block = blocks[0].to(device)

                    if i == 0:
                        new_feat = gather_pinned_tensor_rows(author_feats, in_nodes['author'])
                        new_feat2 = gather_pinned_tensor_rows(institution_feats, in_nodes['institution'])
                        new_feat3 = gather_pinned_tensor_rows(paper_feats, in_nodes['paper'])
                    else:
                        new_feat = gather_pinned_tensor_rows(x['author'], in_nodes['author'])
                        new_feat2 = gather_pinned_tensor_rows(x['institution'], in_nodes['institution'])
                        new_feat3 = gather_pinned_tensor_rows(x['paper'], in_nodes['paper'])

                    profiler.tag()
                    hx = func(block, new_feat, new_feat2, new_feat3)

                    h_dict = {}
                    for j, ntype in enumerate(hg.ntypes):
                        h_dict[ntype] = hx[j]
                    profiler.tag()
                
                    if self._debug:
                        print('\n', blocks[0].num_src_nodes(), blocks[0].num_dst_nodes(), blocks[0].num_edges(), "; max memory = ", torch.cuda.max_memory_allocated() // 1024 ** 2, "MB")

                    for ntype in h_dict:
                        if ntype in out_nodes:
                            y[ntype][out_nodes[ntype]] = h_dict[ntype].cpu()

                    del new_feat, new_feat2, new_feat3, h_dict

                    profiler.tag()
                    auto_tuner.set_max()
                    nxt_max_node, _ = auto_tuner.search(blocks[0])
                    curr_nodes += blocks[0].num_dst_nodes()
                    if curr_nodes >= break_points[curr_type]:
                        nxt_max_node = 200
                        curr_type += 1
                    elif curr_nodes + nxt_max_node > break_points[curr_type]:
                        nxt_max_node = break_points[curr_type] - curr_nodes
                    print(curr_nodes, break_points[curr_type], nxt_max_node)

                    memorys.append(torch.cuda.max_memory_allocated() // 1024 ** 2)
                    nodes.append(blocks[0].num_dst_nodes())
                    max_memory = max(torch.cuda.max_memory_allocated() // 1024 ** 2, max_memory)
                    pbar.update(blocks[0].num_dst_nodes())

                except Exception as e:
                    print(e)
                    profiler.tag()
                    nxt_max_node, _ = auto_tuner.break_peak(blocks[0])
                    dataloader.reset_batch_node(blocks[0].num_dst_nodes())
                    gc.collect()

                finally:
                    dataloader.modify_max_node(nxt_max_node)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    profiler.record_and_reset()

            pbar.close()
            print(memorys)
            print(nodes)
            profiler.show()
            x = y

        return x

    def static_inference(
        self,
        hg: dgl.DGLHeteroGraph,
        batch_size: int,
        num_workers: int,
        embedding_layer: nn.Module,
        device: torch.device,
    ):
        for i, func in enumerate(self._funcs):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                # num_workers=num_workers,
            )

            if i < self.m._num_layers - 1:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._hidden_feats) for ntype in hg.ntypes}
            else:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self.m._out_feats) for ntype in hg.ntypes}

            profiler = Profiler()
            profiler.record_and_reset()
            for in_nodes, out_nodes, blocks in tqdm.tqdm(dataloader):
                profiler.tag()
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
            
                for ntype in h_dict:
                    if ntype in out_nodes:
                        y[ntype][out_nodes[ntype]] = h_dict[ntype].cpu()

                profiler.tag()
                del hx, h_dict
                torch.cuda.empty_cache()
                profiler.record_and_reset()

            x = y
            profiler.show()

        return x
