import pynvml
import gc
import dgl
import torch
from .custom_dataloader import CustomDataset
from .utils import get_new_arg_input
import time


def get_memory_in_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info

def get_next_arg_input(zero_args, graph, device):
    zero_args = list(zero_args)
    for i in range(len(zero_args)):
        if isinstance(zero_args[i], torch.Tensor):
            last_shape = zero_args[i].shape[0]
            curr_shape = graph.number_of_src_nodes()
            if last_shape >= curr_shape:
                zero_args[i] = zero_args[i][:curr_shape]
            else:
                zero_args[i] = torch.cat((zero_args[i], torch.zeros((curr_shape - last_shape, *zero_args[i].shape[1:])).to(device)), 0)
        elif isinstance(zero_args[i], dgl.DGLHeteroGraph):
            zero_args[i] = graph.to(device)
    return tuple(zero_args)

class AutoTunner:
    def __init__(self, device, start_edge_count = 10000):
        self.device = device
        self.start_edge_count = start_edge_count

    def search(self, graph, arg2val_map, layer, func):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        nids = torch.arange(graph.number_of_nodes()).to(graph.device)
        zero_args = ()
        def test(sampler, nids, zero_args, edge_count):
            try:
                custom_dataset = CustomDataset(edge_count, graph, nids)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    custom_dataset,
                    sampler,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)

                for input_nodes, output_nodes, blocks in dataloader:
                    tt = time.time()
                    if zero_args != ():
                        ttt = get_next_arg_input(zero_args, blocks[0], self.device)
                    t = time.time()
                    print("first:", t-tt)
                    zero_args = get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0], self.device)
                    tx = time.time()
                    print("second:", tx-t)
                    print(blocks[0])
                    func(*zero_args)
                    return True, zero_args

            except Exception as e:
                print(e)
                return False, zero_args

        max_edge = graph.num_edges()
        info = get_memory_in_MiB()
        free_memory = info.free * 0.9 // 1024 ** 2

        edge_count = self.start_edge_count
        torch.cuda.reset_max_memory_allocated()
        while edge_count < max_edge:
            flag, zero_args = test(sampler, nids, zero_args, edge_count)
            if not flag:
                break
            max_memory_allocated = torch.cuda.max_memory_allocated() // 1024 ** 2
            print("max memory allocated:", max_memory_allocated)
            increase_rate = free_memory / max_memory_allocated
            if increase_rate < 1.2:
                edge_count *= 2
                break
            edge_count *= max(2, increase_rate)
            edge_count = int(edge_count)

        edge_count /= 2
        print(torch.cuda.memory_allocated() // 1024 ** 2)
        gc.collect()
        torch.cuda.empty_cache()
        return edge_count
