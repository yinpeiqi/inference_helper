import pynvml
import dgl
import torch
from .custom_dataloader import CustomDataset
from .utils import get_new_arg_input

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


class AutoTunner:
    def __init__(self, device):
        self.device = device

    def search(self, graph, arg2val_map, layer, func):
        max_edge = graph.num_edges()

        def test(edge_count):
            try:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                nids = torch.arange(graph.number_of_nodes()).to(graph.device)
                custom_dataset = CustomDataset(edge_count, graph, nids)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    custom_dataset,
                    sampler,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0)

                for input_nodes, output_nodes, blocks in dataloader:
                    func(*get_new_arg_input(layer.inputs, arg2val_map, input_nodes, blocks[0], self.device))
                    return True

            except Exception as e:
                print(e)
                return False

        edge_count = 100
        while test(edge_count) and edge_count < max_edge:
            print(edge_count, get_memory_free_MiB(0))
            print("memory allocated:", torch.cuda.memory_allocated() // 1024 // 1024)
            print("max memory allocated:", torch.cuda.max_memory_allocated() // 1024 // 1024)
            print("memory reserved:", torch.cuda.memory_reserved() // 1024 // 1024)
            print("max memory reserved:", torch.cuda.max_memory_reserved() // 1024 // 1024)
            # print(torch.cuda.memory_stats())
            torch.cuda.empty_cache()
            edge_count *= 2
        edge_count /= 2
        return edge_count
