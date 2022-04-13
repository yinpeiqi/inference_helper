import pynvml
import torch


def get_memory_in_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info

class AutoTurner:
    def __init__(self):
        self.set_free()

    def set_free(self):
        info = get_memory_in_MiB()
        self.free_memory = info.free * 0.9 // 1024 ** 2

    def search(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        max_memory_allocated = torch.cuda.max_memory_allocated() // 1024 ** 2
        increase_rate = self.free_memory / max_memory_allocated
        curr_node = int(curr_node * increase_rate)
        curr_edge = int(curr_edge * increase_rate)
        return curr_node, curr_edge

    def break_peak(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        return curr_node // 2, curr_edge // 2
