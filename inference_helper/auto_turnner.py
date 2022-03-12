import pynvml
import torch


def get_memory_in_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info

class AutoTunner:
    def __init__(self, device, start_edge_count = 10000):
        self.device = device
        self.edge_count = start_edge_count
        self.peak_reached = False
        info = get_memory_in_MiB()
        self.free_memory = info.free * 0.9 // 1024 ** 2

    def search(self):
        if not self.peak_reached:
            max_memory_allocated = torch.cuda.max_memory_allocated() // 1024 ** 2
            increase_rate = self.free_memory / max_memory_allocated
            self.edge_count *= increase_rate
            self.edge_count = int(self.edge_count)
        else:
            self.edge_count *= 1.04
        return self.edge_count

    def break_peak(self):
        self.peak_reached = True
        self.edge_count //= 2
        return self.edge_count
