import pynvml
import torch

def get_auto_turner(device):
    if isinstance(device, torch.device):
        device = device.type
    if 'cuda' in device:
        return GPUAutoTurner()
    elif 'cpu' in device:
        return CPUAutoTurner()
    else:
        raise NotImplementedError("Not implement Auto Turner for device: {}.".format(device))


class AutoTurnerBase:
    def __init__(self):
        self.free_memory = 0
        self.set_free()

    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError

    def search(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        increase_rate = self.free_memory / self.get_max()
        curr_node = int(curr_node * increase_rate)
        curr_edge = int(curr_edge * increase_rate)
        return curr_node, curr_edge

    def break_peak(self, g):
        curr_node = g.number_of_dst_nodes()
        curr_edge = g.num_edges()
        return curr_node // 2, curr_edge // 2


class GPUAutoTurner(AutoTurnerBase):
    def set_free(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.free_memory = info.free * 0.9

    def get_max(self):
        return torch.cuda.max_memory_allocated()


class CPUAutoTurner(AutoTurnerBase):
    def set_free(self):
        raise NotImplementedError

    def get_max(self):
        raise NotImplementedError
